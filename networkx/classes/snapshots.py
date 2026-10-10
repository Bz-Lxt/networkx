"""Versioned graphs with near-constant-time read-only snapshots.

The :class:`Graph`, :class:`DiGraph`, :class:`MultiGraph` and
:class:`MultiDiGraph` classes store their data in plain dictionaries, so a
stable copy of a large graph (``G.copy()``) costs time and memory proportional
to the graph size.  The ``Versioned*`` graph classes in this module keep the
history of every storage dictionary in *layers* instead of overwriting it.

Taking a snapshot pins the current generation in ``O(1)`` time and without
copying any graph data::

    >>> G = nx.VersionedGraph()
    >>> G.add_edge(1, 2, weight=7)
    >>> snap = G.snapshot()
    >>> G.add_edge(2, 3, weight=4)
    >>> list(snap.edges)
    [(1, 2)]
    >>> list(G.edges)
    [(1, 2), (2, 3)]

A snapshot contains the structure (nodes and edges) together with the
top-level graph attribute mapping as they were when it was taken.  Later
updates to the live graph never change an existing snapshot, while the
ordinary views of the live graph stay live (dynamic).  Snapshots are frozen,
read-only graphs and support the usual view machinery such as
:func:`~networkx.classes.graphviews.subgraph_view` and
:func:`~networkx.classes.graphviews.reverse_view`.

Old versions are only kept for as long as they are needed.  When the last
snapshot of a generation is released (deleted, or closed via
:meth:`close <_SnapshotGraph.close>`), history that no snapshot can observe is
collapsed and becomes eligible for garbage collection.
"""

import weakref
from collections.abc import Mapping, MutableMapping

import networkx as nx
from networkx.classes.digraph import DiGraph
from networkx.classes.graph import Graph
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

__all__ = [
    "VersionedGraph",
    "VersionedDiGraph",
    "VersionedMultiGraph",
    "VersionedMultiDiGraph",
    "snapshot",
]

# Marker stored in a layer to record that a key was deleted in that generation.
_TOMBSTONE = object()
# Marker returned by a version lookup when no layer (visible to the reader)
# contains the key at all.
_ABSENT = object()
# Structural event kinds replayed to recover key order at a generation.
_INSERT = 1
_DELETE = 0

_FACTORY_NAMES = (
    "node_dict_factory",
    "node_attr_dict_factory",
    "adjlist_outer_dict_factory",
    "adjlist_inner_dict_factory",
    "edge_attr_dict_factory",
    "edge_key_dict_factory",
    "graph_attr_dict_factory",
)


class _VersionStore:
    """Shared bookkeeping for all storage dicts of one versioned graph."""

    __slots__ = ("_gen", "_pins", "_containers", "_mutation_depth", "_prune_pending")

    def __init__(self):
        # Next generation id.  A snapshot pins the current id, then this is
        # advanced, so a layer is visible to a pin ``g`` iff its tag is <= g.
        self._gen = 0
        self._pins = set()
        self._containers = weakref.WeakSet()
        # Pruning is deferred while a write is in flight.  Cyclic garbage
        # collection can release snapshot pins at arbitrary allocation points
        # (including allocations made by a storage mutation), so collapsing
        # layers immediately could tear an in-progress write.
        self._mutation_depth = 0
        self._prune_pending = False

    def create_dict(self):
        vd = _VersionedDict(self)
        self._containers.add(vd)
        return vd

    def take_pin(self):
        pin = self._gen
        self._gen += 1
        self._pins.add(pin)
        return pin

    def begin_mutation(self):
        self._mutation_depth += 1

    def end_mutation(self):
        self._mutation_depth -= 1
        if self._mutation_depth == 0 and self._prune_pending:
            self._prune_pending = False
            self.prune()

    def release_pin(self, pin):
        self._pins.discard(pin)
        if self._mutation_depth:
            self._prune_pending = True
        else:
            self.prune()

    def prune(self):
        oldest = min(self._pins) if self._pins else None
        for vd in self._containers:
            vd._prune(oldest, self._gen)


class _VersionedDict(MutableMapping):
    """A dict-like mapping with layered, snapshot-able history.

    Each layer is ``[tag, data, events]`` where ``data`` holds the newest
    value (or :data:`_TOMBSTONE`) for keys written in that generation and
    ``events`` records structural inserts/deletes so that the key order at
    any pinned generation can be replayed (plain dict semantics: updating a
    key keeps its position, deleting and re-adding moves it to the end).
    """

    __slots__ = ("_store", "_layers", "_readers", "__weakref__")

    def __init__(self, store):
        self._store = store
        self._layers = [[store._gen, {}, []]]
        # pin generation -> _VersionedDictReader observing that generation
        self._readers = {}

    # ------------------------------------------------------------------
    # version mechanics
    # ------------------------------------------------------------------
    def _visible_layers(self, pin):
        if pin is None:
            return self._layers
        return [layer for layer in self._layers if layer[0] <= pin]

    def _writable_layer(self):
        """Return the layer that live writes go into, opening one if the
        current top layer is pinned by a live snapshot."""
        layers = self._layers
        pins = self._store._pins
        if pins and layers[-1][0] <= max(pins):
            layers.append([self._store._gen, {}, []])
        return layers[-1]

    def _raw_lookup(self, key, pin):
        """Newest raw value visible at ``pin``; ``_TOMBSTONE``/``_ABSENT``."""
        for layer in reversed(self._visible_layers(pin)):
            data = layer[1]
            if key in data:
                return data[key]
        return _ABSENT

    @staticmethod
    def _replay(layers):
        """Replay structural events to get live keys in capture order."""
        order = []
        alive = set()
        for _tag, _data, events in layers:
            for op, key in events:
                if op == _INSERT:
                    if key in alive:
                        order.remove(key)
                    else:
                        alive.add(key)
                    order.append(key)
                elif key in alive:  # _DELETE
                    alive.discard(key)
                    order.remove(key)
        return order

    def _order(self, pin):
        layers = self._layers
        if pin is None and len(layers) == 1:
            return list(layers[0][1])
        if pin is not None and len(layers) == 1 and layers[0][0] <= pin:
            return list(layers[0][1])
        return self._replay(self._visible_layers(pin))

    def _is_alive(self, key):
        raw = self._raw_lookup(key, None)
        return raw is not _ABSENT and raw is not _TOMBSTONE

    def _materialize(self, pin, tag):
        """Collapse all layers visible at ``pin`` into one fresh layer."""
        data = {}
        events = []
        for key in self._order(pin):
            data[key] = self._raw_lookup(key, pin)
            events.append((_INSERT, key))
        return [tag, data, events]

    def _prune(self, oldest, gen):
        # Readers belonging to released pins disappear.
        if self._readers:
            self._readers = {
                pin: reader
                for pin, reader in self._readers.items()
                if pin in self._store._pins
            }
        layers = self._layers
        if oldest is None:
            # No snapshots left: collapse everything, reclaim old versions.
            if len(layers) > 1:
                self._layers = [self._materialize(None, gen)]
            else:
                layers[0][0] = gen
            return
        idx = -1
        for i, layer in enumerate(layers):
            if layer[0] <= oldest:
                idx = i
            else:
                break
        if idx > 0:
            merged = self._materialize(oldest, oldest)
            self._layers = [merged, *layers[idx + 1:]]

    def reader(self, pin):
        """Return the read-only mapping pinned at generation ``pin``."""
        reader = self._readers.get(pin)
        if reader is None:
            reader = _VersionedDictReader(self, pin)
            self._readers[pin] = reader
        return reader

    def _value_at(self, key, pin):
        raw = self._raw_lookup(key, pin)
        if raw is _ABSENT or raw is _TOMBSTONE:
            raise KeyError(key)
        if isinstance(raw, _VersionedDict):
            return raw.reader(pin)
        return raw

    # ------------------------------------------------------------------
    # MutableMapping interface (the live side)
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        raw = self._raw_lookup(key, None)
        if raw is _ABSENT or raw is _TOMBSTONE:
            raise KeyError(key)
        return raw

    def __setitem__(self, key, value):
        store = self._store
        store.begin_mutation()
        try:
            layer = self._writable_layer()
            alive = self._is_alive(key)
            layer[1][key] = value
            if not alive:
                layer[2].append((_INSERT, key))
        finally:
            store.end_mutation()

    def __delitem__(self, key):
        store = self._store
        store.begin_mutation()
        try:
            layer = self._writable_layer()
            if not self._is_alive(key):
                raise KeyError(key)
            layer[1][key] = _TOMBSTONE
            layer[2].append((_DELETE, key))
        finally:
            store.end_mutation()

    def __iter__(self):
        return iter(self._order(None))

    def __len__(self):
        layers = self._layers
        if len(layers) == 1:
            return len(layers[0][1])
        return len(self._order(None))

    def __contains__(self, key):
        raw = self._raw_lookup(key, None)
        return raw is not _ABSENT and raw is not _TOMBSTONE

    def clear(self):
        store = self._store
        store.begin_mutation()
        try:
            layer = self._writable_layer()
            for key in self._order(None):
                layer[1][key] = _TOMBSTONE
                layer[2].append((_DELETE, key))
        finally:
            store.end_mutation()

    def setdefault(self, key, default=None):
        raw = self._raw_lookup(key, None)
        if raw is not _ABSENT and raw is not _TOMBSTONE:
            return raw
        self[key] = default
        return default

    def copy(self):
        return dict(self)

    def __eq__(self, other):
        if isinstance(other, (_VersionedDict, _VersionedDictReader)):
            if len(self) != len(other):
                return False
            for key in self:
                if key not in other or self[key] != other[key]:
                    return False
            return True
        if isinstance(other, dict):
            return dict(self) == other
        return NotImplemented

    # Identity-based hashing (equality is content based, but containers are
    # always referenced as distinct objects; needed for weak registries).
    __hash__ = object.__hash__

    def __repr__(self):
        return repr(dict(self))

    def __reduce__(self):
        return (dict, (dict(self),))


class _VersionedDictReader(Mapping):
    """Read-only view of a :class:`_VersionedDict` pinned at one generation."""

    __slots__ = ("_vd", "_pin", "__weakref__")

    def __init__(self, vd, pin):
        self._vd = vd
        self._pin = pin

    def __getitem__(self, key):
        return self._vd._value_at(key, self._pin)

    def __iter__(self):
        return iter(self._vd._order(self._pin))

    def __len__(self):
        layers = self._vd._layers
        if len(layers) == 1 and layers[0][0] <= self._pin:
            return len(layers[0][1])
        return len(self._vd._order(self._pin))

    def __contains__(self, key):
        raw = self._vd._raw_lookup(key, self._pin)
        return raw is not _ABSENT and raw is not _TOMBSTONE

    def copy(self):
        return dict(self)

    def __eq__(self, other):
        if isinstance(other, (_VersionedDict, _VersionedDictReader)):
            if len(self) != len(other):
                return False
            for key in self:
                if key not in other or self[key] != other[key]:
                    return False
            return True
        if isinstance(other, dict):
            return dict(self) == other
        return NotImplemented

    __hash__ = None

    def __repr__(self):
        return repr(dict(self))

    def __reduce__(self):
        return (dict, (dict(self),))


# ----------------------------------------------------------------------
# Frozen snapshot graph classes
# ----------------------------------------------------------------------
def _snapshot_copy(self, as_view=False):
    # Copies of a snapshot are ordinary, editable networkx graphs that do
    # not retain the version store.
    plain_class = self._plain_class
    if as_view is True:
        return nx.graphviews.generic_graph_view(self, plain_class)
    G = plain_class()
    G.graph.update(self.graph)
    G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
    adjacency = self._succ if self.is_directed() else self._adj
    if self.is_multigraph():
        G.add_edges_from(
            (u, v, key, datadict.copy())
            for u, nbrs in adjacency.items()
            for v, keydict in nbrs.items()
            for key, datadict in keydict.items()
        )
    else:
        G.add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in adjacency.items()
            for v, datadict in nbrs.items()
        )
    return G


class _SnapshotGraph(Graph):
    """A frozen graph reporting storage pinned at one version."""

    __slots__ = ("_version_pin", "_version_finalizer")

    _plain_class = Graph
    copy = _snapshot_copy

    def close(self):
        """Release this snapshot, allowing its versions to be reclaimed."""
        finalizer = self.__dict__.get("_version_finalizer")
        if finalizer is None:
            return
        self.__dict__["_version_finalizer"] = None
        self.__dict__["_version_pin"] = None
        finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class _SnapshotDiGraph(DiGraph):
    """A frozen directed graph reporting storage pinned at one version."""

    __slots__ = ("_version_pin", "_version_finalizer")

    _plain_class = DiGraph
    copy = _snapshot_copy
    close = _SnapshotGraph.close
    __enter__ = _SnapshotGraph.__enter__
    __exit__ = _SnapshotGraph.__exit__


class _SnapshotMultiGraph(MultiGraph):
    """A frozen multigraph reporting storage pinned at one version."""

    __slots__ = ("_version_pin", "_version_finalizer")

    _plain_class = MultiGraph
    copy = _snapshot_copy
    close = _SnapshotGraph.close
    __enter__ = _SnapshotGraph.__enter__
    __exit__ = _SnapshotGraph.__exit__


class _SnapshotMultiDiGraph(MultiDiGraph):
    """A frozen multidigraph reporting storage pinned at one version."""

    __slots__ = ("_version_pin", "_version_finalizer")

    _plain_class = MultiDiGraph
    copy = _snapshot_copy
    close = _SnapshotGraph.close
    __enter__ = _SnapshotGraph.__enter__
    __exit__ = _SnapshotGraph.__exit__


# ----------------------------------------------------------------------
# Versioned graph classes
# ----------------------------------------------------------------------
class VersionedGraph(Graph):
    """:class:`~networkx.Graph` supporting near-constant-time snapshots.

    See the module docstring of :mod:`networkx.classes.snapshots` for an
    overview.  Use :meth:`snapshot` to capture a frozen, read-only graph that
    remains stable while this graph keeps being updated.
    """

    _snapshot_class = _SnapshotGraph

    def __init__(self, incoming_graph_data=None, **attr):
        store = _VersionStore()
        self._version_store = store
        for factory_name in _FACTORY_NAMES:
            setattr(self, factory_name, store.create_dict)
        super().__init__(incoming_graph_data, **attr)

    def snapshot(self):
        """Return a frozen read-only snapshot of the current graph state.

        The snapshot is taken in constant time.  Subsequent updates to this
        graph do not affect the snapshot.  When the snapshot is garbage
        collected (or :meth:`closed <_SnapshotGraph.close>`), versions that no
        remaining snapshot observes are reclaimed.

        Returns
        -------
        snapshot : frozen Graph
            A read-only graph of the same type as this graph.
        """
        store = self._version_store
        pin = store.take_pin()
        new = self._snapshot_class()
        new.graph = self.graph.reader(pin)
        new._node = self._node.reader(pin)
        new._adj = self._adj.reader(pin)
        nx.freeze(new)
        new.__dict__["_version_pin"] = pin
        new.__dict__["_version_finalizer"] = weakref.finalize(
            new, store.release_pin, pin
        )
        return new


class VersionedDiGraph(DiGraph):
    """:class:`~networkx.DiGraph` supporting near-constant-time snapshots."""

    _snapshot_class = _SnapshotDiGraph

    def __init__(self, incoming_graph_data=None, **attr):
        store = _VersionStore()
        self._version_store = store
        for factory_name in _FACTORY_NAMES:
            setattr(self, factory_name, store.create_dict)
        super().__init__(incoming_graph_data, **attr)

    def snapshot(self):
        store = self._version_store
        pin = store.take_pin()
        new = self._snapshot_class()
        new.graph = self.graph.reader(pin)
        new._node = self._node.reader(pin)
        new._succ = self._succ.reader(pin)
        new._pred = self._pred.reader(pin)
        nx.freeze(new)
        new.__dict__["_version_pin"] = pin
        new.__dict__["_version_finalizer"] = weakref.finalize(
            new, store.release_pin, pin
        )
        return new


class VersionedMultiGraph(MultiGraph):
    """:class:`~networkx.MultiGraph` supporting near-constant-time snapshots."""

    _snapshot_class = _SnapshotMultiGraph

    def __init__(self, incoming_graph_data=None, **attr):
        store = _VersionStore()
        self._version_store = store
        for factory_name in _FACTORY_NAMES:
            setattr(self, factory_name, store.create_dict)
        super().__init__(incoming_graph_data, **attr)

    snapshot = VersionedGraph.snapshot


class VersionedMultiDiGraph(MultiDiGraph):
    """:class:`~networkx.MultiDiGraph` with near-constant-time snapshots."""

    _snapshot_class = _SnapshotMultiDiGraph

    def __init__(self, incoming_graph_data=None, **attr):
        store = _VersionStore()
        self._version_store = store
        for factory_name in _FACTORY_NAMES:
            setattr(self, factory_name, store.create_dict)
        super().__init__(incoming_graph_data, **attr)

    snapshot = VersionedDiGraph.snapshot


def snapshot(G):
    """Capture a read-only snapshot of a versioned graph.

    Parameters
    ----------
    G : VersionedGraph, VersionedDiGraph, VersionedMultiGraph or
        VersionedMultiDiGraph
        The live graph to snapshot.  Equivalent to calling ``G.snapshot()``.

    Returns
    -------
    snapshot : frozen graph
        A frozen, read-only graph of the same type as ``G``.

    Raises
    ------
    NetworkXError
        If ``G`` is not one of the ``Versioned*`` graph classes.  Plain
        networkx graphs keep no version history; wrap or recreate your data in
        a ``Versioned*`` graph to use snapshots.
    """
    if not isinstance(
        G,
        (
            VersionedGraph,
            VersionedDiGraph,
            VersionedMultiGraph,
            VersionedMultiDiGraph,
        ),
    ):
        raise nx.NetworkXError(
            "snapshot() requires a VersionedGraph, VersionedDiGraph, "
            "VersionedMultiGraph or VersionedMultiDiGraph; "
            f"got {type(G).__name__}"
        )
    return G.snapshot()
