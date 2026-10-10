"""
================================
Resumable (checkpointable) graph traversals
================================

Checkpointable counterparts of the lazy traversal generators
(:func:`~networkx.algorithms.traversal.breadth_first_search.bfs_edges`,
:func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`,
:func:`~networkx.algorithms.traversal.edgebfs.edge_bfs` and
:func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`).

A traversal can be interrupted at any yielded edge, its progress saved with
:meth:`ResumableTraversal.snapshot`, and resumed in another process by passing
the saved state back to the same traversal function.  Resumption produces
exactly the edges that have not yet been yielded -- no duplicates and no
omissions -- in the same order the uninterrupted traversal would use.

A checkpoint can only be restored against a graph that is structurally
identical (same nodes, edges, multigraph edge keys and graph type) and with
traversal parameters identical to those recorded in the checkpoint; otherwise
:exc:`~networkx.exception.NetworkXCheckpointError` is raised.

Node and edge-key objects must be picklable for checkpoints to cross process
boundaries.
"""

import copy
import hashlib
import pickle
from collections import deque

import networkx as nx
from networkx.algorithms.traversal.edgebfs import FORWARD, REVERSE

__all__ = [
    "ResumableTraversal",
    "resumable_bfs_edges",
    "resumable_dfs_edges",
    "resumable_edge_bfs",
    "resumable_edge_dfs",
]

#: Version of the checkpoint state format produced by this module.
STATE_VERSION = 1

_UNSET = object()
_EXHAUSTED = object()

_MASK64 = (1 << 64) - 1
_PICKLE_PROTOCOL = 4


def _stable_hash(value):
    """Deterministic 64-bit hash that is stable across processes.

    Python's built-in ``hash()`` is salted per process for strings, so the
    checkpoint checksum is derived from a cryptographic hash of a fixed
    protocol pickle instead.
    """
    payload = pickle.dumps(value, protocol=_PICKLE_PROTOCOL)
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def graph_fingerprint(G):
    """Return a picklable structural fingerprint of a graph.

    The fingerprint captures the graph type flags, node/edge counts and an
    order-independent checksum over the nodes, adjacency and multigraph edge
    keys.  Edge attribute values are not part of the fingerprint because
    traversal order does not depend on them.
    """
    multigraph = G.is_multigraph()
    checksum = 0
    for u, nbrs in G.adjacency():
        neighbor_sum = 0
        for v, edges in nbrs.items():
            term = _stable_hash(v)
            if multigraph:
                key_sum = 0
                for key in edges:
                    key_sum = (key_sum + _stable_hash(key)) & _MASK64
                term = (term + (len(edges) ^ key_sum)) & _MASK64
            neighbor_sum = (neighbor_sum + term) & _MASK64
        node_sig = (
            _stable_hash(u) ^ ((neighbor_sum * 0x9E3779B97F4A7C15) & _MASK64)
        ) & _MASK64
        checksum ^= ((node_sig << 17) | (node_sig >> 47)) & _MASK64
    return {
        "directed": G.is_directed(),
        "multigraph": multigraph,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "checksum": checksum,
    }


class ResumableTraversal:
    """An edge traversal whose progress can be saved and resumed.

    Instances are created by :func:`resumable_bfs_edges`,
    :func:`resumable_dfs_edges`, :func:`resumable_edge_bfs` and
    :func:`resumable_edge_dfs`; they are not intended to be instantiated
    directly.  Instances are iterators yielding the same edges, in the same
    order, as the corresponding non-resumable traversal function.

    Attributes
    ----------
    algorithm : str
        Name of the traversal (``"bfs_edges"``, ``"dfs_edges"``,
        ``"edge_bfs"`` or ``"edge_dfs"``).
    yielded : int
        Number of edges produced so far.

    See Also
    --------
    snapshot
    close
    """

    __slots__ = (
        "_G",
        "_algorithm",
        "_params",
        "_p",
        "_done",
        "_closed",
        "_yielded",
        "_last",
        "_children",
        "_edges_from",
        "_edge_id",
        "_check_reverse",
    )

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._done:
            raise StopIteration
        item = self._step()
        if item is _EXHAUSTED:
            self._done = True
            raise StopIteration
        self._yielded += 1
        self._last = item
        return item

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def yielded(self):
        return self._yielded

    @property
    def closed(self):
        return self._closed

    # ------------------------------------------------------------------
    # construction / restoration
    # ------------------------------------------------------------------

    @classmethod
    def _start(cls, G, algorithm, params, callables):
        self = cls()
        self._G = G
        self._algorithm = algorithm
        self._params = params
        self._done = False
        self._closed = False
        self._yielded = 0
        self._last = None
        self._bind(G, params, callables)
        self._p = self._initial_progress(G, algorithm, params)
        if not self._p["active"]:
            self._done = True
        return self

    @classmethod
    def restore(cls, G, state, *, sort_neighbors=None):
        """Restore a resumable traversal from a checkpoint state.

        Users normally call the traversal function with the ``state`` keyword
        instead of this method.
        """
        if not isinstance(state, dict):
            raise nx.NetworkXCheckpointError("checkpoint state must be a dict")
        version = state.get("version")
        if version != STATE_VERSION:
            raise nx.NetworkXCheckpointError(
                f"incompatible checkpoint state version {version!r}; "
                f"this version of NetworkX supports version {STATE_VERSION}"
            )
        algorithm = state.get("algorithm")
        if algorithm not in _ALGORITHMS:
            raise nx.NetworkXCheckpointError(
                f"unknown traversal {algorithm!r} in checkpoint state"
            )
        params = state.get("params")
        if not isinstance(params, dict):
            raise nx.NetworkXCheckpointError(
                "corrupt checkpoint state: missing parameters"
            )
        progress = state.get("progress")
        if not isinstance(progress, dict):
            raise nx.NetworkXCheckpointError(
                "corrupt checkpoint state: missing progress"
            )

        expected_fp = state.get("graph")
        if not isinstance(expected_fp, dict):
            raise nx.NetworkXCheckpointError(
                "corrupt checkpoint state: missing graph fingerprint"
            )
        actual_fp = graph_fingerprint(G)
        if actual_fp["directed"] != expected_fp.get("directed") or (
            actual_fp["multigraph"] != expected_fp.get("multigraph")
        ):
            raise nx.NetworkXCheckpointError(
                "cannot resume traversal: graph type "
                f"(directed={actual_fp['directed']}, "
                f"multigraph={actual_fp['multigraph']}) does not match the graph "
                "the checkpoint was created on"
            )
        for key in ("nodes", "edges", "checksum"):
            if actual_fp.get(key) != expected_fp.get(key):
                raise nx.NetworkXCheckpointError(
                    "cannot resume traversal: the graph has changed since the "
                    "checkpoint was created"
                )

        needs_sort = bool(params.get("sort_neighbors"))
        if needs_sort and sort_neighbors is None:
            raise nx.NetworkXCheckpointError(
                "the checkpoint was created with a custom sort_neighbors "
                "function; the same function must be supplied to resume"
            )
        if not needs_sort and sort_neighbors is not None:
            raise nx.NetworkXCheckpointError(
                "the checkpoint was created without sort_neighbors, but a "
                "sort_neighbors function was supplied to resume"
            )

        self = cls()
        self._G = G
        self._algorithm = algorithm
        self._params = copy.deepcopy(params)
        self._closed = False
        self._done = bool(state.get("done"))
        self._yielded = int(state.get("yielded", 0))
        self._last = copy.deepcopy(state.get("last_item"))
        self._bind(G, self._params, {"sort_neighbors": sort_neighbors})
        self._p = copy.deepcopy(progress)
        if "queue" in self._p:
            self._p["queue"] = deque(self._p["queue"])
        return self

    def _bind(self, G, params, callables):
        """Create the graph accessors used by the state machine."""
        self._edges_from = None
        self._edge_id = None
        self._check_reverse = False

        if self._algorithm in ("bfs_edges", "dfs_edges"):
            if params.get("reverse") and G.is_directed():
                successors = G.predecessors
            else:
                successors = G.neighbors
            sort_neighbors = callables.get("sort_neighbors")
            if sort_neighbors is None:

                def children(node):
                    return list(successors(node))
            else:

                def children(node):
                    return list(sort_neighbors(successors(node)))

            self._children = children
            return

        directed = G.is_directed()
        kwds = {"data": False}
        if G.is_multigraph():
            kwds["keys"] = True
        orientation = params.get("orientation")

        if orientation is None:

            def edges_from(node):
                return list(G.edges(node, **kwds))

        elif not directed or orientation == "original":

            def edges_from(node):
                return [e + (FORWARD,) for e in G.edges(node, **kwds)]

        elif orientation == "reverse":

            def edges_from(node):
                return [e + (REVERSE,) for e in G.in_edges(node, **kwds)]

        elif orientation == "ignore":

            def edges_from(node):
                return [
                    *(e + (FORWARD,) for e in G.edges(node, **kwds)),
                    *(e + (REVERSE,) for e in G.in_edges(node, **kwds)),
                ]

        else:
            raise nx.NetworkXError("invalid orientation argument.")

        if directed:

            def edge_id(edge):
                # remove direction indicator
                return edge[:-1] if orientation is not None else edge

        else:

            def edge_id(edge):
                # single id for undirected requires frozenset on nodes
                return (frozenset(edge[:2]),) + edge[2:]

        self._edges_from = edges_from
        self._edge_id = edge_id
        self._check_reverse = directed and orientation in ("reverse", "ignore")

    def _initial_progress(self, G, algorithm, params):
        if algorithm == "bfs_edges":
            return {
                "active": True,
                "seen": {params["source"]},
                "n": len(G),
                "depth": 0,
                "layer": [[params["source"], self._children(params["source"]), 0]],
                "next_layer": [],
                "pi": 0,
            }
        if algorithm == "dfs_edges":
            source = params["source"]
            nodes = [source] if source is not None else list(G)
            return {
                "active": True,
                "visited": set(),
                "stack": [],
                "depth_now": 0,
                "nodes": nodes,
                "si": 0,
                "pending": None,
            }
        # edge traversals
        nodes = list(G.nbunch_iter(params.get("source")))
        if algorithm == "edge_bfs":
            return {
                "active": bool(nodes),
                "visited_nodes": set(nodes),
                "visited_edges": set(),
                "queue": deque([[n, self._edges_from(n), 0] for n in nodes]),
            }
        # edge_dfs
        return {
            "active": bool(nodes),
            "visited_nodes": set(),
            "visited_edges": set(),
            "progress": {},
            "stack": [],
            "nodes": nodes,
            "si": 0,
        }

    # ------------------------------------------------------------------
    # state machines -- one step yields one edge
    # ------------------------------------------------------------------

    def _step(self):
        algorithm = self._algorithm
        if algorithm == "bfs_edges":
            return self._step_bfs()
        if algorithm == "dfs_edges":
            return self._step_dfs()
        if algorithm == "edge_bfs":
            return self._step_edge_bfs()
        return self._step_edge_dfs()

    def _step_bfs(self):
        # Mirrors generic_bfs_edges, with adjacency iterators replaced by
        # materialized neighbor lists and an explicit position in each list.
        p = self._p
        depth_limit = self._params["depth_limit"]
        while p["layer"] and p["depth"] < depth_limit:
            layer = p["layer"]
            while p["pi"] < len(layer):
                frame = layer[p["pi"]]
                parent, children, ci = frame
                while ci < len(children):
                    child = children[ci]
                    frame[2] = ci + 1
                    ci += 1
                    if child in p["seen"]:
                        continue
                    p["seen"].add(child)
                    p["next_layer"].append([child, self._children(child), 0])
                    return (parent, child)
                p["pi"] += 1
                if len(p["seen"]) == p["n"]:
                    p["layer"] = []
                    return _EXHAUSTED
            p["layer"] = p["next_layer"]
            p["next_layer"] = []
            p["pi"] = 0
            p["depth"] += 1
        return _EXHAUSTED

    def _step_dfs(self):
        # Mirrors dfs_edges.  A newly discovered edge is returned *before* the
        # child is marked visited and pushed, so the pending child is recorded
        # and committed at the start of the next step.
        p = self._p
        depth_limit = self._params["depth_limit"]
        pending = p["pending"]
        if pending is not None:
            p["pending"] = None
            child = pending[1]
            p["visited"].add(child)
            if p["depth_now"] < depth_limit:
                p["stack"].append([child, self._children(child), 0])
                p["depth_now"] += 1
        while True:
            if p["stack"]:
                frame = p["stack"][-1]
                parent, children, ci = frame
                while ci < len(children):
                    child = children[ci]
                    frame[2] = ci + 1
                    if child in p["visited"]:
                        ci = frame[2]
                        continue
                    p["pending"] = [parent, child]
                    return (parent, child)
                p["stack"].pop()
                p["depth_now"] -= 1
                continue
            nodes = p["nodes"]
            while p["si"] < len(nodes) and nodes[p["si"]] in p["visited"]:
                p["si"] += 1
            if p["si"] >= len(nodes):
                return _EXHAUSTED
            start = nodes[p["si"]]
            p["si"] += 1
            p["visited"].add(start)
            p["stack"] = [[start, self._children(start), 0]]
            p["depth_now"] = 1

    def _step_edge_bfs(self):
        # Mirrors edge_bfs.
        p = self._p
        queue = p["queue"]
        while queue:
            frame = queue[0]
            _, edges, ci = frame
            while ci < len(edges):
                edge = edges[ci]
                frame[2] = ci + 1
                if self._check_reverse and edge[-1] == REVERSE:
                    child = edge[0]
                else:
                    child = edge[1]
                if child not in p["visited_nodes"]:
                    p["visited_nodes"].add(child)
                    queue.append([child, self._edges_from(child), 0])
                edgeid = self._edge_id(edge)
                if edgeid not in p["visited_edges"]:
                    p["visited_edges"].add(edgeid)
                    return edge
                ci = frame[2]
            queue.popleft()
        return _EXHAUSTED

    def _step_edge_dfs(self):
        # Mirrors edge_dfs.  The edge iterator of a node is shared across all
        # occurrences of that node on the stack, so the per-node position lives
        # in p["progress"] rather than in the stack frame.
        p = self._p
        while True:
            if not p["stack"]:
                if p["si"] >= len(p["nodes"]):
                    return _EXHAUSTED
                start = p["nodes"][p["si"]]
                p["si"] += 1
                p["stack"] = [start]
            current = p["stack"][-1]
            if current not in p["visited_nodes"]:
                p["visited_nodes"].add(current)
                p["progress"][current] = [self._edges_from(current), 0]
            frame = p["progress"][current]
            edges, ci = frame
            while ci < len(edges):
                edge = edges[ci]
                frame[1] = ci + 1
                edgeid = self._edge_id(edge)
                if edgeid in p["visited_edges"]:
                    ci = frame[1]
                    continue
                p["visited_edges"].add(edgeid)
                if self._check_reverse and edge[-1] == REVERSE:
                    child = edge[0]
                else:
                    child = edge[1]
                p["stack"].append(child)
                return edge
            p["stack"].pop()

    # ------------------------------------------------------------------
    # checkpointing / resource handling
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return a picklable checkpoint of the current traversal state.

        The returned dict contains only plain Python data; it never references
        the graph.  It can be serialized (for example with :mod:`pickle`),
        stored and passed as the ``state`` keyword to the same traversal
        function in another process to continue the traversal without
        repeating or omitting edges.
        """
        if self._closed:
            raise nx.NetworkXError("cannot snapshot a traversal that has been closed")
        progress = copy.deepcopy(self._p)
        if "queue" in progress:
            progress["queue"] = list(progress["queue"])
        return {
            "version": STATE_VERSION,
            "algorithm": self._algorithm,
            "params": copy.deepcopy(self._params),
            "graph": graph_fingerprint(self._G),
            "progress": progress,
            "yielded": self._yielded,
            "last_item": copy.deepcopy(self._last),
            "done": self._done,
        }

    def close(self):
        """Release the graph and all resources held by the traversal.

        Closing is idempotent.  A closed traversal cannot be iterated or
        snapshotted further; checkpointed state returned by earlier
        :meth:`snapshot` calls is unaffected.
        """
        if not self._closed:
            self._closed = True
            self._G = None
            self._p = None
            self._params = None
            self._children = None
            self._edges_from = None
            self._edge_id = None
            self._last = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


_ALGORITHMS = {"bfs_edges", "dfs_edges", "edge_bfs", "edge_dfs"}


def _check_resume_kwargs(state, algorithm, provided):
    """Validate structural keyword arguments supplied with a checkpoint."""
    if not isinstance(state, dict):
        raise nx.NetworkXCheckpointError("checkpoint state must be a dict")
    if state.get("version") != STATE_VERSION:
        raise nx.NetworkXCheckpointError(
            f"incompatible checkpoint state version {state.get('version')!r}; "
            f"this version of NetworkX supports version {STATE_VERSION}"
        )
    if state.get("algorithm") != algorithm:
        raise nx.NetworkXCheckpointError(
            f"checkpoint for {state.get('algorithm')!r} cannot be resumed with "
            f"{algorithm!r}"
        )
    params = state.get("params")
    if not isinstance(params, dict):
        raise nx.NetworkXCheckpointError("corrupt checkpoint state: missing parameters")
    for key, value in provided.items():
        if value is not _UNSET and value != params.get(key):
            raise nx.NetworkXCheckpointError(
                f"parameter {key!r} supplied to resume does not match the value "
                "recorded in the checkpoint"
            )


@nx._dispatchable
def resumable_bfs_edges(
    G,
    source=_UNSET,
    reverse=_UNSET,
    depth_limit=_UNSET,
    sort_neighbors=_UNSET,
    *,
    state=None,
):
    """A resumable version of :func:`bfs_edges`.

    Iterates over the edges of a breadth-first search exactly like
    :func:`~networkx.algorithms.traversal.breadth_first_search.bfs_edges`, but
    returns a :class:`ResumableTraversal` whose progress can be saved at any
    yielded edge with :meth:`~ResumableTraversal.snapshot` and restored later,
    including in another process, by passing the saved state back through the
    ``state`` keyword.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search.  Must not be given when
        resuming from ``state`` (it is read from the checkpoint).

    reverse : bool, optional
        Traverse a directed graph in the reverse direction.

    depth_limit : int, optional (default=len(G))
        Maximum search depth.

    sort_neighbors : function, optional
        A function that takes an iterator over nodes and returns an iterable
        with a custom ordering.  When resuming a checkpoint created with a
        custom ``sort_neighbors`` function, the *same* function must be passed
        again.

    state : dict, optional
        Checkpoint state returned by a previous
        :meth:`ResumableTraversal.snapshot` call.  When given, the traversal
        continues after the edges already produced when the checkpoint was
        taken.

    Yields
    ------
    edge : 2-tuple of nodes
        Edges of the breadth-first search, in the same order as
        :func:`bfs_edges`.

    Raises
    ------
    NetworkXCheckpointError
        If ``state`` has an incompatible version, belongs to a different
        traversal, the graph has changed, or supplied parameters do not match
        the checkpoint.

    Examples
    --------
    >>> import pickle
    >>> G = nx.path_graph(6)
    >>> traversal = nx.resumable_bfs_edges(G, 0)
    >>> head = [next(traversal) for _ in range(2)]
    >>> checkpoint = pickle.dumps(traversal.snapshot())

    The checkpoint can be restored in another process against a graph with
    the same structure:

    >>> G_new = nx.path_graph(6)
    >>> resumed = nx.resumable_bfs_edges(G_new, state=pickle.loads(checkpoint))
    >>> head + list(resumed) == list(nx.bfs_edges(G, 0))
    True

    See Also
    --------
    bfs_edges
    ResumableTraversal
    """
    if state is not None:
        _check_resume_kwargs(
            state,
            "bfs_edges",
            {
                "source": source,
                "reverse": reverse,
                "depth_limit": depth_limit,
            },
        )
        return ResumableTraversal.restore(
            G,
            state,
            sort_neighbors=None if sort_neighbors is _UNSET else sort_neighbors,
        )
    if source is _UNSET:
        raise TypeError("resumable_bfs_edges() requires a 'source' node")
    if source not in G:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")
    reverse = False if reverse is _UNSET else reverse
    if depth_limit is _UNSET or depth_limit is None:
        depth_limit = len(G)
    sort_neighbors = None if sort_neighbors is _UNSET else sort_neighbors
    params = {
        "source": source,
        "reverse": bool(reverse),
        "depth_limit": depth_limit,
        "sort_neighbors": sort_neighbors is not None,
    }
    return ResumableTraversal._start(
        G, "bfs_edges", params, {"sort_neighbors": sort_neighbors}
    )


@nx._dispatchable
def resumable_dfs_edges(
    G,
    source=_UNSET,
    depth_limit=_UNSET,
    sort_neighbors=_UNSET,
    *,
    state=None,
):
    """A resumable version of :func:`dfs_edges`.

    Iterates over the edges of a depth-first search exactly like
    :func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`, but
    returns a :class:`ResumableTraversal` whose progress can be saved and
    restored (also in another process) through
    :meth:`~ResumableTraversal.snapshot` and the ``state`` keyword.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
        Starting node.  If not given, all components are searched.  Must not
        be given when resuming from ``state``.

    depth_limit : int, optional (default=len(G))
        Maximum search depth.

    sort_neighbors : function, optional
        A function that takes an iterator over nodes and returns an iterable
        with a custom ordering.  The same function must be supplied again when
        resuming a checkpoint that used one.

    state : dict, optional
        Checkpoint state from a previous traversal.

    Yields
    ------
    edge : 2-tuple of nodes
        Edges of the depth-first search, in the same order as
        :func:`dfs_edges`.

    Raises
    ------
    NetworkXCheckpointError
        If the checkpoint cannot be applied to this graph or parameters.

    Examples
    --------
    >>> import pickle
    >>> G = nx.path_graph(6)
    >>> traversal = nx.resumable_dfs_edges(G, source=0, depth_limit=4)
    >>> head = [next(traversal) for _ in range(2)]
    >>> checkpoint = pickle.dumps(traversal.snapshot())
    >>> resumed = nx.resumable_dfs_edges(
    ...     nx.path_graph(6), depth_limit=4, state=pickle.loads(checkpoint)
    ... )
    >>> head + list(resumed) == list(nx.dfs_edges(G, source=0, depth_limit=4))
    True

    See Also
    --------
    dfs_edges
    ResumableTraversal
    """
    if state is not None:
        _check_resume_kwargs(
            state,
            "dfs_edges",
            {"source": source, "depth_limit": depth_limit},
        )
        return ResumableTraversal.restore(
            G,
            state,
            sort_neighbors=None if sort_neighbors is _UNSET else sort_neighbors,
        )
    source = None if source is _UNSET else source
    if source is not None and source not in G:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")
    if depth_limit is _UNSET or depth_limit is None:
        depth_limit = len(G)
    sort_neighbors = None if sort_neighbors is _UNSET else sort_neighbors
    params = {
        "source": source,
        "depth_limit": depth_limit,
        "sort_neighbors": sort_neighbors is not None,
    }
    return ResumableTraversal._start(
        G, "dfs_edges", params, {"sort_neighbors": sort_neighbors}
    )


@nx._dispatchable
def resumable_edge_bfs(G, source=_UNSET, orientation=_UNSET, *, state=None):
    """A resumable version of :func:`edge_bfs`.

    Performs the same breadth-first traversal of *all* edges (including
    parallel edges in multigraphs and edges back to visited nodes) as
    :func:`~networkx.algorithms.traversal.edgebfs.edge_bfs`, but returns a
    :class:`ResumableTraversal` that can be checkpointed and resumed in
    another process.

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    source : node, list of nodes, optional
        Node (or nodes) where the traversal begins.  If ``None``, a source is
        chosen until all edges are searched.  Must not be given when resuming
        from ``state``.

    orientation : None | 'original' | 'reverse' | 'ignore', optional
        Edge traversal direction, with the same meaning as in
        :func:`edge_bfs`.

    state : dict, optional
        Checkpoint state from a previous traversal.

    Yields
    ------
    edge : tuple
        Edges in breadth-first order, with the same tuple shape (including
        multigraph keys and direction markers) as :func:`edge_bfs`.

    Raises
    ------
    NetworkXCheckpointError
        If the checkpoint cannot be applied to this graph or parameters.

    Examples
    --------
    >>> import pickle
    >>> edges = [(0, 1), (1, 0), (1, 0), (2, 0), (2, 1), (3, 1)]
    >>> G = nx.MultiDiGraph(edges)
    >>> traversal = nx.resumable_edge_bfs(G, [0, 1, 2, 3])
    >>> head = [next(traversal) for _ in range(3)]
    >>> checkpoint = pickle.dumps(traversal.snapshot())
    >>> resumed = nx.resumable_edge_bfs(
    ...     nx.MultiDiGraph(edges), state=pickle.loads(checkpoint)
    ... )
    >>> head + list(resumed) == list(nx.edge_bfs(G, [0, 1, 2, 3]))
    True

    See Also
    --------
    edge_bfs
    ResumableTraversal
    """
    if state is not None:
        _check_resume_kwargs(
            state,
            "edge_bfs",
            {"source": source, "orientation": orientation},
        )
        return ResumableTraversal.restore(G, state)
    source = None if source is _UNSET else source
    orientation = None if orientation is _UNSET else orientation
    params = {"source": source, "orientation": orientation}
    return ResumableTraversal._start(G, "edge_bfs", params, {})


@nx._dispatchable
def resumable_edge_dfs(G, source=_UNSET, orientation=_UNSET, *, state=None):
    """A resumable version of :func:`edge_dfs`.

    Performs the same depth-first traversal of *all* edges (including parallel
    edges in multigraphs and edges back to visited nodes) as
    :func:`~networkx.algorithms.traversal.edgedfs.edge_dfs`, but returns a
    :class:`ResumableTraversal` that can be checkpointed and resumed in
    another process.

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    source : node, list of nodes, optional
        Node (or nodes) where the traversal begins.  If ``None``, a source is
        chosen until all edges are searched.  Must not be given when resuming
        from ``state``.

    orientation : None | 'original' | 'reverse' | 'ignore', optional
        Edge traversal direction, with the same meaning as in
        :func:`edge_dfs`.

    state : dict, optional
        Checkpoint state from a previous traversal.

    Yields
    ------
    edge : tuple
        Edges in depth-first order, with the same tuple shape (including
        multigraph keys and direction markers) as :func:`edge_dfs`.

    Raises
    ------
    NetworkXCheckpointError
        If the checkpoint cannot be applied to this graph or parameters.

    Examples
    --------
    >>> import pickle
    >>> edges = [(0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]
    >>> G = nx.MultiDiGraph(edges)
    >>> traversal = nx.resumable_edge_dfs(G, [0, 1, 2, 3])
    >>> head = [next(traversal) for _ in range(2)]
    >>> checkpoint = pickle.dumps(traversal.snapshot())
    >>> resumed = nx.resumable_edge_dfs(
    ...     nx.MultiDiGraph(edges), state=pickle.loads(checkpoint)
    ... )
    >>> head + list(resumed) == list(nx.edge_dfs(G, [0, 1, 2, 3]))
    True

    See Also
    --------
    edge_dfs
    ResumableTraversal
    """
    if state is not None:
        _check_resume_kwargs(
            state,
            "edge_dfs",
            {"source": source, "orientation": orientation},
        )
        return ResumableTraversal.restore(G, state)
    source = None if source is _UNSET else source
    orientation = None if orientation is _UNSET else orientation
    params = {"source": source, "orientation": orientation}
    return ResumableTraversal._start(G, "edge_dfs", params, {})
