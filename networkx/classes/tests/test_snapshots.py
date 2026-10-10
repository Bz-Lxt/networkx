"""Tests for near-constant-time versioned graph snapshots."""

import gc
import time
import weakref

import pytest

import networkx as nx
from networkx.classes.snapshots import (
    VersionedDiGraph,
    VersionedGraph,
    VersionedMultiDiGraph,
    VersionedMultiGraph,
)

versioned_graphs = [
    VersionedGraph,
    VersionedDiGraph,
    VersionedMultiGraph,
    VersionedMultiDiGraph,
]


@pytest.mark.parametrize("graph_type", versioned_graphs)
def test_snapshot_is_frozen_graph_of_matching_type(graph_type):
    G = graph_type()
    G.add_edge(1, 2)
    S = G.snapshot()

    assert type(S).__mro__[1] is graph_type.__mro__[1]
    assert nx.is_frozen(S)
    assert S.is_directed() == G.is_directed()
    assert S.is_multigraph() == G.is_multigraph()
    with pytest.raises(nx.NetworkXError):
        S.add_node(3)
    with pytest.raises(nx.NetworkXError):
        S.add_edge(1, 3)
    with pytest.raises(nx.NetworkXError):
        S.remove_node(1)


def test_snapshot_pins_structure_and_attributes():
    G = VersionedGraph()
    G.graph["version"] = "v1"
    G.add_edge(1, 2, weight=7)
    G.add_node(9, color="red")
    S = G.snapshot()

    G.add_edge(2, 3, weight=4)
    G.remove_node(9)
    G[1][2]["weight"] = 99
    G.graph["version"] = "v2"

    assert list(S.nodes) == [1, 2, 9]
    assert list(S.edges) == [(1, 2)]
    assert list(S.edges(data=True)) == [(1, 2, {"weight": 7})]
    assert S.nodes[9] == {"color": "red"}
    assert S.graph["version"] == "v1"
    assert S.name == ""
    # the live graph reports the updated state
    assert list(G.edges) == [(1, 2), (2, 3)]
    assert G[1][2]["weight"] == 99
    assert G.graph["version"] == "v2"
    assert 9 not in G


def test_snapshot_includes_top_level_graph_attributes():
    G = VersionedGraph(name="g", owner="me")
    S = G.snapshot()
    G.name = "live"
    G.graph["owner"] = "someone else"

    assert S.graph == {"name": "g", "owner": "me"}
    assert S.name == "g"


def test_live_views_remain_dynamic():
    G = VersionedGraph()
    G.add_edge(1, 2)
    S = G.snapshot()
    G.add_edge(2, 3)

    assert list(G.nodes) == [1, 2, 3]
    assert list(G.edges) == [(1, 2), (2, 3)]
    assert G.degree[2] == 2
    # the snapshot is unaffected by live view changes
    assert list(S.nodes) == [1, 2]
    assert S.degree[2] == 1


def test_directed_snapshot():
    G = VersionedDiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    S = G.snapshot()
    G.add_edge(3, 4)
    G.remove_edge(1, 2)

    assert list(S.edges) == [(0, 1), (1, 2), (2, 3)]
    assert dict(S.out_degree()) == {0: 1, 1: 1, 2: 1, 3: 0}
    assert dict(S.in_degree()) == {0: 0, 1: 1, 2: 1, 3: 1}
    assert list(G.edges) == [(0, 1), (2, 3), (3, 4)]


def test_reverse_view_keeps_capture_order():
    G = VersionedDiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    S = G.snapshot()
    R = nx.reverse_view(S)
    G.add_edge(9, 0)
    G.remove_node(1)

    assert list(R.nodes) == [0, 1, 2, 3]
    assert list(R.edges) == [(1, 0), (2, 1), (3, 2)]
    # incoming edges of the reversed view are the original outward edges
    assert list(R.in_edges()) == [(1, 0), (2, 1), (3, 2)]
    assert list(R.out_edges()) == [(1, 0), (2, 1), (3, 2)]
    assert R.degree[1] == 2
    # same for G.reverse(copy=False)
    R2 = S.reverse(copy=False)
    assert list(R2.edges) == [(1, 0), (2, 1), (3, 2)]


def test_subgraph_view_combined_filters_keep_capture_order():
    G = VersionedDiGraph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)])
    S = G.snapshot()
    G.add_edge(4, 0, weight=9)
    G.remove_edge(2, 3)

    sub = nx.subgraph_view(
        S,
        filter_node=lambda n: n in {0, 1, 2},
        filter_edge=lambda u, v: (u, v) in {(0, 1), (1, 2)},
    )
    assert list(sub.nodes) == [0, 1, 2]
    assert list(sub.edges) == [(0, 1), (1, 2)]
    # the filter closure sees pinned data, not later live updates
    edge_filtered = nx.subgraph_view(S, filter_edge=lambda u, v: S[u][v].get("x", False))
    assert list(edge_filtered.edges) == []
    G[0][1]["x"] = True
    assert list(edge_filtered.edges) == []


def test_subgraph_view_undirected_snapshot():
    G = VersionedGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    S = G.snapshot()
    G.add_edge(4, 5)
    sub = nx.subgraph_view(S, filter_node=lambda n: n < 3)

    assert list(sub.nodes) == [0, 1, 2]
    assert sorted(sub.edges) == [(0, 1), (1, 2)]


def test_multigraph_snapshot_keys_and_data():
    G = VersionedMultiGraph()
    k0 = G.add_edge("a", "b", weight=1)
    k1 = G.add_edge("a", "b", weight=2)
    G.add_edge("b", "c", weight=3)
    S = G.snapshot()
    G.add_edge("a", "b", weight=4)
    G.remove_edge("a", "b", k0)

    assert k0 == 0 and k1 == 1
    assert sorted(S.edges(keys=True, data=True)) == [
        ("a", "b", 0, {"weight": 1}),
        ("a", "b", 1, {"weight": 2}),
        ("b", "c", 0, {"weight": 3}),
    ]
    assert sorted(S.edges(keys=True)) == [("a", "b", 0), ("a", "b", 1), ("b", "c", 0)]
    assert S["a"]["b"][0] == {"weight": 1}
    # live graph lost key 0 and gained key 2
    assert sorted(G.edges(keys=True)) == [("a", "b", 1), ("a", "b", 2), ("b", "c", 0)]


def test_multidigraph_snapshot_and_reverse():
    G = VersionedMultiDiGraph()
    G.add_edge(1, 2, key="x", weight=5)
    S = G.snapshot()
    G.add_edge(1, 2, key="y", weight=6)

    assert list(S.edges(keys=True, data=True)) == [(1, 2, "x", {"weight": 5})]
    assert list(G.edges(keys=True, data=True)) == [
        (1, 2, "x", {"weight": 5}),
        (1, 2, "y", {"weight": 6}),
    ]
    R = nx.reverse_view(S)
    assert list(R.edges(keys=True)) == [(2, 1, "x")]


def test_capture_order_after_delete_and_readd():
    G = VersionedGraph()
    G.add_nodes_from([0, 1, 2, 3])
    S = G.snapshot()
    G.add_node(4)
    G.remove_node(1)
    s1 = G.snapshot()
    G.add_node(1)  # re-added node moves to the end, like a plain dict

    assert list(S.nodes) == [0, 1, 2, 3]
    assert list(s1.nodes) == [0, 2, 3, 4]
    assert list(G.nodes) == [0, 2, 3, 4, 1]
    # edge order in snapshots similarly follows capture-time insertion
    assert list(S.edges) == []
    assert list(s1.edges) == []


def test_overlapping_snapshots_are_independent():
    G = VersionedGraph()
    G.add_edge(0, 1)
    s0 = G.snapshot()
    G.add_edge(1, 2)
    s1 = G.snapshot()
    G.add_edge(2, 3)
    s2 = G.snapshot()
    G.remove_edge(0, 1)

    assert list(s0.edges) == [(0, 1)]
    assert list(s1.edges) == [(0, 1), (1, 2)]
    assert list(s2.edges) == [(0, 1), (1, 2), (2, 3)]

    del s0
    gc.collect()
    # releasing one snapshot does not disturb the others
    assert list(s1.edges) == [(0, 1), (1, 2)]
    assert list(s2.edges) == [(0, 1), (1, 2), (2, 3)]


def test_repeated_snapshot_cycles_on_live_graph():
    G = VersionedDiGraph()
    G.add_edge(0, 1)
    for i in range(1, 20):
        S = G.snapshot()
        G.add_edge(i, i + 1)
        G.nodes[i]["tag"] = i
        assert S.number_of_edges() == i
        assert list(S.pred[i]) == [i - 1]
        del S
        gc.collect()
    assert G.number_of_edges() == 20
    assert len(G._version_store._pins) == 0
    # history collapsed back to a single layer per container
    assert len(G._succ._layers) == 1
    assert len(G._succ[0]._layers) == 1


def test_gc_during_mutation_does_not_corrupt_live_state():
    # Cyclic garbage collection can release snapshot pins at allocation
    # points inside a graph update (key hashing is one of them).  Old
    # versions may only be pruned between complete writes.
    class GarbageCollectingNode:
        # Runs cyclic garbage collection on every hash; within add_edge some
        # of these hashes occur inside a guarded storage __setitem__.
        def __hash__(self):
            gc.collect()
            return hash(id(self))

        def __eq__(self, other):
            return self is other

    G = VersionedGraph()
    trigger = GarbageCollectingNode()
    G.add_node(trigger)
    G.add_edge(0, 1, weight=1)
    keep = G.snapshot()
    stale = G.snapshot()
    stale.number_of_edges()  # creates the snapshot/view reference cycle
    # ``stale`` is unreachable but still pinned until the cyclic GC runs;
    # the gc inside add_edge must not prune layers while the write is open.
    del stale
    G.add_edge(trigger, 0, weight=99)
    keep2 = G.snapshot()
    G.remove_node(trigger)

    assert list(keep.edges) == [(0, 1)]
    assert keep2.has_edge(trigger, 0)
    assert trigger not in G
    assert G[0][1]["weight"] == 1
    del keep, keep2
    gc.collect()
    assert G._version_store._pins == set()


def test_releasing_last_snapshot_reclaims_old_versions():
    class Payload:
        pass

    def make_refs():
        G = VersionedGraph()
        marker = Payload()
        G.add_node(0, payload=marker)
        G.add_edge(0, 1)
        marker_ref = weakref.ref(marker)
        adj_ref = weakref.ref(G._adj[0])
        S = G.snapshot()
        G.remove_node(0)
        # old node payload and its adjacency dict are retained while S lives
        return G, S, marker_ref, adj_ref

    G, S, marker_ref, adj_ref = make_refs()
    gc.collect()
    assert marker_ref() is not None
    assert adj_ref() is not None
    del S
    gc.collect()
    assert marker_ref() is None
    assert adj_ref() is None
    assert G._version_store._pins == set()
    assert len(G._node._layers) == 1


def test_close_releases_pin():
    G = VersionedGraph()
    G.add_edge(0, 1)
    S = G.snapshot()
    assert G._version_store._pins == {0}
    S.close()
    assert G._version_store._pins == set()
    S.close()  # idempotent


def test_snapshot_context_manager():
    G = VersionedGraph()
    G.add_edge(0, 1)
    with G.snapshot() as S:
        assert list(S.edges) == [(0, 1)]
    assert G._version_store._pins == set()


def test_snapshot_is_taken_in_constant_time():
    # snapshots must not walk the graph: ~20k nodes/edges snapshot should be
    # orders of magnitude faster than a full copy (and well under a second).
    G = VersionedGraph()
    G.add_edges_from((i, i + 1) for i in range(20000))

    start = time.perf_counter()
    S = G.snapshot()
    elapsed = time.perf_counter() - start

    start = time.perf_counter()
    G.copy()
    copy_elapsed = time.perf_counter() - start

    assert elapsed < 0.01
    assert copy_elapsed > 10 * elapsed
    assert S.number_of_edges() == 20000


def test_free_function_snapshot():
    G = VersionedDiGraph([(0, 1)])
    S = nx.snapshot(G)
    G.add_edge(1, 2)
    assert list(S.edges) == [(0, 1)]
    with pytest.raises(nx.NetworkXError, match="Versioned"):
        nx.snapshot(nx.Graph())


def test_snapshot_copy_is_independent_plain_graph():
    G = VersionedGraph()
    G.add_edge(0, 1, weight=3)
    S = G.snapshot()
    H = S.copy()
    G.add_edge(1, 2)

    assert type(H) is nx.Graph
    assert not nx.is_frozen(H)
    assert list(H.edges(data=True)) == [(0, 1, {"weight": 3})]
    H.add_edge(2, 3)
    assert list(S.edges) == [(0, 1)]


def test_versioned_graph_copy_is_independent():
    G = VersionedGraph()
    G.add_edge(0, 1, weight=3)
    H = G.copy()
    assert type(H) is VersionedGraph
    H.add_edge(1, 2)
    H[0][1]["weight"] = 99
    assert not G.has_edge(1, 2)
    assert G[0][1]["weight"] == 3


def test_data_views_on_snapshot():
    G = VersionedGraph()
    G.add_node(1, color="red")
    G.add_edge(1, 2, weight=4)
    S = G.snapshot()
    G.nodes[1]["color"] = "blue"

    assert list(S.nodes(data="color")) == [(1, "red"), (2, None)]
    assert (1, {"color": "red"}) in S.nodes(data=True)
    assert (1, {"color": "blue"}) not in S.nodes(data=True)
    assert list(S.edges.data("weight")) == [(1, 2, 4)]
    assert set(S.nodes) >= {1, 2}


def test_clear_and_clear_edges_do_not_touch_snapshot():
    G = VersionedGraph()
    G.add_edges_from([(0, 1), (1, 2)])
    S = G.snapshot()
    G.clear_edges()
    assert list(S.edges) == [(0, 1), (1, 2)]
    assert list(G.nodes) == [0, 1, 2]
    G.clear()
    assert list(S.nodes) == [0, 1, 2]
    assert list(S.edges) == [(0, 1), (1, 2)]
    assert G.number_of_nodes() == 0
