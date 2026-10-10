"""Tests for checkpointable/resumable lazy traversals."""

import gc
import multiprocessing as mp
import pickle
import weakref

import pytest

import networkx as nx
from networkx.algorithms.traversal.resumable import (
    STATE_VERSION,
    ResumableTraversal,
    graph_fingerprint,
)

EDGES = [(0, 1), (1, 0), (1, 0), (2, 0), (2, 1), (3, 1)]
GRAPH_TYPES = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
TRAVERSALS = ["bfs_edges", "dfs_edges", "edge_bfs", "edge_dfs"]

REF = {
    "bfs_edges": nx.bfs_edges,
    "dfs_edges": nx.dfs_edges,
    "edge_bfs": nx.edge_bfs,
    "edge_dfs": nx.edge_dfs,
}
RESUMABLE = {
    "bfs_edges": nx.resumable_bfs_edges,
    "dfs_edges": nx.resumable_dfs_edges,
    "edge_bfs": nx.resumable_edge_bfs,
    "edge_dfs": nx.resumable_edge_dfs,
}


def rebuild(G):
    """A brand-new graph object with the same structure as G."""
    if G.is_multigraph():
        return G.__class__(G.edges(keys=True))
    return G.__class__(G.edges())


def reverse_sorted(iterable):
    return sorted(iterable, reverse=True)


def start(kind, G, source, **kwargs):
    return RESUMABLE[kind](G, source, **kwargs)


def resume(kind, G, state, **kwargs):
    return RESUMABLE[kind](G, state=state, **kwargs)


def reference(kind, G, source, **kwargs):
    return list(REF[kind](G, source, **kwargs))


def assert_resume_equivalent(kind, G, source, cuts=None, **kwargs):
    """The traversal interrupted at every cut resumes without dup or omission."""
    resume_kwargs = {}
    if "sort_neighbors" in kwargs:
        resume_kwargs["sort_neighbors"] = kwargs["sort_neighbors"]
    full = reference(kind, G, source, **kwargs)
    cut_points = range(len(full) + 1) if cuts is None else cuts
    for cut in cut_points:
        traversal = start(kind, G, source, **kwargs)
        head = [next(traversal) for _ in range(cut)]
        assert traversal.yielded == cut
        # pickle round-trip + fresh graph object simulates a new process
        state = pickle.loads(pickle.dumps(traversal.snapshot()))
        resumed = resume(kind, rebuild(G), state, **resume_kwargs)
        tail = list(resumed)
        assert head == full[:cut]
        assert tail == full[cut:]
        assert head + tail == full
        assert resumed.yielded == len(full)


# ---------------------------------------------------------------------------
# Equivalence with the existing traversals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("graph_type", GRAPH_TYPES)
@pytest.mark.parametrize("kind", TRAVERSALS)
def test_equivalent_to_existing_traversals(graph_type, kind):
    G = graph_type(EDGES)
    source = [0, 1, 2, 3] if kind.startswith("edge") else 0
    assert_resume_equivalent(kind, G, source)


@pytest.mark.parametrize("kind", ["dfs_edges", "edge_bfs", "edge_dfs"])
def test_all_components(kind):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (10, 11), (11, 12), (20, 21)])
    assert_resume_equivalent(kind, G, None)


@pytest.mark.parametrize("depth_limit", [0, 1, 2, 3, 10])
def test_bfs_depth_limit(depth_limit):
    assert_resume_equivalent("bfs_edges", nx.DiGraph(EDGES), 0, depth_limit=depth_limit)


@pytest.mark.parametrize("depth_limit", [0, 1, 2, 3, 10])
def test_dfs_depth_limit(depth_limit):
    assert_resume_equivalent("dfs_edges", nx.DiGraph(EDGES), 0, depth_limit=depth_limit)


def test_reverse_directed_bfs():
    assert_resume_equivalent("bfs_edges", nx.DiGraph(EDGES), 1, reverse=True)


def test_reverse_flag_ignored_for_undirected_bfs():
    assert_resume_equivalent("bfs_edges", nx.Graph(EDGES), 0, reverse=True)


@pytest.mark.parametrize("orientation", [None, "original", "reverse", "ignore"])
@pytest.mark.parametrize("kind", ["edge_bfs", "edge_dfs"])
def test_orientations(kind, orientation):
    assert_resume_equivalent(
        kind, nx.MultiDiGraph(EDGES), [0, 1, 2, 3], orientation=orientation
    )


@pytest.mark.parametrize("kind", ["edge_bfs", "edge_dfs"])
def test_undirected_multigraph_orientation(kind):
    assert_resume_equivalent(
        kind, nx.MultiGraph(EDGES), [0, 1, 2, 3], orientation="ignore"
    )


def test_sort_neighbors_bfs():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    assert_resume_equivalent("bfs_edges", G, 0, sort_neighbors=reverse_sorted)


def test_sort_neighbors_dfs():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    assert_resume_equivalent("dfs_edges", G, 0, sort_neighbors=reverse_sorted)


def test_explicit_multigraph_keys_preserved():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b", key="first")
    G.add_edge("a", "b", key="second")
    G.add_edge("b", "c", key=0)
    assert_resume_equivalent("edge_bfs", G, ["a", "b", "c"])
    assert_resume_equivalent("edge_dfs", G, ["a", "b", "c"])
    full = list(nx.edge_bfs(G, ["a", "b", "c"]))
    assert all(len(e) == 3 for e in full)


def test_non_integer_nodes_pickle_safely():
    G = nx.path_graph(["alpha", "beta", "gamma", "delta", "epsilon"])
    assert_resume_equivalent("bfs_edges", G, "alpha")
    assert_resume_equivalent("dfs_edges", G, "alpha")


def test_larger_graph_sampled_cuts():
    G = nx.gnp_random_graph(60, 0.12, seed=42, directed=True)
    for kind in TRAVERSALS:
        source = sorted(G.nodes)[0]
        source = [source] if kind.startswith("edge") else source
        full = reference(kind, G, source)
        cuts = sorted({0, len(full) // 3, 2 * len(full) // 3, len(full)})
        assert_resume_equivalent(kind, G, source, cuts=cuts)


def test_yielded_and_last_item_and_algorithm():
    G = nx.DiGraph(EDGES)
    traversal = nx.resumable_bfs_edges(G, 0)
    assert traversal.algorithm == "bfs_edges"
    assert traversal.yielded == 0
    assert traversal.closed is False
    first = next(traversal)
    state = traversal.snapshot()
    assert state["yielded"] == 1
    assert state["last_item"] == first
    assert state["done"] is False
    resumed = nx.resumable_bfs_edges(rebuild(G), state=state)
    list(resumed)
    assert resumed.yielded == len(list(nx.bfs_edges(G, 0)))


def test_completed_traversal_resumes_empty():
    G = nx.DiGraph(EDGES)
    traversal = nx.resumable_dfs_edges(G, 0)
    list(traversal)
    state = pickle.loads(pickle.dumps(traversal.snapshot()))
    assert state["done"] is True
    resumed = nx.resumable_dfs_edges(rebuild(G), state=state)
    assert list(resumed) == []
    assert resumed.yielded == traversal.yielded


# ---------------------------------------------------------------------------
# Checkpoint rejection
# ---------------------------------------------------------------------------


@pytest.fixture
def edge_dfs_checkpoint():
    G = nx.MultiDiGraph(EDGES)
    traversal = nx.resumable_edge_dfs(G, [0, 1, 2, 3])
    next(traversal)
    return G, pickle.loads(pickle.dumps(traversal.snapshot()))


def test_reject_added_edge(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    changed = nx.MultiDiGraph(EDGES)
    changed.add_edge(4, 5)
    with pytest.raises(nx.NetworkXCheckpointError, match="has changed"):
        nx.resumable_edge_dfs(changed, state=state)


def test_reject_removed_edge(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    changed = nx.MultiDiGraph(EDGES)
    changed.remove_edge(1, 0, key=0)
    with pytest.raises(nx.NetworkXCheckpointError):
        nx.resumable_edge_dfs(changed, state=state)


def test_reject_added_node(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    changed = nx.MultiDiGraph(EDGES)
    changed.add_node(99)
    with pytest.raises(nx.NetworkXCheckpointError):
        nx.resumable_edge_dfs(changed, state=state)


def test_reject_removed_node(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    changed = nx.MultiDiGraph(EDGES)
    changed.remove_node(3)
    with pytest.raises(nx.NetworkXCheckpointError):
        nx.resumable_edge_dfs(changed, state=state)


def test_reject_changed_graph_type(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    with pytest.raises(nx.NetworkXCheckpointError, match="graph type"):
        nx.resumable_edge_dfs(nx.MultiGraph(EDGES), state=state)
    with pytest.raises(nx.NetworkXCheckpointError, match="graph type"):
        nx.resumable_edge_dfs(nx.DiGraph(EDGES), state=state)


def test_reject_incompatible_version(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    state["version"] = STATE_VERSION + 1
    with pytest.raises(nx.NetworkXCheckpointError, match="version"):
        nx.resumable_edge_dfs(nx.MultiDiGraph(EDGES), state=state)


def test_reject_other_traversal(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    with pytest.raises(nx.NetworkXCheckpointError, match="edge_bfs"):
        nx.resumable_edge_bfs(nx.MultiDiGraph(EDGES), state=state)


def test_reject_mismatched_orientation(edge_dfs_checkpoint):
    _, state = edge_dfs_checkpoint
    with pytest.raises(nx.NetworkXCheckpointError, match="orientation"):
        nx.resumable_edge_dfs(
            nx.MultiDiGraph(EDGES), orientation="reverse", state=state
        )


def test_reject_mismatched_reverse():
    traversal = nx.resumable_bfs_edges(nx.DiGraph(EDGES), 1, reverse=True)
    next(traversal)
    state = pickle.loads(pickle.dumps(traversal.snapshot()))
    with pytest.raises(nx.NetworkXCheckpointError, match="reverse"):
        nx.resumable_bfs_edges(nx.DiGraph(EDGES), reverse=False, state=state)


def test_reject_mismatched_depth_limit():
    traversal = nx.resumable_dfs_edges(nx.DiGraph(EDGES), 0, depth_limit=2)
    list(traversal)
    state = pickle.loads(pickle.dumps(traversal.snapshot()))
    with pytest.raises(nx.NetworkXCheckpointError, match="depth_limit"):
        nx.resumable_dfs_edges(nx.DiGraph(EDGES), depth_limit=3, state=state)
    # the recorded value is accepted
    resumed = nx.resumable_dfs_edges(nx.DiGraph(EDGES), depth_limit=2, state=state)
    assert list(resumed) == []


def test_reject_missing_sort_neighbors():
    G = nx.DiGraph(EDGES)
    traversal = nx.resumable_dfs_edges(G, 0, sort_neighbors=reverse_sorted)
    next(traversal)
    state = pickle.loads(pickle.dumps(traversal.snapshot()))
    with pytest.raises(nx.NetworkXCheckpointError, match="sort_neighbors"):
        nx.resumable_dfs_edges(rebuild(G), state=state)


def test_reject_unexpected_sort_neighbors():
    state = nx.resumable_bfs_edges(nx.DiGraph(EDGES), 0).snapshot()
    with pytest.raises(nx.NetworkXCheckpointError, match="sort_neighbors"):
        nx.resumable_bfs_edges(nx.DiGraph(EDGES), state=state, sort_neighbors=iter)


def test_reject_non_dict_state():
    with pytest.raises(nx.NetworkXCheckpointError, match="dict"):
        nx.resumable_bfs_edges(nx.DiGraph(EDGES), state=[1, 2, 3])


def test_reject_corrupt_state():
    state = nx.resumable_bfs_edges(nx.DiGraph(EDGES), 0).snapshot()
    del state["params"]
    with pytest.raises(nx.NetworkXCheckpointError):
        nx.resumable_bfs_edges(nx.DiGraph(EDGES), state=state)


def test_attribute_changes_are_not_structural():
    # Traversal order ignores edge/node attributes; only structure is guarded.
    traversal = nx.resumable_bfs_edges(nx.path_graph(5), 0)
    next(traversal)
    state = pickle.loads(pickle.dumps(traversal.snapshot()))
    G = nx.path_graph(5)
    G.nodes[2]["weight"] = 42
    G.edges[1, 2]["color"] = "red"
    resumed = nx.resumable_bfs_edges(G, state=state)
    assert list(resumed) == list(nx.bfs_edges(nx.path_graph(5), 0))[1:]


# ---------------------------------------------------------------------------
# Snapshot contents, isolation and graph resources
# ---------------------------------------------------------------------------


def test_snapshot_is_independent_of_later_iteration():
    G = nx.DiGraph(EDGES)
    traversal = nx.resumable_bfs_edges(G, 0)
    next(traversal)
    state = traversal.snapshot()
    list(traversal)
    assert state["yielded"] == 1
    full = list(nx.bfs_edges(rebuild(G), 0))
    resumed = nx.resumable_bfs_edges(
        rebuild(G), state=pickle.loads(pickle.dumps(state))
    )
    assert list(resumed) == full[1:]


def test_snapshot_holds_no_graph_data():
    G = nx.MultiDiGraph(EDGES)
    traversal = nx.resumable_edge_dfs(G, [0, 1, 2, 3])
    next(traversal)
    state = traversal.snapshot()
    # the checkpoint must serialize on its own, without dragging in the graph
    raw = pickle.dumps(state)
    assert b"AtlasView" not in raw and b"AdjacencyView" not in raw
    restored = pickle.loads(raw)
    assert restored["algorithm"] == "edge_dfs"


def test_close_stops_traversal():
    traversal = nx.resumable_bfs_edges(nx.DiGraph(EDGES), 0)
    next(traversal)
    traversal.close()
    assert traversal.closed is True
    with pytest.raises(StopIteration):
        next(traversal)
    with pytest.raises(nx.NetworkXError):
        traversal.snapshot()
    traversal.close()  # idempotent


def test_context_manager_closes_on_exception():
    G = nx.DiGraph(EDGES)
    traversal = nx.resumable_bfs_edges(G, 0)
    with pytest.raises(RuntimeError):
        with traversal:
            next(traversal)
            raise RuntimeError("abandon")
    assert traversal.closed is True


def test_close_releases_graph():
    G = nx.path_graph(40)
    ref = weakref.ref(G)
    traversal = nx.resumable_dfs_edges(G, 0)
    for _ in range(5):
        next(traversal)
    traversal.close()
    del G
    gc.collect()
    assert ref() is None


def test_dropping_traversal_releases_graph():
    G = nx.path_graph(20)
    ref = weakref.ref(G)
    traversal = nx.resumable_edge_bfs(G)
    next(traversal)
    del traversal, G
    gc.collect()
    assert ref() is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["edge_bfs", "edge_dfs", "dfs_edges"])
def test_empty_graph(kind):
    assert list(RESUMABLE[kind](nx.Graph())) == []


def test_missing_source_node():
    with pytest.raises(nx.NetworkXError):
        nx.resumable_bfs_edges(nx.Graph(), 0)
    with pytest.raises(nx.NetworkXError):
        nx.resumable_dfs_edges(nx.Graph(), 0)


def test_bfs_requires_source():
    with pytest.raises(TypeError):
        nx.resumable_bfs_edges(nx.path_graph(3))


def test_invalid_orientation():
    with pytest.raises(nx.NetworkXError):
        nx.resumable_edge_bfs(nx.DiGraph(EDGES), orientation="hello")


def test_restore_classmethod():
    G = nx.DiGraph(EDGES)
    traversal = nx.resumable_bfs_edges(G, 0)
    next(traversal)
    state = pickle.loads(pickle.dumps(traversal.snapshot()))
    resumed = ResumableTraversal.restore(rebuild(G), state)
    assert isinstance(resumed, ResumableTraversal)
    full = list(nx.bfs_edges(G, 0))
    assert list(resumed) == full[1:]


# ---------------------------------------------------------------------------
# Fingerprint properties
# ---------------------------------------------------------------------------


def test_fingerprint_equal_for_same_structure():
    edges = [(0, 1), (1, 2), (2, 0), (3, 1)]
    assert graph_fingerprint(nx.DiGraph(edges)) == graph_fingerprint(
        nx.DiGraph(reversed(edges))
    )
    assert graph_fingerprint(nx.MultiDiGraph(edges)) != graph_fingerprint(
        nx.DiGraph(edges)
    )


def test_fingerprint_independent_of_insertion_order_and_attributes():
    G1 = nx.MultiDiGraph()
    G1.add_edge("a", "b", key="x", weight=1)
    G1.add_edge("a", "b", key="y", weight=2)
    G1.add_edge("b", "c", key="z", weight=3)
    G2 = nx.MultiDiGraph()
    G2.add_edge("b", "c", key="z", weight=99)  # attribute differs
    G2.add_edge("a", "b", key="y")
    G2.add_edge("a", "b", key="x")
    assert graph_fingerprint(G1) == graph_fingerprint(G2)


def test_fingerprint_detects_key_change():
    G1 = nx.MultiDiGraph([(0, 1, "a")])
    G2 = nx.MultiDiGraph([(0, 1, "b")])
    assert graph_fingerprint(G1) != graph_fingerprint(G2)


# ---------------------------------------------------------------------------
# Genuine cross-process resumption
# ---------------------------------------------------------------------------


def _spawn_resume_worker(payload, queue):
    import networkx as nx

    state, num_nodes = pickle.loads(payload)
    tail = list(nx.resumable_bfs_edges(nx.path_graph(num_nodes), state=state))
    queue.put(tail)


def test_resume_in_spawned_process():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    G = nx.path_graph(20)
    traversal = nx.resumable_bfs_edges(G, 0)
    head = [next(traversal) for _ in range(7)]
    payload = pickle.dumps((traversal.snapshot(), 20))
    process = ctx.Process(target=_spawn_resume_worker, args=(payload, queue))
    process.start()
    try:
        tail = queue.get(timeout=60)
    finally:
        process.join()
    assert process.exitcode == 0
    assert head + tail == list(nx.bfs_edges(G, 0))
    assert len(tail) == 12
