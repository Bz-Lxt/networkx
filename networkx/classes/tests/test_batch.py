"""Tests for transactional batch updates (`Graph.batch`).

A batch update must either commit every change (clearing the converted-graph
cache once) or, when any input item is invalid or an iterator raises partway
through, restore the exact pre-call structure, attributes, cache and the
state visible through outstanding views. Batches can be nested and must not
change the immediate, item-by-item behavior of the singular add/remove
methods outside of a batch.
"""

import copy

import pytest

import networkx as nx

GRAPH_TYPES = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def make_graph(cls):
    G = cls(name="g", meta={"x": 1})
    if G.is_multigraph():
        G.add_edges_from(
            [
                (0, 1, 0, {"color": "r"}),
                (0, 1, 1, {"color": "g"}),
                (1, 2, {"color": "b"}),
            ]
        )
    else:
        G.add_edges_from([(0, 1, {"color": "r"}), (1, 2, {"color": "b"})])
    G.add_node(7, label="seven")
    return G


def structures(G):
    """Deep copy of every mutable structure a batch may touch."""
    result = {
        "node": copy.deepcopy(G._node),
        "adj": copy.deepcopy(G._adj),
        "graph": copy.deepcopy(G.graph),
    }
    if G.is_directed():
        result["pred"] = copy.deepcopy(G._pred)
    return result


def raising_iterator(items, exc=None):
    yield from items
    raise exc if exc is not None else RuntimeError("boom")


# ---------------------------------------------------------------------------
# Invalid input and failing iterators: full rollback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_add_edges_from_invalid_tuple_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    bad_edge = (3, 4, 5, 6, 7) if G.is_multigraph() else (3, 4, 5, 6)
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(2, 3, {"new": True}), bad_edge])
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_add_edges_from_iterator_error_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError):
        G.add_edges_from(raising_iterator([(2, 3, {"new": True}), (4, 5)]))
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_add_nodes_from_none_node_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(ValueError):
        G.add_nodes_from([(8, {"a": 1}), None])
    assert structures(G) == before
    # an attribute update on a pre-existing node must be rolled back too
    with pytest.raises(ValueError):
        G.add_nodes_from([(7, {"label": "changed"}), None])
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_add_nodes_from_iterator_error_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError):
        G.add_nodes_from(raising_iterator([(8, {"a": 1}), 9]))
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_remove_edges_from_midway_error_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    # remove an existing edge first, then an item that fails while unpacking
    with pytest.raises((TypeError, ValueError)):
        G.remove_edges_from(raising_iterator([(0, 1), (0,)], ValueError("x")))
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_remove_nodes_from_iterator_error_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError):
        G.remove_nodes_from(raising_iterator([0, 1]))
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_add_weighted_edges_from_iterator_error_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError):
        G.add_weighted_edges_from(raising_iterator([(2, 3, 1.5), (4, 5, 2.5)]))
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_update_rolls_back(cls):
    G = make_graph(cls)
    before = structures(G)
    bad = (3, 4, 5, 6) if not G.is_multigraph() else (1, 2, 3, 4, 5)
    with pytest.raises(nx.NetworkXError):
        G.update(edges=[(2, 3), bad], nodes=[(8, {"a": 1})])
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_graph_attribute_changes_roll_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError), G.batch():
        G.graph["new"] = 1
        del G.graph["name"]
        raise RuntimeError
    assert structures(G) == before


# ---------------------------------------------------------------------------
# Successful batches commit in the existing order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_successful_add_edges_commits_in_order(cls):
    G = make_graph(cls)
    if G.is_multigraph():
        keys = G.add_edges_from([(2, 3, "k", {"w": 1}), (3, 4)], weight=2)
        assert keys == ["k", 0]
        assert G[2][3]["k"] == {"w": 1, "weight": 2}
        assert G[3][4][0] == {"weight": 2}
    else:
        G.add_edges_from([(2, 3, {"w": 1}), (3, 4)], weight=2)
        assert G[2][3] == {"w": 1, "weight": 2}
        assert G[3][4] == {"weight": 2}
    # committed in insertion order; pre-existing nodes keep their position
    assert list(G.nodes) == [0, 1, 2, 7, 3, 4]
    assert 8 not in G


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_successful_remove_commits(cls):
    G = make_graph(cls)
    G.remove_edges_from([(0, 1)])
    if G.is_multigraph():
        # the 2-tuple removes the most recent edge between 0 and 1
        assert set(G.edges(keys=True)) == {(0, 1, 0), (1, 2, 0)}
    else:
        assert not G.has_edge(0, 1)
    G.remove_nodes_from([7])
    assert 7 not in G


def test_multigraph_edge_key_counter_restored():
    G = make_graph(nx.MultiGraph)
    assert G.new_edge_key(0, 1) == 2
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(0, 1, {"w": 1}), (9, 9, 1, 2, 3)])
    # no phantom edges, and key allocation continues exactly where it stopped
    assert G.new_edge_key(0, 1) == 2
    assert list(G.edges(keys=True)) == [(0, 1, 0), (0, 1, 1), (1, 2, 0)]
    assert 9 not in G


def test_multidigraph_pred_succ_consistent_after_rollback():
    G = make_graph(nx.MultiDiGraph)
    before = structures(G)
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(2, 3, "k", {"w": 1}), (3, 4, 5, 6, 7)])
    assert structures(G) == before
    assert set(G._succ) == set(G._pred) == {0, 1, 2, 7}
    assert set(G._succ[2]) == set()
    assert set(G._pred[3]) == set() if 3 in G._pred else True
    assert 3 not in G._succ and 3 not in G._pred


# ---------------------------------------------------------------------------
# Structural aliases must remain aliases after rollback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_shared_dictionaries_stay_shared_after_rollback(cls):
    G = make_graph(cls)
    bad = (1, 2, 3, 4, 5) if G.is_multigraph() else (1, 2, 3, 4)
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(2, 3, {"w": 1}), bad])

    def edge_data(u, v):
        return G._adj[u][v][0] if G.is_multigraph() else G._adj[u][v]

    if G.is_directed():
        # edge data dicts are shared between _succ and _pred
        if G.is_multigraph():
            assert G._succ[0][1][0] is G._pred[1][0][0]
            assert G._succ[1][2][0] is G._pred[2][1][0]
            forward, backward = G._succ[0][1][0], G._pred[1][0][0]
        else:
            assert G._succ[0][1] is G._pred[1][0]
            assert G._succ[1][2] is G._pred[2][1]
            forward, backward = G._succ[0][1], G._pred[1][0]
    else:
        # edge (data/key) dictionaries are shared by both endpoints
        assert G._adj[0][1] is G._adj[1][0]
        forward = backward = edge_data(0, 1)
    # mutating through one side stays visible through the other
    forward["added_after"] = 1
    assert backward["added_after"] == 1
    del forward["added_after"]


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_cache_cleared_once_on_commit(cls, monkeypatch):
    G = make_graph(cls)
    calls = 0
    original = nx._clear_cache

    def counting(g):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(nx, "_clear_cache", counting)
    try:
        G.add_edges_from([(2, 3)])
        assert calls == 1
        with G.batch():
            G.add_node(8)
            G.add_edge(8, 2)
            G.add_edges_from([(2, 9)])
            with G.batch():  # nested batch joins the outer transaction
                G.add_node(10)
                G.graph["nested"] = True
        assert calls == 2
    finally:
        monkeypatch.setattr(nx, "_clear_cache", original)
    assert set(G.nodes) >= {8, 9, 10}


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_cache_restored_on_failure(cls):
    G = make_graph(cls)
    sentinel = object()
    G.__networkx_cache__["sentinel"] = sentinel
    bad = (3, 4, 5, 6, 7) if G.is_multigraph() else (3, 4, 5, 6)
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(2, 3), bad])
    assert G.__networkx_cache__.get("sentinel") is sentinel
    assert G._batch_state is None
    G.add_edges_from([(2, 3)])
    assert "sentinel" not in G.__networkx_cache__


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_batch_works_with_caching_disabled(cls):
    # some algorithms (e.g. maxflow) set the cache to None on purpose
    G = cls()
    G.__networkx_cache__ = None
    G.add_edges_from([(0, 1), (1, 2)])
    edges = list(G.edges(keys=True)) if G.is_multigraph() else list(G.edges)
    assert edges == ([(0, 1, 0), (1, 2, 0)] if G.is_multigraph() else [(0, 1), (1, 2)])
    assert G.__networkx_cache__ is None
    before = structures(G)
    bad = (3, 4, 5, 6, 7) if G.is_multigraph() else (3, 4, 5, 6)
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(2, 3), bad])
    assert structures(G) == before
    assert G.__networkx_cache__ is None


# ---------------------------------------------------------------------------
# Outstanding views must stay valid and show the rolled-back state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_outstanding_views_reflect_rollback(cls):
    G = make_graph(cls)
    multi = G.is_multigraph()

    def listed(view):
        return list(view(keys=True)) if multi else list(view)

    nodes_view = G.nodes
    edges_view = G.edges
    adj_view = G.adj
    sub = G.subgraph([0, 1, 7])
    expected_nodes = list(nodes_view)
    expected_edges = listed(edges_view)
    expected_sub = listed(sub.edges)
    bad = (3, 4, 5, 6, 7) if multi else (3, 4, 5, 6)
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(2, 3), bad])
    # the very same view objects now report the restored state
    assert G.nodes is nodes_view
    assert G.edges is edges_view
    assert G.adj is adj_view
    assert list(nodes_view) == expected_nodes
    assert listed(edges_view) == expected_edges
    assert listed(sub.edges) == expected_sub
    assert list(adj_view[1]) == list(G._adj[1])
    # a fresh edge added after the rollback is visible through the old views
    G.add_edge(7, 0, 0) if multi else G.add_edge(7, 0)
    assert 0 in adj_view[7]
    if G.is_directed():
        assert 7 in G.pred[0]
    else:
        assert 7 in adj_view[0]


# ---------------------------------------------------------------------------
# Nested batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_nested_batch_outer_failure_rolls_everything_back(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError), G.batch():
        G.add_node(8)
        G.add_edges_from([(2, 3)])
        with G.batch():
            G.add_node(9)
            G.add_edges_from([(3, 4)])
        raise RuntimeError
    assert structures(G) == before
    assert G._batch_state is None


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_nested_batch_inner_failure_propagates(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(nx.NetworkXError), G.batch():
        G.add_node(8)
        with G.batch():
            bad = (3, 4, 5, 6, 7) if G.is_multigraph() else (3, 4, 5, 6)
            G.add_edges_from([(2, 3), bad])
    assert structures(G) == before
    assert G._batch_state is None


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_batches_can_be_reused(cls):
    G = make_graph(cls)
    with G.batch():
        G.add_node(8)
    assert G._batch_state is None
    with G.batch():
        G.add_node(9)
    assert G._batch_state is None
    assert list(G.nodes)[-2:] == [8, 9]


# ---------------------------------------------------------------------------
# Default item-by-item semantics are unchanged outside a batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_singular_methods_commit_immediately_outside_batch(cls):
    G = make_graph(cls)
    G.__networkx_cache__["sentinel"] = 1
    G.add_node(11)
    assert "sentinel" not in G.__networkx_cache__
    G.__networkx_cache__["sentinel"] = 1
    G.add_edge(11, 7) if not G.is_multigraph() else G.add_edge(11, 7, 0)
    assert "sentinel" not in G.__networkx_cache__


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_singular_mutations_visible_immediately_inside_batch(cls):
    G = make_graph(cls)
    with G.batch():
        G.add_node(11)
        assert 11 in G  # visible, but the cache is not cleared per item
        G.__networkx_cache__["sentinel"] = 1
        G.add_edge(11, 7) if not G.is_multigraph() else G.add_edge(11, 7, 0)
        assert G.__networkx_cache__.get("sentinel") == 1
    assert "sentinel" not in G.__networkx_cache__


# ---------------------------------------------------------------------------
# Attribute helper functions
# ---------------------------------------------------------------------------


class _BoomMapping:
    """Mapping-like object whose ``items`` raises after yielding once."""

    def __init__(self, first_item):
        self.key, self.value = first_item

    def __getitem__(self, key):
        if key == self.key:
            return self.value
        raise KeyError(key)

    def items(self):
        yield self.key, self.value
        raise RuntimeError("boom")


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_set_node_attributes_rollback_on_error(cls):
    G = make_graph(cls)
    before = structures(G)
    with pytest.raises(RuntimeError):
        nx.set_node_attributes(G, _BoomMapping((0, {"color": "zzz"})))
    assert structures(G) == before
    with pytest.raises(RuntimeError):
        nx.set_node_attributes(G, _BoomMapping((7, "zzz")), name="label")
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_set_edge_attributes_rollback_on_error(cls):
    G = make_graph(cls)
    before = structures(G)
    if G.is_multigraph():
        first = ((0, 1, 0), {"color": "zzz"})
    else:
        first = ((0, 1), {"color": "zzz"})
    with pytest.raises(RuntimeError):
        nx.set_edge_attributes(G, _BoomMapping(first))
    assert structures(G) == before


def test_set_edge_attributes_wrong_tuple_shape_rolls_back():
    G = make_graph(nx.MultiGraph)
    before = structures(G)
    # 2-tuple keys are invalid for multigraphs (3-tuples with key expected)
    with pytest.raises(ValueError):
        nx.set_edge_attributes(G, {(0, 1): 5, (1, 2): 6}, name="w")
    assert structures(G) == before


@pytest.mark.parametrize("cls", GRAPH_TYPES)
def test_set_node_attributes_commit(cls):
    G = make_graph(cls)
    nx.set_node_attributes(G, {7: "v2", 0: "v0"}, name="label")
    assert G.nodes[7]["label"] == "v2"
    assert G.nodes[0]["label"] == "v0"
    nx.set_node_attributes(G, "same", name="grp")
    assert all(G.nodes[n]["grp"] == "same" for n in G.nodes)


class _ExplodingAttrDict(dict):
    def __delitem__(self, key):
        if key == "boom":
            raise RuntimeError("boom")
        super().__delitem__(key)


class _ExplodingGraph(nx.Graph):
    node_attr_dict_factory = _ExplodingAttrDict
    edge_attr_dict_factory = _ExplodingAttrDict


class _ExplodingMultiGraph(nx.MultiGraph):
    node_attr_dict_factory = _ExplodingAttrDict
    edge_attr_dict_factory = _ExplodingAttrDict


@pytest.mark.parametrize("cls", [_ExplodingGraph, _ExplodingMultiGraph])
def test_remove_node_attributes_rollback_on_error(cls):
    G = cls()
    G.add_edge(0, 1, good=1, boom=2)
    G.add_node(2, good=3, boom=4)
    before = structures(G)
    with pytest.raises(RuntimeError):
        nx.remove_node_attributes(G, "good", "boom")
    assert structures(G) == before


@pytest.mark.parametrize("cls", [_ExplodingGraph, _ExplodingMultiGraph])
def test_remove_edge_attributes_rollback_on_error(cls):
    G = cls()
    G.add_edge(0, 1, good=1, boom=2)
    before = structures(G)
    with pytest.raises(RuntimeError):
        nx.remove_edge_attributes(G, "good", "boom")
    assert structures(G) == before


@pytest.mark.parametrize("cls", [nx.Graph, nx.MultiGraph])
def test_remove_attributes_commit(cls):
    G = cls()
    G.add_edge(0, 1, color="r", weight=1)
    G.add_node(2, color="b", size=3)
    nx.remove_node_attributes(G, "color")
    assert "color" not in G.nodes[2]
    assert G.nodes[2]["size"] == 3
    nx.remove_edge_attributes(G, "weight")
    data = G[0][1][0] if G.is_multigraph() else G[0][1]
    assert "weight" not in data
    assert data["color"] == "r"
