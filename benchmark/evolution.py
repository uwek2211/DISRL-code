"""Expansion, shrinkage, and background perturbation rules."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from community_utils import build_cid2nodes


def _pick_communities(
    all_cids, required, fraction, head_k=0, tail_k=0, ordered_rest=True
):
    all_cids = sorted(int(cid) for cid in all_cids)
    if not all_cids:
        return set()
    head_k, tail_k = max(0, int(head_k or 0)), max(0, int(tail_k or 0))
    picked = (
        set(required)
        | set(all_cids[:head_k])
        | set(all_cids[-tail_k:] if tail_k else [])
    )
    target = max(1, int(len(all_cids) * fraction))
    rest = (
        [cid for cid in all_cids if cid not in picked]
        if ordered_rest
        else list(set(all_cids) - picked)
    )
    if len(picked) < target and rest:
        picked |= set(
            np.random.choice(
                rest, size=min(target - len(picked), len(rest)), replace=False
            )
        )
    return picked


def _scaled_count(size, ratio, cap=None, force_one=False):
    ratio = max(0.0, float(ratio) if ratio is not None else 0.0)
    count = int(size * ratio)
    if count == 0 and (ratio > 0 or force_one):
        count = 1
    if cap is not None:
        cap = int(cap)
        count = 0 if cap <= 0 else min(count, cap)
    return count


def add_nodes(graph, primary_labels):
    """Add a random node batch using the current community/degree patterns."""
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return [], []
    batch = max(1, int(n_nodes * 0.1))
    existing_nodes = list(graph.nodes())
    try:
        max_id = max(int(node) for node in existing_nodes)
        cast = str if isinstance(existing_nodes[0], str) else int
        new_nodes = [cast(i) for i in range(max_id + 1, max_id + 1 + batch)]
    except ValueError:
        existing_set, new_nodes, idx = {str(node) for node in existing_nodes}, [], 0
        while len(new_nodes) < batch:
            if f"new_{idx}" not in existing_set:
                new_nodes.append(f"new_{idx}")
            idx += 1
    comm_sizes = dict(Counter(primary_labels.values()).most_common())
    communities = random.choices(
        list(comm_sizes), weights=list(comm_sizes.values()), k=batch
    )
    nodes_to_add = list(zip(new_nodes, communities))
    inner_com_nodes = defaultdict(list)
    for node, cid in primary_labels.items():
        inner_com_nodes[cid].append(node)
    inner_com_nodes = dict(sorted(inner_com_nodes.items()))
    inner_com_degree, outer_com_degree, outer_com_nodes = {}, {}, {}
    all_nodes_set = set(graph.nodes())
    for cid, inner_nodes in inner_com_nodes.items():
        inner_set = set(inner_nodes)
        outer_com_nodes[cid] = list(all_nodes_set - inner_set)
        inner_com_degree[cid], outer_com_degree[cid] = [], []
        for node in inner_nodes:
            if node not in graph:
                continue
            neighbors = list(graph.neighbors(node))
            k_in = sum(primary_labels.get(neighbor) == cid for neighbor in neighbors)
            k_out = len(neighbors) - k_in
            inner_com_degree[cid].append(k_in)
            outer_com_degree[cid].append(k_out)
    edges_to_add = []
    for node, cid in nodes_to_add:
        if not inner_com_degree.get(cid):
            continue
        for degrees, destinations in (
            (inner_com_degree[cid], inner_com_nodes[cid]),
            (outer_com_degree[cid], outer_com_nodes.get(cid, [])),
        ):
            values, frequency = np.unique(degrees, return_counts=True)
            count = random.choices(values, weights=frequency)[0]
            if destinations and count > 0:
                edges_to_add.extend(
                    (node, dst) for dst in np.random.choice(destinations, size=count)
                )
    return [(node, {"community": cid}) for node, cid in nodes_to_add], edges_to_add


def remove_edges(graph):
    """Randomly remove about 10% of edges."""
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return []

    batch = max(1, int(n_edges * 0.1))
    edges = list(graph.edges())
    batch = min(batch, len(edges))

    idx = np.random.choice(len(edges), size=batch, replace=False)
    return [edges[i] for i in idx]


def remove_nodes(graph):
    """Randomly remove about 10% of nodes."""
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return []

    batch = max(1, int(n_nodes * 0.1))
    nodes = list(graph.nodes())
    batch = min(batch, len(nodes))

    return list(np.random.choice(nodes, size=batch, replace=False))


class ExpandRule:
    """Rule-level community expansion. Seeds are optional."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def step(self, sim, graph, t):
        apply_expand(sim, graph, getattr(sim, "seeds", []), **self.kwargs)


class ShrinkRule:
    """Rule-level community shrinkage. Seeds are optional."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def step(self, sim, graph, t):
        apply_shrink(sim, graph, getattr(sim, "seeds", []), **self.kwargs)


def _remove_isolates(sim, graph, protect_seeds, min_size):
    isolated = [str(node) for node in nx.isolates(graph)]
    if protect_seeds:
        isolated = [node for node in isolated if node not in sim.seeds_str]
    isolated = sim._limit_deletion_in_seed_communities(
        graph, isolated, sim.seeds, min_size=min_size
    )
    graph.remove_nodes_from(isolated)


class _NoiseRule:
    def __init__(self, remove_isolated=True, protect_seeds=True, min_size=3):
        self.remove_isolated = remove_isolated
        self.protect_seeds = protect_seeds
        self.min_size = min_size

    def _finish(self, sim, graph):
        if self.remove_isolated:
            _remove_isolates(sim, graph, self.protect_seeds, self.min_size)
        sim._refresh_from_graph(graph)


class AddNodesRule(_NoiseRule):
    def step(self, sim, graph, t):
        nodes_to_add, edges_to_add = add_nodes(graph, sim.noise_labels)

        graph.add_nodes_from(nodes_to_add)
        graph.add_edges_from(edges_to_add)

        for node, attr in nodes_to_add:
            nid = str(node)
            if nid not in graph:
                continue
            if isinstance(attr, dict) and "community" in attr:
                cid = int(attr["community"])
                sim.orig_comm[nid] = cid
                sim.multi_labels.setdefault(nid, set()).add(cid)

        self._finish(sim, graph)


class RemoveNodesRule(_NoiseRule):
    def step(self, sim, graph, t):
        nodes_to_remove = [str(node) for node in remove_nodes(graph)]

        if self.protect_seeds:
            nodes_to_remove = [
                node for node in nodes_to_remove if node not in sim.seeds_str
            ]

        nodes_to_remove = sim._limit_deletion_in_seed_communities(
            graph,
            nodes_to_remove,
            sim.seeds,
            min_size=self.min_size,
        )

        graph.remove_nodes_from(nodes_to_remove)
        self._finish(sim, graph)


class RemoveEdgesRule(_NoiseRule):
    def step(self, sim, graph, t):
        graph.remove_edges_from(remove_edges(graph))
        self._finish(sim, graph)


def apply_expand(
    sim,
    graph,
    seeds=None,
    comm_frac=0.5,
    add_ratio=0.1,
    max_add_per_comm=3,
    edge_frac=0.1,
    max_edges_per_new=3,
    force_head_k: int = 0,
    force_tail_k: int = 0,
):
    """Expand selected communities by adding new labeled nodes."""
    seeds = [str(seed) for seed in (seeds or [])]
    nodes_in_graph = {str(node) for node in graph.nodes()}

    if not hasattr(sim, "_next_global_node_id") or sim._next_global_node_id is None:
        max_id = -1
        for node in nodes_in_graph:
            try:
                max_id = max(max_id, int(node))
            except Exception:
                pass
        sim._next_global_node_id = max_id + 1 if max_id >= 0 else 0

    cid2nodes = build_cid2nodes(sim.multi_labels, nodes_in_graph, ordered=True)
    if not cid2nodes:
        return graph
    frac = float(comm_frac) if comm_frac is not None else 0.0
    frac = max(0.0, min(1.0, frac))
    seed_cids = {
        int(cid)
        for seed in seeds
        if seed in nodes_in_graph
        for cid in sim.multi_labels.get(seed, set())
    }
    picked = _pick_communities(cid2nodes, seed_cids, frac, force_head_k, force_tail_k)
    if not picked:
        return graph

    for cid in sorted(picked):
        cid = int(cid)
        comm_nodes = cid2nodes.get(cid, [])
        if not comm_nodes:
            continue

        comm_size = len(comm_nodes)
        n_new = _scaled_count(comm_size, add_ratio, max_add_per_comm)
        if n_new <= 0:
            continue
        m_edges = min(_scaled_count(comm_size, edge_frac, max_edges_per_new), comm_size)

        for _ in range(n_new):
            while True:
                new_id = str(sim._next_global_node_id)
                sim._next_global_node_id += 1
                if new_id not in nodes_in_graph and new_id not in graph:
                    break

            graph.add_node(new_id)
            nodes_in_graph.add(new_id)

            sim.multi_labels[new_id] = {cid}
            sim.orig_comm[new_id] = cid

            if m_edges <= 0:
                continue
            selected = []
            core_nodes = [
                node
                for node in comm_nodes
                if sim.multi_labels.get(node, set()) == {cid}
            ]
            if core_nodes:
                selected.append(random.choice(core_nodes))
            seeds_in_cid = [
                seed
                for seed in seeds
                if seed in nodes_in_graph and cid in sim.multi_labels.get(seed, set())
            ]
            if seeds_in_cid and len(selected) < m_edges:
                seed_pick = random.choice(seeds_in_cid)
                if seed_pick not in selected:
                    selected.append(seed_pick)
            need = m_edges - len(selected)
            if need > 0:
                candidates = [node for node in comm_nodes if node not in selected]
                if candidates:
                    weights = np.array(
                        [
                            1.0 / max(1, len(sim.multi_labels.get(node, set())))
                            for node in candidates
                        ],
                        dtype=float,
                    )
                    probs = None if weights.sum() <= 0 else (weights / weights.sum())
                    k = min(need, len(candidates))
                    picked_nodes = (
                        random.sample(candidates, k=k)
                        if probs is None
                        else list(
                            np.random.choice(candidates, size=k, replace=False, p=probs)
                        )
                    )
                    selected.extend(picked_nodes)
            graph.add_edges_from(
                (new_id, node) for node in selected if not graph.has_edge(new_id, node)
            )

    return graph


def apply_shrink(
    sim,
    graph,
    seeds=None,
    comm_frac=0.5,
    shrink_frac=0.1,
    max_rm_per_comm=3,
    min_comm_size=3,
    remove_isolated=True,
    force_head_k: int = 0,
    force_tail_k: int = 0,
):
    """Shrink selected communities by removing weak internal members."""
    seeds = set(str(seed) for seed in (seeds or []))
    protected_nodes = sim.get_protected_nodes(include_seeds=True, include_tail=True)
    nodes_in_graph = {str(node) for node in graph.nodes()}

    cid2nodes = build_cid2nodes(sim.multi_labels, nodes_in_graph, ordered=True)
    if not cid2nodes:
        return graph
    seed_cids = set().union(*(sim.multi_labels.get(seed, set()) for seed in seeds))
    picked = _pick_communities(
        cid2nodes, seed_cids, comm_frac, force_head_k, force_tail_k, ordered_rest=False
    )

    neighbors = {
        str(node): {str(neighbor) for neighbor in graph.neighbors(node)}
        for node in graph.nodes()
    }
    node_pressure = defaultdict(set)

    for cid in picked:
        comm_nodes = cid2nodes.get(int(cid), [])
        if len(comm_nodes) <= min_comm_size:
            continue

        candidates = [node for node in comm_nodes if node not in protected_nodes]
        if not candidates:
            continue

        comm_set = set(comm_nodes)
        scored = [
            (node, len(neigh & comm_set), len(neigh))
            for node in candidates
            if (neigh := neighbors.get(node, set()))
        ]
        if not scored:
            continue
        scored.sort(key=lambda item: (item[1], item[2]))
        n_rm_target = _scaled_count(
            len(scored), shrink_frac, max_rm_per_comm, force_one=True
        )
        n_rm = min(n_rm_target, len(comm_nodes) - min_comm_size)
        if n_rm <= 0:
            continue
        for node, _, _ in scored[:n_rm]:
            node_pressure[node].add(cid)

    final_remove = defaultdict(set)
    for node, cids in node_pressure.items():
        for cid in cids:
            final_remove[cid].add(node)

    for cid, nodes in final_remove.items():
        comm_nodes = [
            node
            for node, cids in sim.multi_labels.items()
            if node in nodes_in_graph and cid in cids
        ]
        if len(comm_nodes) - len(nodes) < min_comm_size:
            continue

        comm_set = set(comm_nodes)
        for node in nodes:
            if node in graph:
                edges = [
                    (node, neighbor)
                    for neighbor in graph.neighbors(node)
                    if str(neighbor) in comm_set
                ]
                graph.remove_edges_from(edges)
            labels = sim.multi_labels.get(node, set())
            labels.discard(cid)
            sim.multi_labels[node] = labels
            if not labels:
                if node in graph and node not in seeds:
                    graph.remove_node(node)
                sim.multi_labels.pop(node, None)
                sim.orig_comm.pop(node, None)

    if remove_isolated:
        sim._remove_isolated_nodes(graph, protect_seeds=True)

    return graph
