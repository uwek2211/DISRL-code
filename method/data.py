import os
from typing import List, Dict, Optional

import networkx as nx
import numpy as np

from graph import Graph


def list_snapshots(dataset_root: str) -> List[str]:
    """List snapshot directories in temporal order."""
    snap_root = os.path.join(dataset_root, "snapshots")
    if not os.path.isdir(snap_root):
        raise FileNotFoundError(f"snapshot root not found: {snap_root}")

    names = []
    for name in os.listdir(snap_root):
        p = os.path.join(snap_root, name)
        if os.path.isdir(p):
            names.append(name)

    def _key(x: str):
        try:
            return int("".join(ch for ch in x if ch.isdigit()))
        except Exception:
            return x

    return sorted(names, key=_key)


def _read_tokens(path: str):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return handle.read().split()


def _dataset_paths(root: str, dataset: str, snapshot: Optional[str]):
    if snapshot is None:
        base = os.path.join(root, dataset, f"{dataset}-1.90")
        return f"{base}.ungraph.txt", f"{base}.cmty.txt"
    base = os.path.join(root, dataset, "snapshots", snapshot, snapshot)
    return f"{base}_ungraph.txt", f"{base}_cmty.txt"


def load_graph_communities(root: str, dataset: str, snapshot: Optional[str] = None):
    edge_path, community_path = _dataset_paths(root, dataset, snapshot)
    with open(edge_path, encoding="utf-8") as handle:
        edges = np.asarray(
            [[int(value) for value in line.split()] for line in handle if line.strip()],
            dtype=np.int64,
        ).reshape(-1, 2)
    with open(community_path, encoding="utf-8") as handle:
        communities = [[int(value) for value in line.split()] for line in handle if line.strip()]
    nodes = set(edges.reshape(-1).tolist()) if edges.size else set()
    nodes.update(node for community in communities for node in community)
    return Graph(edges, nodes=nodes), communities


def build_snapshot_cache(args, root: str, dataset: str, snapshot: Optional[str] = None) -> Dict:
    graph, communities = load_graph_communities(root, dataset, snapshot)
    train_size = max(0, int(args.train_size))
    known_indices = list(range(max(0, len(communities) - train_size), len(communities)))
    if dataset == "twitter" and snapshot is None:
        edge_path, _ = _dataset_paths(root, dataset, snapshot)
        static_graph = nx.read_edgelist(edge_path, nodetype=int, comments="#")
        known_indices = [
            index for index in known_indices
            if nx.is_connected(static_graph.subgraph(communities[index]))
        ]
    return {
        "graph": graph,
        "raw_coms": communities,
        "base_known_community_indices": known_indices,
        "coms_local": [graph.to_local_nodes(community, allow_local=False) for community in communities],
    }


def read_fixed_seeds(dataset_root: str, snapshot0: str = "t000") -> List[int]:
    """Read the fixed target seeds from the initial snapshot."""
    p = os.path.join(dataset_root, "snapshots", snapshot0, f"{snapshot0}_seed.txt")
    seeds = [int(value) for value in _read_tokens(p)]
    if not seeds:
        raise FileNotFoundError(f"Cannot read seeds from: {p}")
    return seeds


def _read_seed_cmty_index(path: str, seeds: List[int]) -> Dict[int, int]:
    """Read either an index list or explicit seed-index pairs."""
    mapping: Dict[int, int] = {}
    parts = _read_tokens(path)
    if not parts:
        return mapping

    if len(parts) == len(seeds):
        for i, idx in enumerate(parts):
            mapping[seeds[i]] = int(idx)
        return mapping

    if len(parts) % 2 == 0:
        for i in range(0, len(parts), 2):
            s = int(parts[i])
            idx = int(parts[i + 1])
            mapping[s] = idx
        return mapping

    limit = min(len(parts), len(seeds))
    for i in range(limit):
        mapping[seeds[i]] = int(parts[i])
    return mapping


def read_seed_cmty_index(dataset_root: str, snapshot: str, seeds: List[int]) -> Dict[int, int]:
    """Read the seed-to-community mapping for one snapshot."""
    p = os.path.join(dataset_root, "snapshots", snapshot, f"{snapshot}_seed_cmty_index.txt")
    return _read_seed_cmty_index(p, seeds)
