"""Input helpers for static real-world graph datasets."""

import os

import networkx as nx


def load_graph(path):
    return nx.read_edgelist(path, nodetype=str, data=False)


def load_communities(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as handle:
        for cid, line in enumerate(handle):
            for node in line.split():
                labels.setdefault(node, set()).add(cid)
    return labels


def _read_tokens(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read().split()


def load_seed_context(dataset_root, enabled=True, limit=None):
    if not enabled:
        return [], {}
    seeds = _read_tokens(os.path.join(dataset_root, "t000_seed.txt"))
    indices = _read_tokens(os.path.join(dataset_root, "t000_seed_cmty_index.txt"))
    if len(seeds) != len(indices):
        raise ValueError(
            f"Seed/index length mismatch: {len(seeds)} seeds and {len(indices)} community indices."
        )
    if limit is not None:
        size = max(0, int(limit))
        seeds, indices = seeds[:size], indices[:size]
    return seeds, {seed: int(cid) for seed, cid in zip(seeds, indices)}
