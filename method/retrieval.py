from typing import Iterable, List, Optional

import numpy as np
from grakel import Graph as GraKelGraph
from grakel.kernels import ShortestPath

from graph import Graph


def empty_grakel_graph(num_nodes: int = 0) -> GraKelGraph:
    return GraKelGraph(np.zeros((num_nodes, num_nodes), dtype=np.int32), node_labels=None)


def build_grakel_graph(base_graph: Graph, nodes: Iterable[int]) -> GraKelGraph:
    nodes = sorted(set(int(node) for node in nodes))
    if len(nodes) <= 1:
        return empty_grakel_graph(len(nodes))
    coo = base_graph.adj_mat[nodes][:, nodes].tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    return GraKelGraph(edges if edges else np.zeros((len(nodes), len(nodes)), dtype=np.int32), node_labels=None)


def induce_community_graph(
    base_graph: Graph,
    community: Iterable[int],
    seed_raw: Optional[int] = None,
) -> GraKelGraph:
    local_nodes = []
    for node in community:
        local = base_graph.find_local(node, allow_local=False)
        if local is not None:
            local_nodes.append(local)
    local_nodes = sorted(set(local_nodes))

    if seed_raw is not None:
        seed_local = base_graph.find_local(seed_raw, allow_local=False)
        if seed_local is not None and seed_local in local_nodes:
            allowed, visited, stack = set(local_nodes), set(), [seed_local]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(
                    neighbor
                    for neighbor in base_graph.neighbors.get(node, set())
                    if neighbor in allowed and neighbor not in visited
                )
            local_nodes = sorted(visited)
    return build_grakel_graph(base_graph, local_nodes)


def select_topk(scores, communities: List[List[int]], k: int = 50) -> List[List[int]]:
    if scores is None or not communities:
        print("[Supervision Retrieval: TopK] no known communities available; relevant_comms=0", flush=True)
        return []

    scores = np.asarray(scores)
    if scores.ndim == 2:
        scores = scores[0] if scores.shape[0] else np.asarray([])
    elif scores.ndim != 1:
        print(f"[Supervision Retrieval: TopK] invalid scores shape={scores.shape}; relevant_comms=0", flush=True)
        return []

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    num_candidates = min(len(communities), int(scores.shape[0]))
    top_k = min(max(int(k), 0), num_candidates)
    if num_candidates <= 0 or top_k <= 0:
        print(
            f"[Supervision Retrieval: TopK] empty selection; relevant_comms=0/{num_candidates}",
            flush=True,
        )
        return []

    scores = scores[:num_candidates]
    selected_offsets = np.argsort(-scores, kind="mergesort")[:top_k]
    selected = [communities[int(offset)] for offset in selected_offsets]
    selected_scores = scores[selected_offsets]
    sizes = [len(community) for community in selected]
    preview_count = min(30, len(sizes))
    suffix = "" if len(sizes) <= preview_count else f" ... (+{len(sizes) - preview_count} more)"
    print(
        f"[Supervision Retrieval: TopK] mode=reference_to_candidate | top_k={top_k} | "
        f"relevant_comms={len(selected)}/{num_candidates} | "
        f"score_min={float(np.min(selected_scores)):.6f} | "
        f"score_mean={float(np.mean(selected_scores)):.6f} | "
        f"score_max={float(np.max(selected_scores)):.6f} | "
        f"size_min={int(np.min(sizes))} | "
        f"size_mean={float(np.mean(sizes)):.2f} | "
        f"size_max={int(np.max(sizes))} | "
        f"sizes={sizes[:preview_count]}{suffix}",
        flush=True,
    )
    return selected


def retrieve_topk(
    reference_graph: GraKelGraph,
    candidate_graphs: List[GraKelGraph],
    communities: List[List[int]],
    k: int,
    label: str,
) -> Optional[List[List[int]]]:
    if not candidate_graphs:
        print(f"[WARN] no candidate known communities for {label}; keep previous relevant_comms.")
        return []
    try:
        kernel = ShortestPath(normalize=True, with_labels=False)
        kernel.fit(candidate_graphs)
        scores = np.nan_to_num(kernel.transform([reference_graph])[0])
    except Exception as exc:
        print(f"[WARN] ShortestPath retrieval failed: {exc}; keep previous relevant_comms.")
        return None

    selected = select_topk(scores, communities, k)
    if not selected:
        print(f"[WARN] {label} returned empty; keep previous relevant_comms.")
    return selected
