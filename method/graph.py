from __future__ import annotations

from typing import Union, Optional, List, Set, Dict, Iterable, Tuple
import collections
import random

import numpy as np
from scipy import sparse as sp

IntLike = Union[int, np.integer]


class Graph:
    """Undirected graph wrapper with contiguous local node indexing."""

    def __init__(self, edges: np.ndarray, nodes: Optional[Iterable[IntLike]] = None):
        """Build a graph from raw edges, optionally preserving isolated nodes."""
        edges = np.asarray(edges, dtype=np.int64)
        if edges.size == 0:
            edges = edges.reshape(0, 2)

        raw_nodes: Set[int] = set()
        if nodes is not None:
            raw_nodes.update(int(x) for x in nodes)
        if edges.size:
            raw_nodes.update(int(x) for x in edges.reshape(-1).tolist())

        self._raw_nodes_sorted: List[int] = sorted(raw_nodes)
        self.n_nodes: int = len(self._raw_nodes_sorted)

        self.node_to_idx: Dict[int, int] = {raw: i for i, raw in enumerate(self._raw_nodes_sorted)}
        self.idx_to_node: Dict[int, int] = {i: raw for raw, i in self.node_to_idx.items()}

        self._mapped: bool = (self._raw_nodes_sorted != list(range(self.n_nodes)))

        if edges.size:
            local_edges = np.array(
                [[self.node_to_idx[int(u)], self.node_to_idx[int(v)]] for u, v in edges],
                dtype=np.int64,
            )
        else:
            local_edges = np.zeros((0, 2), dtype=np.int64)

        self.neighbors, self.adj_mat = self._init_from_edges(local_edges, n_nodes=self.n_nodes)

        self.parentGraph: Optional["Graph"] = None
        self.parent_to_local: Dict[int, int] = {}
        self.local_to_parent: Dict[int, int] = {}

    def to_local(self, node: IntLike, *, allow_local: bool = True) -> int:
        """Convert a raw node ID to its local index."""
        n = int(node)
        if not self._mapped:
            return n
        if allow_local and 0 <= n < self.n_nodes and n in self.idx_to_node:
            return n
        if n not in self.node_to_idx:
            raise KeyError(f"Raw node id {n} not present in this graph.")
        return self.node_to_idx[n]

    def find_local(self, node: IntLike, *, allow_local: bool = True):
        try:
            return self.to_local(node, allow_local=allow_local)
        except (KeyError, TypeError, ValueError):
            return None

    def to_raw(self, node_local: IntLike, *, allow_raw: bool = True) -> int:
        """Convert a local index back to its raw node ID."""
        n = int(node_local)
        if not self._mapped:
            return n
        if 0 <= n < self.n_nodes:
            return self.idx_to_node[n]
        if allow_raw and n in self.node_to_idx:
            return n
        raise KeyError(f"Local node id {n} not present in mapping.")

    def to_local_nodes(self, nodes: Iterable[IntLike], *, allow_local: bool = True) -> List[int]:
        return [self.to_local(x, allow_local=allow_local) for x in nodes]

    def to_raw_nodes(self, nodes_local: Iterable[IntLike], *, allow_raw: bool = True) -> List[int]:
        return [self.to_raw(x, allow_raw=allow_raw) for x in nodes_local]

    @staticmethod
    def _init_from_edges(edges_local: np.ndarray, n_nodes: int) -> Tuple[Dict[int, Set[int]], sp.csr_matrix]:
        """Build adjacency structures for a local-indexed graph."""
        neighbors: Dict[int, Set[int]] = collections.defaultdict(set)

        rows: List[int] = []
        cols: List[int] = []
        if edges_local.size:
            for u, v in edges_local:
                u = int(u)
                v = int(v)
                if u == v:
                    neighbors[u].add(v)
                    rows.append(u)
                    cols.append(v)
                    continue
                neighbors[u].add(v)
                neighbors[v].add(u)
                rows.extend([u, v])
                cols.extend([v, u])

        for i in range(n_nodes):
            neighbors.setdefault(i, set())

        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        return neighbors, adj

    def setParentGraph(self, parent: "Graph", parent_to_local: Optional[Dict[int, int]] = None) -> None:
        self.parentGraph = parent
        self.parent_to_local = {} if parent_to_local is None else parent_to_local
        self.local_to_parent = {local: node for node, local in self.parent_to_local.items()}

    def expand_parent_neighborhood(self, local_node: int, k: int) -> bool:
        if self.parentGraph is None or local_node not in self.local_to_parent:
            return False
        parent_node = self.local_to_parent[local_node]
        new_nodes = [node for node in self.parentGraph.k_ego([parent_node], k) if node not in self.parent_to_local]
        if not new_nodes:
            return False
        start = max(self.local_to_parent, default=-1) + 1
        new_mapping = {node: start + offset for offset, node in enumerate(new_nodes)}
        self.parent_to_local.update(new_mapping)
        self.local_to_parent.update({local: node for node, local in new_mapping.items()})
        additions = {
            local: {
                self.parent_to_local[neighbor]
                for neighbor in self.parentGraph.neighbors[node]
                if neighbor in self.parent_to_local
            }
            for node, local in new_mapping.items()
        }
        self.add_nodes_with_neighbors(additions)
        return True

    def outer_boundary(self, community: Union[List[int], Set[int]]) -> Set[int]:
        """Return nodes adjacent to, but outside, the community."""
        C = set(int(x) for x in community)
        boundary: Set[int] = set()
        for u in C:
            for v in self.neighbors.get(u, ()):
                if v not in C:
                    boundary.add(v)
        return boundary

    def k_ego(self, nodes: Union[List[int], Set[int]], k: int = 2) -> Set[int]:
        """Return the local-indexed k-hop neighborhood."""
        if k <= 0:
            return set(int(x) for x in nodes)
        frontier = set(int(x) for x in nodes)
        visited = set(frontier)
        for _ in range(k):
            nxt: Set[int] = set()
            for u in frontier:
                nxt.update(self.neighbors.get(u, ()))
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        return visited

    def get_k_layer_subgraph_and_mapping(self, node_list: Union[List[int], Set[int]], k: int = 2) -> Tuple[
        "Graph", Dict[int, int]]:
        """Return the induced k-hop subgraph and parent-to-local mapping."""
        nodes_sorted = sorted(self.k_ego(node_list, k))
        old_to_new = {old: i for i, old in enumerate(nodes_sorted)}
        edges_new: List[Tuple[int, int]] = []
        for old_u in nodes_sorted:
            for old_v in self.neighbors.get(old_u, ()):
                if old_v in old_to_new:
                    u = old_to_new[old_u]
                    v = old_to_new[old_v]
                    if u < v:
                        edges_new.append((u, v))
        edges_arr = np.array(edges_new, dtype=np.int64) if edges_new else np.zeros((0, 2), dtype=np.int64)

        # Preserve isolated nodes in the induced subgraph.
        subgraph = Graph(edges_arr, nodes=range(len(nodes_sorted)))
        subgraph.setParentGraph(self, old_to_new)
        return subgraph, old_to_new

    def sample_expansion_from_community(self, community: Union[List[int], Set[int]]) -> List[int]:
        C = list(set(int(x) for x in community))
        if not C:
            return []
        start = random.choice(C)
        trajectory = [start]
        used = {start}
        cur = start
        while True:
            cand = [v for v in self.neighbors.get(cur, ()) if (v in C and v not in used)]
            if not cand:
                break
            nxt = random.choice(cand)
            trajectory.append(nxt)
            used.add(nxt)
            cur = nxt
            if len(used) >= len(C):
                break
        return trajectory

    def add_nodes_with_neighbors(self, newIDnode_nei: Dict[int, Union[List[int], Set[int]]]) -> None:
        """Add local-indexed nodes and their neighbor sets."""
        for i in range(self.n_nodes):
            self.neighbors.setdefault(i, set())

        for new_node, nbs in newIDnode_nei.items():
            new_node = int(new_node)
            if new_node >= self.n_nodes:
                for i in range(self.n_nodes, new_node + 1):
                    self.neighbors[i] = set()
                self.n_nodes = new_node + 1

            for nb in nbs:
                nb = int(nb)
                if nb >= self.n_nodes:
                    for i in range(self.n_nodes, nb + 1):
                        self.neighbors[i] = set()
                    self.n_nodes = nb + 1

                if nb == new_node:
                    continue
                if nb not in self.neighbors[new_node]:
                    self.neighbors[new_node].add(nb)
                    self.neighbors[nb].add(new_node)

        edges = [(u, v) for u, neighbors in self.neighbors.items() for v in neighbors if u < v]
        edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.neighbors, self.adj_mat = self._init_from_edges(edge_array, self.n_nodes)
