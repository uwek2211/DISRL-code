from typing import Union, Optional, List
import numpy as np
from scipy import sparse as sp

from graph import Graph


class CommunityExplorationEnv:

    def __init__(self, graph: Graph, selected_nodes: List[List[int]], max_size: int):
        self.max_size = max_size
        self.graph = graph
        self.n_nodes = self.graph.n_nodes

        self.data = [self._normalize_node_list(x) for x in selected_nodes]

        self.bs = len(self.data)
        self.trajectories = None
        self.dones = None

    def _to_local_if_needed(self, v):
        """Map a raw node ID to its local index when a mapping is available."""
        if v is None or v == 'EOS':
            return v
        try:
            node_id = int(v)
        except Exception:
            return v
        mapped = self.graph.find_local(node_id, allow_local=False)
        return node_id if mapped is None else mapped

    def _normalize_node_list(self, xs: List[Union[int, str]]):
        return [self._to_local_if_needed(v) for v in xs]

    @property
    def done(self):
        return all(self.dones)

    @property
    def valid_index(self) -> List[int]:
        return [i for i, d in enumerate(self.dones) if not d]

    def updateGraph(self, graph):
        self.graph = graph
        self.n_nodes = self.graph.n_nodes

    def reset(self):
        self.trajectories = [x.copy() for x in self.data]
        self.dones = [
            x[-1] == 'EOS' or len(x) >= self.max_size or len(self.graph.outer_boundary(x)) == 0
            for x in self.trajectories
        ]

        assert not any(self.dones)

        seeds = [self.data[i][0] for i in range(self.bs)]
        nodes = [self.data[i] for i in range(self.bs)]

        x_seeds = self.make_single_node_encoding(seeds)
        x_nodes = self.make_nodes_encoding(nodes)
        return x_seeds, x_nodes

    def step(self, new_nodes: List[Union[int, str]], index: List[int]):
        assert len(new_nodes) == len(index)

        new_nodes = [self._to_local_if_needed(v) for v in new_nodes]

        full_new_nodes: List[Optional[int]] = [None for _ in range(self.bs)]
        for i, v in zip(index, new_nodes):
            self.trajectories[i].append(v)
            if v == 'EOS':
                self.dones[i] = True
            elif len(self.trajectories[i]) == self.max_size:
                self.dones[i] = True
            elif len(self.graph.outer_boundary(self.trajectories[i])) == 0:
                self.dones[i] = True
            else:
                full_new_nodes[i] = v

        delta_x_nodes = self.make_single_node_encoding(full_new_nodes)
        return delta_x_nodes

    def make_single_node_encoding(self, nodes):
        return self._make_encoding([[node] for node in nodes])

    def make_nodes_encoding(self, nodes: List[List[int]]):
        assert len(nodes) == self.bs
        return self._make_encoding(nodes)

    def _make_encoding(self, columns):
        pairs = []
        for column, values in enumerate(columns):
            for value in values:
                if value is None or value == 'EOS':
                    continue
                try:
                    node_id = int(value)
                except Exception:
                    continue
                if 0 <= node_id < self.n_nodes:
                    pairs.append((node_id, column))
        if not pairs:
            return sp.csc_matrix((self.n_nodes, len(columns)), dtype=np.float32)
        indices = np.asarray(pairs, dtype=np.int64).T
        return sp.csc_matrix(
            (np.ones(indices.shape[1], dtype=np.float32), indices),
            shape=(self.n_nodes, len(columns)),
        )
