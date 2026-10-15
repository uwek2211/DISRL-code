"""Simulation state, seed protection, lineage-aware output, and snapshots."""

import os
from collections import defaultdict

import networkx as nx
from tqdm import tqdm

from community_utils import build_cid2nodes, normalize_labels


def choose_descendant_by_overlap(
    prev_nodes: set, candidates: dict[int, set], min_intersection: int = 1
) -> int:
    """Choose the candidate community with maximum overlap to previous nodes."""
    ranked = []
    for cid, nodes in candidates.items():
        intersection = len(prev_nodes & nodes)
        if intersection >= min_intersection:
            union = len(prev_nodes | nodes)
            ranked.append(
                (intersection, intersection / union if union else 0.0, -cid, cid)
            )
    return max(ranked)[-1] if ranked else -1


def _node_sort_key(node):
    """Stable output order for int-like and string node ids."""
    text = str(node)
    try:
        return (0, int(text))
    except Exception:
        return (1, text)


def _write_lines(path, lines, trailing_newline=True):
    text = "\n".join(str(line) for line in lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + ("\n" if text and trailing_newline else ""))


class GraphMovementSimulation:
    """Maintain graph/community state, seed tracking, and snapshot output."""

    def __init__(
        self,
        graph,
        communities,
        seed2cid=None,
        seeds=None,
        tail_evolve_k=0,
    ):
        self.G_base = graph.copy()
        self.seed2cid = seed2cid or {}
        self.seeds = [str(seed) for seed in (seeds or [])]
        self.seeds_str = set(self.seeds)
        self.multi_labels = normalize_labels(communities)
        self.base_multi_labels = {
            node: set(cids) for node, cids in self.multi_labels.items()
        }
        self.base_cid2nodes = build_cid2nodes(self.base_multi_labels)
        all_base_cids = sorted(self.base_cid2nodes)
        self.tail_evolve_k = max(0, int(tail_evolve_k or 0))
        self.tail_root_cids = (
            set(all_base_cids[-self.tail_evolve_k :]) if self.tail_evolve_k else set()
        )
        self.tail_anchor_nodes = {
            int(cid): set(self.base_cid2nodes[cid]) for cid in self.tail_root_cids
        }
        self.protected_tail_nodes = set().union(*self.tail_anchor_nodes.values())
        self.community_roots = {int(cid): {int(cid)} for cid in all_base_cids}
        self.merge_split_active_states = {}
        self._base_root_pair_edge_counts = None

        primary = {node: min(cids) for node, cids in self.multi_labels.items() if cids}
        default_comm = (max(primary.values()) + 1) if primary else 0

        self.orig_comm = {
            str(node): int(primary.get(str(node), default_comm))
            for node in self.G_base.nodes
        }
        self.noise_labels = dict(self.orig_comm)

        self._next_global_node_id = None
        self._seed_tracked_cid_prev: dict[str, int] = {}
        self._seed_tracked_nodes_prev: dict[str, set] = {}
        self._tracker_initialized = False

    def get_roots_for_cid(self, cid: int) -> set[int]:
        cid = int(cid)
        return {int(root) for root in self.community_roots.get(cid, {cid})}

    def set_roots_for_cid(self, cid: int, roots):
        self.community_roots[int(cid)] = set(int(root) for root in roots)

    def drop_roots_for_cid(self, cid: int):
        self.community_roots.pop(int(cid), None)

    def get_protected_nodes(self, include_seeds=True, include_tail=True) -> set[str]:
        return (set(self.seeds_str) if include_seeds else set()) | (
            set(self.protected_tail_nodes) if include_tail else set()
        )

    def _build_base_root_pair_edge_counts(self):
        pair_counts = defaultdict(int)
        for u, v in self.G_base.edges():
            u = str(u)
            v = str(v)
            labs_u = self.base_multi_labels.get(u, set())
            labs_v = self.base_multi_labels.get(v, set())
            if not labs_u or not labs_v:
                continue
            for cu in labs_u:
                for cv in labs_v:
                    cu = int(cu)
                    cv = int(cv)
                    if cu == cv:
                        continue
                    key = (cu, cv) if cu < cv else (cv, cu)
                    pair_counts[key] += 1
        self._base_root_pair_edge_counts = dict(pair_counts)

    def base_cross_edge_count(self, roots_a, roots_b) -> int:
        if self._base_root_pair_edge_counts is None:
            self._build_base_root_pair_edge_counts()

        roots_a = {int(root) for root in roots_a}
        roots_b = {int(root) for root in roots_b}
        if not roots_a or not roots_b:
            return 0
        return sum(
            int(
                self._base_root_pair_edge_counts.get(
                    (ra, rb) if ra < rb else (rb, ra), 0
                )
            )
            for ra in roots_a
            for rb in roots_b
            if ra != rb
        )

    def _refresh_from_graph(self, graph):
        nodes_in_graph = {str(node) for node in graph.nodes()}
        self.multi_labels = {
            node: cids
            for node, cids in self.multi_labels.items()
            if node in nodes_in_graph
        }
        default_comm = (max(self.orig_comm.values()) + 1) if self.orig_comm else 0
        self.orig_comm = {
            str(node): int(self.orig_comm.get(str(node), default_comm))
            for node in graph.nodes()
        }
        self.noise_labels = dict(self.orig_comm)

    def _prune_singleton_communities(self, graph, protect_seeds=True, delete_node=True):
        nodes_in_graph = {str(node) for node in graph.nodes()}
        if not nodes_in_graph:
            return
        protected_nodes = self.get_protected_nodes(
            include_seeds=protect_seeds, include_tail=True
        )
        cid2nodes = build_cid2nodes(self.multi_labels, nodes_in_graph)
        for cid, nodes in cid2nodes.items():
            if len(nodes) != 1:
                continue
            node = next(iter(nodes))
            if node in protected_nodes:
                continue
            labels = set(self.multi_labels.get(node, set()))
            labels.discard(int(cid))
            if labels:
                self.multi_labels[node] = labels
            else:
                self.multi_labels.pop(node, None)
            if (
                delete_node
                and node in graph
                and not (protect_seeds and node in self.seeds_str)
            ):
                graph.remove_node(node)
                nodes_in_graph.discard(node)
                self.multi_labels.pop(node, None)
                self.orig_comm.pop(node, None)

    def _remove_isolated_nodes(self, graph, protect_seeds=True):
        protected_nodes = self.get_protected_nodes(
            include_seeds=protect_seeds, include_tail=True
        )
        isolated = [
            str(node) for node in nx.isolates(graph) if str(node) not in protected_nodes
        ]
        if not isolated:
            return
        graph.remove_nodes_from(isolated)
        for node in isolated:
            self.multi_labels.pop(node, None)
            self.orig_comm.pop(node, None)
        self._refresh_from_graph(graph)

    def _limit_deletion_in_seed_communities(
        self, graph, nodes_to_remove, seeds, min_size=3
    ):
        nodes_to_remove = [str(node) for node in nodes_to_remove]
        seeds_str = {str(seed) for seed in seeds} if seeds is not None else set()

        if not seeds_str:
            protected_nodes = self.get_protected_nodes(
                include_seeds=False, include_tail=True
            )
            return [node for node in nodes_to_remove if node not in protected_nodes]

        protected_cids = set().union(
            *(self.base_multi_labels.get(seed, set()) for seed in seeds_str)
        )
        protected_cids |= set(self.tail_root_cids)

        if not protected_cids:
            protected_nodes = self.get_protected_nodes(
                include_seeds=True, include_tail=True
            )
            return [node for node in nodes_to_remove if node not in protected_nodes]

        nodes_in_graph = {str(node) for node in graph.nodes()}
        comm2nodes = {
            cid: self.base_cid2nodes.get(cid, set()) & nodes_in_graph
            for cid in protected_cids
        }
        capacity = {
            cid: max(0, len(nodes) - min_size) for cid, nodes in comm2nodes.items()
        }
        can_delete = []
        protected_nodes = self.get_protected_nodes(
            include_seeds=True, include_tail=True
        )
        for node in nodes_to_remove:
            if node in protected_nodes:
                continue

            related = [cid for cid, nodes in comm2nodes.items() if node in nodes]
            if not related:
                can_delete.append(node)
                continue

            if all(capacity.get(cid, 0) > 0 for cid in related):
                can_delete.append(node)
                for cid in related:
                    capacity[cid] -= 1

        return can_delete

    def _init_seed_tracker_if_needed(self, comm2nodes: dict[int, set]):
        if self._tracker_initialized:
            return

        for seed in self.seeds:
            labels = {int(cid) for cid in self.multi_labels.get(seed, set())}
            fixed = int(self.seed2cid[seed]) if seed in self.seed2cid else None
            init_cid = (
                fixed
                if fixed in comm2nodes and comm2nodes[fixed]
                else next(
                    (
                        cid
                        for cid in sorted(labels)
                        if cid in comm2nodes and comm2nodes[cid]
                    ),
                    min(labels) if labels else -1,
                )
            )
            if init_cid in comm2nodes:
                self._seed_tracked_cid_prev[seed] = int(init_cid)
                self._seed_tracked_nodes_prev[seed] = set(comm2nodes[init_cid])
            else:
                self._seed_tracked_cid_prev[seed] = -1
                self._seed_tracked_nodes_prev[seed] = set()

        self._tracker_initialized = True

    def _choose_seed_comm_index_lineage_aware(
        self, seed: str, comm2nodes: dict[int, set], cid2row: dict[int, int]
    ) -> int:
        seed = str(seed)
        labels = {int(cid) for cid in self.multi_labels.get(seed, set())}
        if not labels:
            return -1
        candidates = {
            cid: set(comm2nodes[cid]) for cid in labels if comm2nodes.get(cid)
        }
        if not candidates:
            return -1

        prev_nodes = self._seed_tracked_nodes_prev.get(seed, set())
        prev_cid = self._seed_tracked_cid_prev.get(seed, -1)

        fixed_cid = (
            int(self.seed2cid.get(seed, -1)) if self.seed2cid is not None else -1
        )
        if fixed_cid != -1 and fixed_cid in candidates:
            chosen_cid = fixed_cid
        elif prev_cid in candidates:
            chosen_cid = int(prev_cid)
        elif not prev_nodes or prev_cid == -1:
            chosen_cid = min(candidates.keys())
        else:
            chosen_cid = choose_descendant_by_overlap(
                prev_nodes=prev_nodes,
                candidates=candidates,
                min_intersection=1,
            )
            if chosen_cid == -1:
                return -1

        self._seed_tracked_cid_prev[seed] = int(chosen_cid)
        self._seed_tracked_nodes_prev[seed] = set(comm2nodes[int(chosen_cid)])

        return int(cid2row.get(int(chosen_cid), -1))

    def run(
        self,
        graph,
        timesteps=20,
        scheduler=None,
        out_root=None,
        seeds=None,
        write_seed_files=None,
        pb=True,
    ):
        if scheduler is None:
            raise ValueError("A scheduler is required")

        if seeds is not None:
            self.seeds = [str(seed) for seed in seeds]
            self.seeds_str = set(self.seeds)

        if write_seed_files is None:
            write_seed_files = bool(self.seeds)

        snapshots_root = None
        if out_root is not None:
            snapshots_root = os.path.join(out_root, "snapshots")
            os.makedirs(snapshots_root, exist_ok=True)

        started_events = set()
        iterator = tqdm(range(timesteps), desc="Simulating") if pb else range(timesteps)

        for t in iterator:
            for rule in scheduler.get_active_rules(t):
                rule.step(self, graph, t)

            for event in scheduler.get_active_events(t):
                if id(event) not in started_events:
                    event.on_start(self, graph, t)
                    started_events.add(id(event))
                event.step(self, graph, t)

            for seed in self.seeds:
                if seed not in graph:
                    graph.add_node(seed)

            self._remove_isolated_nodes(graph, protect_seeds=True)
            self._prune_singleton_communities(
                graph, protect_seeds=True, delete_node=True
            )

            if snapshots_root is not None:
                self._write_snapshot(snapshots_root, graph, t, write_seed_files)

        return graph

    def _write_snapshot(self, snapshots_root, graph, t, write_seed_files):
        snap_name = f"t{t:03d}"
        snap_dir = os.path.join(snapshots_root, snap_name)
        os.makedirs(snap_dir, exist_ok=True)

        edges = [(str(u), str(v)) for u, v in graph.edges()]
        edges.sort(key=lambda edge: (_node_sort_key(edge[0]), _node_sort_key(edge[1])))
        _write_lines(
            os.path.join(snap_dir, f"{snap_name}_ungraph.txt"),
            (f"{u} {v}" for u, v in edges),
        )

        nodes_in_graph = {str(node) for node in graph.nodes()}
        comm2nodes_list = build_cid2nodes(
            self.multi_labels, nodes_in_graph, ordered=True
        )

        items = sorted(comm2nodes_list.items(), key=lambda item: item[0])
        nonempty_items = [(cid, nodes) for cid, nodes in items if nodes]

        _write_lines(
            os.path.join(snap_dir, f"{snap_name}_cmty.txt"),
            (" ".join(nodes) for _cid, nodes in nonempty_items),
        )
        _write_lines(
            os.path.join(snap_dir, f"{snap_name}_row2cid.txt"),
            (cid for cid, _nodes in nonempty_items),
        )

        if not write_seed_files:
            return

        cid2row = {cid: row for row, (cid, _) in enumerate(nonempty_items)}
        comm2nodes = {cid: set(nodes) for cid, nodes in nonempty_items}

        self._init_seed_tracker_if_needed(comm2nodes)

        _write_lines(
            os.path.join(snap_dir, f"{snap_name}_seed.txt"),
            [" ".join(self.seeds)],
            trailing_newline=False,
        )
        indices = [
            str(self._choose_seed_comm_index_lineage_aware(seed, comm2nodes, cid2row))
            for seed in self.seeds
        ]
        _write_lines(
            os.path.join(snap_dir, f"{snap_name}_seed_cmty_index.txt"),
            [" ".join(indices)],
            trailing_newline=False,
        )
