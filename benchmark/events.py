"""Stateful merge and split events for dynamic benchmark generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from community_utils import (
    build_cid2nodes as _build_cid2nodes_from_labels,
    copy_set_map as _set_map,
    discard_label as _discard_label,
    int_set as _ints,
    merge_set_maps as _merge_set_maps,
)


@dataclass
class BaseEvent:
    start: int
    duration: int
    _done: bool = field(init=False, default=False)

    def active(self, t):
        return self.start <= t < self.start + self.duration and not self._done

    def on_start(self, sim, graph, t):
        pass

    def step(self, sim, graph, t):
        raise NotImplementedError


@dataclass
class ParallelPairwiseMergeEvent(BaseEvent):
    """Merge legal community pairs with the most original cross-community edges."""

    m_frac: float = 0.05
    min_comm_size: int = 2
    tail_merge_frac: float = 1.0
    bridge_frac: float = 0.01
    bridge_cap: int = 20000

    _tail_merge_blocked_cids: set[int] = field(init=False, default_factory=set)

    def on_start(self, sim, graph, t):
        nodes_in_graph = {str(node) for node in graph.nodes()}
        cid2nodes = _build_cid2nodes_from_labels(sim.multi_labels, nodes_in_graph)
        if not cid2nodes:
            self._done = True
            return

        sim.merge_split_last_selected = set()

    def _add_bridge_edges(self, graph, left, right):
        added_edges = set()
        if not left or not right:
            return added_edges

        target = int(math.ceil(self.bridge_frac * len(left) * len(right)))
        target = min(target, self.bridge_cap)
        if target <= 0:
            return added_edges

        tries = 0
        added = 0
        max_tries = target * 20

        def norm_edge(u, v):
            u = str(u)
            v = str(v)
            return (u, v) if u <= v else (v, u)

        while added < target and tries < max_tries:
            u = random.choice(left)
            v = random.choice(right)
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
                added_edges.add(norm_edge(u, v))
                added += 1
            tries += 1
        return added_edges

    def _merge_into_new_cid(
        self, sim, nodes_a, nodes_b, c1: int, c2: int, new_cid: int
    ):
        old_set = {int(c1), int(c2)}
        for node in set(list(nodes_a) + list(nodes_b)):
            node = str(node)
            original = set(int(cid) for cid in sim.multi_labels.get(node, set()))
            sim.multi_labels[node] = (original - old_set) | {int(new_cid)}

    def _choose_parent_cid(
        self, cid_a: int, cid_b: int, nodes_a: set[str], nodes_b: set[str]
    ) -> int:
        cid_a = int(cid_a)
        cid_b = int(cid_b)
        size_a, size_b = len(nodes_a), len(nodes_b)
        if size_a == size_b:
            return min(cid_a, cid_b)
        return cid_a if size_a > size_b else cid_b

    def _is_legal_pair(self, sim, roots_a: set[int], roots_b: set[int]) -> bool:
        if not roots_a or not roots_b or roots_a & roots_b:
            return False
        seed_roots = _ints(getattr(sim, "seed2cid", {}).values())
        train_roots = _ints(getattr(sim, "tail_root_cids", set()))
        if (roots_a & seed_roots) and (roots_b & seed_roots):
            return False
        return not (
            ((roots_a & train_roots) and (roots_b & seed_roots))
            or ((roots_b & train_roots) and (roots_a & seed_roots))
        )

    def _state_for_active_cid(self, sim, active_states, cid: int, nodes: set[str]):
        cid = int(cid)
        roots = _ints(sim.get_roots_for_cid(cid))
        state = active_states.get(cid)
        if state is not None:
            root_nodes = _set_map(state.get("root_nodes", {}), str)
            bridge_edges_by_root = _set_map(
                state.get("bridge_edges_by_root", {}), tuple
            )
        else:
            root_nodes = {root: {str(node) for node in nodes} for root in roots}
            bridge_edges_by_root = {}

        for root in roots:
            root_nodes.setdefault(int(root), set())
            if int(root) in getattr(sim, "tail_anchor_nodes", {}):
                root_nodes[int(root)] |= set(sim.tail_anchor_nodes[int(root)])
            bridge_edges_by_root.setdefault(int(root), set())

        return {
            "parent_cid": cid,
            "remaining_roots": roots,
            "root_nodes": root_nodes,
            "bridge_edges_by_root": bridge_edges_by_root,
        }

    def _candidate_pairs(self, sim, cid2nodes_all):
        cids = [
            int(cid)
            for cid, nodes in cid2nodes_all.items()
            if len(nodes) >= int(self.min_comm_size)
        ]
        if len(cids) <= 1:
            return []
        root_cache = {cid: _ints(sim.get_roots_for_cid(cid)) for cid in cids}
        candidates = []
        for i, cid_a in enumerate(cids):
            if self._tail_cid_blocked(cid_a):
                continue
            roots_a = root_cache[cid_a]
            for cid_b in cids[i + 1 :]:
                if self._tail_cid_blocked(cid_b):
                    continue
                roots_b = root_cache[cid_b]
                if not self._is_legal_pair(sim, roots_a, roots_b):
                    continue
                cross_edges = int(sim.base_cross_edge_count(roots_a, roots_b))
                if cross_edges < 1:
                    continue
                balance = -abs(len(cid2nodes_all[cid_a]) - len(cid2nodes_all[cid_b]))
                candidates.append((cross_edges, balance, random.random(), cid_a, cid_b))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return candidates

    def _size_fallback_pairs(self, sim, cid2nodes_all, used_cids, max_pairs):
        """Pair remaining legal communities by largest-smallest size order."""
        if max_pairs <= 0:
            return []

        available = {
            int(cid)
            for cid, nodes in cid2nodes_all.items()
            if len(nodes) >= int(self.min_comm_size)
            and int(cid) not in used_cids
            and not self._tail_cid_blocked(int(cid))
        }
        pairs = []
        while len(pairs) < max_pairs and len(available) >= 2:
            largest = max(available, key=lambda cid: (len(cid2nodes_all[cid]), cid))
            smallest_candidates = sorted(
                [cid for cid in available if cid != largest],
                key=lambda cid: (len(cid2nodes_all[cid]), cid),
            )

            roots_largest = _ints(sim.get_roots_for_cid(largest))
            matched = next(
                (
                    smallest
                    for smallest in smallest_candidates
                    if self._is_legal_pair(
                        sim, roots_largest, _ints(sim.get_roots_for_cid(smallest))
                    )
                ),
                None,
            )
            if matched is None:
                available.remove(largest)
                continue
            pairs.append((largest, matched))
            available.difference_update((largest, matched))
        return pairs

    def _forced_tail_active_cids(self, sim, cid2nodes_all):
        tail_roots = _ints(getattr(sim, "tail_root_cids", set()))
        if not tail_roots:
            return []

        forced = [
            int(cid)
            for cid, nodes in cid2nodes_all.items()
            if len(nodes) >= int(self.min_comm_size)
            and _ints(sim.get_roots_for_cid(cid)) & tail_roots
        ]
        forced.sort(
            key=lambda cid: (-len(_ints(sim.get_roots_for_cid(cid)) & tail_roots), cid)
        )
        frac = max(0.0, min(1.0, float(self.tail_merge_frac)))
        if frac <= 0.0:
            return []
        if frac < 1.0 and forced:
            k = max(1, int(math.ceil(frac * len(forced))))
            sampled = set(random.sample(forced, min(k, len(forced))))
            forced = [cid for cid in forced if cid in sampled]
        return forced

    def _tail_cid_blocked(self, cid: int) -> bool:
        return int(cid) in self._tail_merge_blocked_cids

    def _refresh_tail_merge_filter(self, sim, cid2nodes_all, forced_tail_cids) -> None:
        allowed_cids = _ints(forced_tail_cids)
        self._tail_merge_blocked_cids = set()
        frac = max(0.0, min(1.0, float(self.tail_merge_frac)))
        if frac >= 1.0:
            return
        tail_roots = _ints(getattr(sim, "tail_root_cids", set()))
        if not tail_roots:
            return
        self._tail_merge_blocked_cids = {
            int(cid)
            for cid, nodes in cid2nodes_all.items()
            if len(nodes) >= int(self.min_comm_size)
            and _ints(sim.get_roots_for_cid(cid)) & tail_roots
            and int(cid) not in allowed_cids
        }

    def _best_partner_for_cid(self, sim, cid2nodes_all, cid, used_cids):
        cid = int(cid)
        roots = _ints(sim.get_roots_for_cid(cid))
        partners = []
        for other, nodes in cid2nodes_all.items():
            other = int(other)
            if (
                other == cid
                or other in used_cids
                or self._tail_cid_blocked(other)
                or len(nodes) < int(self.min_comm_size)
            ):
                continue
            other_roots = _ints(sim.get_roots_for_cid(other))
            if not self._is_legal_pair(sim, roots, other_roots):
                continue
            cross_edges = int(sim.base_cross_edge_count(roots, other_roots))
            size_gap = abs(len(cid2nodes_all[cid]) - len(cid2nodes_all[other]))
            partners.append((cross_edges, size_gap, random.random(), other))

        if not partners:
            return None

        cross_partners = [item for item in partners if item[0] >= 1]
        if cross_partners:
            cross_partners.sort(key=lambda item: (-item[0], -item[1], item[2]))
            return cid, cross_partners[0][3]

        partners.sort(key=lambda item: (-item[1], item[2]))
        return cid, partners[0][3]

    def _forced_tail_pairs(self, sim, cid2nodes_all, used_cids, forced_cids):
        pairs = []
        for cid in forced_cids:
            if cid in used_cids:
                continue
            pair = self._best_partner_for_cid(sim, cid2nodes_all, cid, used_cids)
            if pair is None:
                continue
            pairs.append(pair)
            used_cids.update(pair)
        return pairs

    def step(self, sim, graph, t):
        if not self.active(t):
            return

        nodes_in_graph = {str(node) for node in graph.nodes()}
        cid2nodes_all = _build_cid2nodes_from_labels(sim.multi_labels, nodes_in_graph)

        if not cid2nodes_all:
            self._done = True
            return

        frac = max(0.0, min(1.0, float(self.m_frac)))
        active_count = sum(
            1
            for nodes in cid2nodes_all.values()
            if len(nodes) >= int(self.min_comm_size)
        )
        max_pairs = max(2, int(math.floor(frac * active_count))) // 2
        if max_pairs <= 0:
            self._done = True
            return

        forced_tail_cids = self._forced_tail_active_cids(sim, cid2nodes_all)
        self._refresh_tail_merge_filter(sim, cid2nodes_all, forced_tail_cids)

        candidates = self._candidate_pairs(sim, cid2nodes_all)
        used_cids = set()
        selected_pairs = self._forced_tail_pairs(
            sim, cid2nodes_all, used_cids, forced_tail_cids
        )

        for _cross_edges, _balance, _tie, cid_a, cid_b in candidates:
            if len(selected_pairs) >= max_pairs:
                break
            if cid_a in used_cids or cid_b in used_cids:
                continue
            selected_pairs.append((cid_a, cid_b))
            used_cids.update((cid_a, cid_b))

        if len(selected_pairs) < max_pairs:
            fallback_pairs = self._size_fallback_pairs(
                sim,
                cid2nodes_all,
                used_cids,
                max_pairs - len(selected_pairs),
            )
            selected_pairs.extend(fallback_pairs)
            used_cids.update(cid for pair in fallback_pairs for cid in pair)

        sim.merge_split_last_selected = set(used_cids)

        if not selected_pairs:
            self._done = True
            return

        active_states = sim.merge_split_active_states
        for cid_a, cid_b in selected_pairs:
            cid_a, cid_b = int(cid_a), int(cid_b)
            nodes_a = {str(node) for node in cid2nodes_all.get(cid_a, set())}
            nodes_b = {str(node) for node in cid2nodes_all.get(cid_b, set())}
            state_a = self._state_for_active_cid(sim, active_states, cid_a, nodes_a)
            state_b = self._state_for_active_cid(sim, active_states, cid_b, nodes_b)
            roots_a = set(state_a["remaining_roots"])
            roots_b = set(state_b["remaining_roots"])
            merged_roots = roots_a | roots_b
            parent_cid = self._choose_parent_cid(cid_a, cid_b, nodes_a, nodes_b)
            bridge_edges = self._add_bridge_edges(graph, list(nodes_a), list(nodes_b))
            self._merge_into_new_cid(sim, nodes_a, nodes_b, cid_a, cid_b, parent_cid)
            root_nodes = _merge_set_maps(state_a["root_nodes"], state_b["root_nodes"])
            for root in merged_roots:
                root_nodes.setdefault(root, set())
                if root in getattr(sim, "tail_anchor_nodes", {}):
                    root_nodes[root] |= set(sim.tail_anchor_nodes[root])
            bridge_edges_by_root = _merge_set_maps(
                state_a["bridge_edges_by_root"], state_b["bridge_edges_by_root"]
            )
            for root, nodes in root_nodes.items():
                touched = {
                    edge
                    for edge in bridge_edges
                    if edge[0] in nodes or edge[1] in nodes
                }
                bridge_edges_by_root.setdefault(root, set()).update(touched)
            new_state = {
                "parent_cid": parent_cid,
                "remaining_roots": merged_roots,
                "root_nodes": root_nodes,
                "bridge_edges_by_root": bridge_edges_by_root,
                "created_at": int(t),
            }
            for cid in (cid_a, cid_b):
                active_states.pop(cid, None)
                if cid != parent_cid:
                    sim.drop_roots_for_cid(cid)
            active_states[parent_cid] = new_state
            sim.set_roots_for_cid(parent_cid, merged_roots)

        sim.merge_split_active_states = active_states


@dataclass
class RandomRecordedSplitEvent(BaseEvent):
    """Randomly release root communities from recorded merge states."""

    def _remaining_release_ops(self, active_states):
        return sum(
            max(0, len(state.get("remaining_roots", set())) - 1)
            for state in active_states.values()
        )

    def _split_count_for_step(self, active_states, t):
        remaining_ops = self._remaining_release_ops(active_states)
        if remaining_ops <= 0:
            return 0

        remaining_steps = max(1, (self.start + self.duration) - int(t))
        return max(1, int(math.ceil(remaining_ops / remaining_steps)))

    def _choose_root_from_state(self, state):
        roots = list(state.get("remaining_roots", set()))
        return random.choice(roots) if roots else None

    def _remove_edges(self, graph, edges):
        to_remove = [(u, v) for u, v in edges if graph.has_edge(u, v)]
        if to_remove:
            graph.remove_edges_from(to_remove)

    def _ensure_nodes_exist(self, sim, graph, nodes, cid):
        for node in nodes:
            node = str(node)
            if node not in graph:
                graph.add_node(node)
            sim.multi_labels.setdefault(node, set())
            sim.orig_comm.setdefault(node, int(cid))

    def _release_root(self, sim, graph, active_states, state, root):
        parent_cid = int(state["parent_cid"])
        root = int(root)
        remaining_before = _ints(state.get("remaining_roots", set()))
        if root not in remaining_before:
            return False
        root_nodes = _set_map(state.get("root_nodes", {}), str)
        if root in getattr(sim, "tail_anchor_nodes", {}):
            root_nodes.setdefault(root, set()).update(sim.tail_anchor_nodes[root])
        release_nodes = set(root_nodes.get(root, set()))
        remaining_after = remaining_before - {root}
        remaining_nodes = set().union(
            *(root_nodes.get(other_root, set()) for other_root in remaining_after)
        )

        self._ensure_nodes_exist(sim, graph, release_nodes, root)

        for node in release_nodes:
            labels = sim.multi_labels.setdefault(str(node), set())
            labels.add(root)
            if str(node) not in remaining_nodes:
                labels.discard(parent_cid)
            sim.multi_labels[str(node)] = labels

        bridge_edges_by_root = _set_map(state.get("bridge_edges_by_root", {}), tuple)
        released_edges = set(bridge_edges_by_root.get(root, set()))
        self._remove_edges(graph, released_edges)
        for other_root in list(bridge_edges_by_root.keys()):
            bridge_edges_by_root[other_root] -= released_edges

        sim.set_roots_for_cid(root, {root})

        if not remaining_after:
            active_states.pop(parent_cid, None)
            sim.drop_roots_for_cid(parent_cid)
            return True

        if root == parent_cid:
            old_parent_cid = parent_cid
            new_parent_cid = int(random.choice(list(remaining_after)))
            for node in remaining_nodes:
                labels = sim.multi_labels.setdefault(node, set())
                labels.discard(parent_cid)
                labels.add(new_parent_cid)
                sim.multi_labels[node] = labels
            for node in release_nodes:
                labels = sim.multi_labels.setdefault(node, set())
                labels.add(root)
                if node in remaining_nodes:
                    labels.add(new_parent_cid)
                sim.multi_labels[node] = labels
            _discard_label(
                sim.multi_labels, parent_cid, release_nodes | remaining_nodes
            )
            state["parent_cid"] = new_parent_cid
            parent_cid = new_parent_cid
            active_states.pop(old_parent_cid, None)
            active_states[parent_cid] = state

        if len(remaining_after) == 1:
            last_root = int(next(iter(remaining_after)))
            if last_root in getattr(sim, "tail_anchor_nodes", {}):
                root_nodes.setdefault(last_root, set()).update(
                    sim.tail_anchor_nodes[last_root]
                )
            last_nodes = set(root_nodes.get(last_root, set()))
            self._ensure_nodes_exist(sim, graph, last_nodes, last_root)
            for node in last_nodes:
                labels = sim.multi_labels.setdefault(node, set())
                labels.add(last_root)
                if last_root != parent_cid:
                    labels.discard(parent_cid)
                sim.multi_labels[node] = labels
            _discard_label(
                sim.multi_labels,
                parent_cid,
                last_nodes if last_root == parent_cid else set(),
            )
            self._remove_edges(graph, set(bridge_edges_by_root.get(last_root, set())))
            sim.set_roots_for_cid(last_root, {last_root})
            active_states.pop(parent_cid, None)
            if parent_cid != last_root:
                sim.drop_roots_for_cid(parent_cid)
            return True

        state["remaining_roots"] = remaining_after
        state["root_nodes"] = {
            r: set(nodes) for r, nodes in root_nodes.items() if r in remaining_after
        }
        state["bridge_edges_by_root"] = {
            r: set(edges)
            for r, edges in bridge_edges_by_root.items()
            if r in remaining_after
        }
        active_states[parent_cid] = state
        sim.set_roots_for_cid(parent_cid, remaining_after)
        return True

    def step(self, sim, graph, t):
        if not self.active(t):
            return

        active_states = sim.merge_split_active_states
        if not active_states:
            self._done = True
            return

        split_count = self._split_count_for_step(active_states, t)
        if split_count <= 0:
            self._done = True
            return

        released = 0
        while released < split_count:
            round_states = [
                state
                for state in active_states.values()
                if len(state.get("remaining_roots", set())) >= 2
            ]
            if not round_states:
                break
            random.shuffle(round_states)
            released_before = released
            for state in round_states:
                if released >= split_count:
                    break
                parent_cid = int(state.get("parent_cid", -1))
                current_state = active_states.get(parent_cid)
                if (
                    current_state is None
                    or len(current_state.get("remaining_roots", set())) < 2
                ):
                    continue
                root = self._choose_root_from_state(current_state)
                if root is None:
                    continue
                if self._release_root(
                    sim, graph, active_states, current_state, int(root)
                ):
                    released += 1
            if released == released_before:
                break
        sim.merge_split_active_states = active_states
        if released == 0:
            self._done = True
