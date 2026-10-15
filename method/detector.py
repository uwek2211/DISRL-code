import time
import random
import os
from typing import Optional, Tuple, List, Dict, Iterable

import numpy as np
from torch import optim

from data import build_snapshot_cache, load_graph_communities
from policy import build_policy
from explorer import CommunityExplorer
from graph import Graph
from retrieval import build_grakel_graph, empty_grakel_graph, induce_community_graph, retrieve_topk
from utils import model_params, resolve_device


EXPAND_MAX_SIZE_PERCENTILE_BY_DATASET = {
    "amazon": 90.0,
    "dblp": 90.0,
    "twitter": 100.0,
    "psd": 100.0,
    "youtube": 100.0,
}


class Detector:
    """Construct supervision and run seed-centered exploration for one snapshot."""

    build_snapshot_cache = staticmethod(build_snapshot_cache)
    loadDataset_static = staticmethod(load_graph_communities)

    def __init__(self, args, seed_raw: int, com_index: int, snapshot: Optional[str] = None):
        cache = build_snapshot_cache(args, args.root, args.dataset, snapshot)
        self._initialize_from_cache(args, seed_raw, com_index, cache, snapshot)

    def _validate_supervision_split(self) -> None:
        leaked_indices = set(self.known_community_indices).intersection(self.excluded_community_indices)
        if leaked_indices:
            raise RuntimeError(
                "Supervision leakage detected: excluded target communities remain "
                f"in the training pool: {sorted(leaked_indices)}"
            )
        if not self.known_community_indices:
            raise RuntimeError(
                "No known communities remain after target-community exclusion. "
                "Increase --train_size or revise the train/evaluation split."
            )

    def _initialize_from_cache(
        self,
        args,
        seed_raw: int,
        com_index: int,
        cache: Dict,
        snapshot: Optional[str],
        excluded_com_indices: Optional[Iterable[int]] = None,
    ) -> None:
        self.args, self.snapshot = args, snapshot
        self.device = resolve_device(getattr(args, "device", "cuda:0"))
        self.graph, self.raw_coms, self.coms = (
            cache["graph"],
            cache["raw_coms"],
            cache["coms_local"],
        )
        self.oldSeed_raw, self.com_index = int(seed_raw), int(com_index)
        self.base_known_community_indices = list(cache["base_known_community_indices"])
        self.excluded_community_indices = (
            {self.com_index} if excluded_com_indices is None else {int(idx) for idx in excluded_com_indices}
        )
        self.known_community_indices = [
            idx for idx in self.base_known_community_indices if idx not in self.excluded_community_indices
        ]
        self._validate_supervision_split()
        self.raw_known_comms = [self.raw_coms[idx] for idx in self.known_community_indices]
        self.known_comms_local = [self.coms[idx] for idx in self.known_community_indices]

        try:
            seed_local = self.graph.to_local(self.oldSeed_raw, allow_local=False)
        except KeyError as exc:
            raise KeyError(f"Seed {self.oldSeed_raw} not found in snapshot={snapshot} graph.") from exc

        active_nodes = {seed_local}
        active_nodes.update(node for community in self.known_comms_local for node in community)
        self.known_seed_subgraph, self.old_to_new_node_mapping = self.graph.get_k_layer_subgraph_and_mapping(
            active_nodes, args.k_ego_subG
        )
        self.new_to_old_node_mapping = self.known_seed_subgraph.local_to_parent
        self.known_comms_subgraph = [
            [self.old_to_new_node_mapping[node] for node in community] for community in self.known_comms_local
        ]
        args.max_size = (
            max(map(len, self.known_comms_subgraph)) if self.known_comms_subgraph else getattr(args, "max_size", 100)
        )
        self.relevant_comms = self.known_comms_subgraph
        self.dynamic_expand_max_size = self._compute_dynamic_expand_max_size(self.relevant_comms)
        self.seed = self.old_to_new_node_mapping[seed_local]
        self.explorer = self.init_explorer()

    @classmethod
    def from_snapshot_cache(
        cls,
        args,
        seed_raw: int,
        com_index: int,
        cache: Dict,
        snapshot: Optional[str] = None,
        excluded_com_indices: Optional[Iterable[int]] = None,
    ):
        self = cls.__new__(cls)
        self._initialize_from_cache(args, seed_raw, com_index, cache, snapshot, excluded_com_indices)
        return self

    def _dataset_key_candidates(self) -> List[str]:
        dataset = str(getattr(self.args, "dataset", "") or "")
        dataset_root = str(getattr(self.args, "dataset_root", "") or "")
        candidates = []
        suffixes = ("_expandshrink", "_expand_shrink", "_mergesplit", "_merge_split")

        for raw_name in (dataset, os.path.basename(dataset_root.rstrip("/\\"))):
            name = raw_name.strip().lower().replace("-", "_")
            if not name:
                continue
            candidates.extend([name, name.split("_", 1)[0]])
            candidates.extend(name[: -len(s)] for s in suffixes if name.endswith(s))
        return list(dict.fromkeys(candidates))

    def _get_expand_max_size_percentile(self) -> float:
        return next(
            (EXPAND_MAX_SIZE_PERCENTILE_BY_DATASET[key] for key in self._dataset_key_candidates()
             if key in EXPAND_MAX_SIZE_PERCENTILE_BY_DATASET),
            100.0,
        )

    def _compute_dynamic_expand_max_size(self, communities: Optional[List[List[int]]]) -> int:
        fallback = int(getattr(self.args, "max_size", 100))
        sizes = [len(com) for com in communities if com is not None] if communities else []
        if not sizes:
            return max(1, fallback)
        upper = int(np.ceil(np.percentile(
            np.asarray(sizes, dtype=np.float32), self._get_expand_max_size_percentile()
        )))
        return max(1, upper)

    def _refresh_dynamic_expand_max_size(self) -> None:
        self.dynamic_expand_max_size = self._compute_dynamic_expand_max_size(self.relevant_comms)

    def init_explorer(self):
        model = build_policy(self.args.hidden_size, self.device)
        return CommunityExplorer(
            self.args,
            self.known_seed_subgraph,
            model,
            optim.Adam(model.parameters(), lr=self.args.g_lr),
            self.device,
            max_size=self.args.max_size,
            k=int(getattr(self.args, "L", 3)),
        )

    def select_lists(self, matrix, n):
        indices = list(range(len(matrix)))
        while len(indices) < n:
            indices += indices
        return [matrix[i] for i in np.random.choice(indices, size=n, replace=True)]

    def sample_training_batch(self):
        communities = self.select_lists(self.relevant_comms, self.args.g_batch_size)
        return [random.choice(community) for community in communities], communities

    def train_explorer(self):
        seeds, communities = self.sample_training_batch()
        self.explorer.train_policy(seeds, communities)

        tf_coms = random.choices(self.relevant_comms, k=self.args.g_batch_size)
        episodes = [self.known_seed_subgraph.sample_expansion_from_community(x) for x in tf_coms]
        self.explorer.train_from_sets(episodes)

    def train_epochs(self, epochs=None) -> None:
        for _ in range(self.args.epochs if epochs is None else int(epochs)):
            self.train_explorer()

    def detect(
        self,
        previous_community=None,
        previous_graph=None,
        direct_supervision=False,
        return_intermediate=True,
    ):
        if bool(getattr(self.args, "ablate_rcs", False)):
            self.retrieve_relevant_from_seed_neighborhood()
            return self._train_and_predict("seed-neighborhood reference + exploration", return_intermediate)
        if direct_supervision:
            return self.detect_direct_known_supervision(return_intermediate)
        if previous_community is not None and previous_graph is not None:
            return self.detect_warm_start(previous_community, previous_graph, return_intermediate)
        return self.detect_cold_start(return_intermediate)

    def transition_gradient(
        self,
        params,
        previous_community=None,
        previous_graph=None,
        direct_supervision=False,
    ):
        if bool(getattr(self.args, "ablate_rcs", False)):
            self.retrieve_relevant_from_seed_neighborhood()
        elif direct_supervision:
            self.set_direct_known_supervision()
        elif previous_community is not None and previous_graph is not None:
            try:
                self.retrieve_relevant_from_history(previous_community, previous_graph)
            except Exception as exc:
                print(f"[WARN] window-step relevant retrieval failed; using default supervision. err={exc}")
        seeds, communities = self.sample_training_batch()
        return self.explorer.policy_gradients(seeds, communities, params)

    def _generate_prediction(self) -> Tuple[List[int], List[int]]:
        prediction = self.explorer.generate_community([[self.seed]], max_size=self.dynamic_expand_max_size)[0]
        if prediction and prediction[-1] == "EOS":
            prediction = prediction[:-1]
        pred_local = [self.new_to_old_node_mapping[node] for node in prediction]
        return prediction, self.graph.to_raw_nodes(pred_local)

    def _result(self, pred_raw, return_intermediate, params_before=None, params_after=None):
        result = [self.oldSeed_raw, self.com_index, pred_raw]
        return (result, params_before, params_after) if return_intermediate else result

    def _train_and_predict(self, label: str, return_intermediate: bool):
        params_before = model_params(self.explorer.model) if return_intermediate else None
        tic = time.time()
        self.train_epochs()
        params_after = model_params(self.explorer.model) if return_intermediate else None
        _, pred_raw = self._generate_prediction()
        elapsed = time.time() - tic
        print(f"Elapsed Time ({label}): {elapsed // 60} min {elapsed % 60}s")
        return self._result(pred_raw, return_intermediate, params_before, params_after)

    def detect_cold_start(self, return_intermediate: bool = False):
        """Train and predict without a historical reference community."""
        tic = time.time()
        pretrained = bool(getattr(self.args, "use_pretrained_init_policy", False))
        if pretrained:
            params_before = model_params(self.explorer.model) if return_intermediate else None
            print("=" * 50)
            print("DISRL cold-start pretrained reference generation")
            reference, _ = self._generate_prediction()
            self.retrieve_relevant_from_reference(reference)
            self.train_epochs()
            print("=" * 50)
            print("DISRL cold-start seed-specific refinement")
        else:
            self.train_epochs()
            params_before = model_params(self.explorer.model) if return_intermediate else None
            print("=" * 50)
            print("DISRL cold-start iteration 0")
            reference, _ = self._generate_prediction()
            self.retrieve_relevant_from_reference(reference)
            self.train_epochs()
            print("=" * 50)
            print("DISRL cold-start iteration 1")

        _, pred_raw = self._generate_prediction()
        params_after = model_params(self.explorer.model) if return_intermediate else None
        elapsed = time.time() - tic
        print(f"Elapsed Time: {elapsed // 60} min {elapsed % 60}s")
        return self._result(pred_raw, return_intermediate, params_before, params_after)

    def detect_warm_start(
        self,
        prev_comm_raw: List[int],
        prev_graph: "Graph",
        return_intermediate: bool = False,
    ):
        """Train and predict using the previous snapshot as reference."""
        self.retrieve_relevant_from_history(prev_comm_raw=prev_comm_raw, prev_graph=prev_graph)
        return self._train_and_predict("warm-start supervision + exploration", return_intermediate)

    def detect_direct_known_supervision(self, return_intermediate: bool = False):
        """Train the supervision ablation directly on known communities."""
        self.set_direct_known_supervision()
        return self._train_and_predict("direct known supervision", return_intermediate)

    def set_direct_known_supervision(self) -> None:
        pool = [com for com in self.known_comms_subgraph if com]
        sample_size = int(getattr(self.args, "ablation_supervision_sample_size", getattr(self.args, "k", 30)))
        if not pool:
            self.relevant_comms = self.known_comms_subgraph
        elif sample_size <= 0 or sample_size >= len(pool):
            self.relevant_comms = list(pool)
        else:
            self.relevant_comms = random.sample(pool, sample_size)

        print(
            f"[Supervision: w/o sim] random known-community selection | "
            f"requested={sample_size} | selected={len(self.relevant_comms)}/{len(pool)}",
            flush=True,
        )
        self._refresh_dynamic_expand_max_size()

    def _apply_retrieval(self, selected) -> None:
        if selected is None:
            return
        if selected:
            self.relevant_comms = selected
        self._refresh_dynamic_expand_max_size()

    def retrieve_relevant_from_seed_neighborhood(self) -> None:
        """Use the seed's closed k-hop neighborhood in the current full graph."""
        radius = int(getattr(self.args, "ref_ego_k", 3))
        if radius < 0:
            raise ValueError("ref_ego_k must be nonnegative")
        seed_local = self.graph.to_local(self.oldSeed_raw, allow_local=False)
        reference_nodes = sorted(self.graph.k_ego([seed_local], radius))
        reference_raw = self.graph.to_raw_nodes(reference_nodes)
        reference_graph = induce_community_graph(self.graph, reference_raw, seed_raw=self.oldSeed_raw)
        candidate_graphs = []
        for community in self.raw_known_comms:
            try:
                candidate_graphs.append(induce_community_graph(self.graph, community))
            except Exception:
                candidate_graphs.append(empty_grakel_graph())
        print(f"[Reference] seed neighborhood | k={radius} | nodes={len(reference_nodes)}")
        self._apply_retrieval(retrieve_topk(
            reference_graph, candidate_graphs, self.known_comms_subgraph,
            self.args.k, "seed-neighborhood reference retrieval",
        ))

    def retrieve_relevant_from_history(
            self,
            prev_comm_raw: List[int],
            prev_graph: Optional["Graph"] = None,
    ):
        """Construct the warm-start reference community on the current snapshot.

        Normally project C_pred^{t-1}(s) onto G^t and retain the
        seed-connected component. The w/o cc ablation uses the previous
        result on its original graph without this synthesis step.
        """
        if prev_comm_raw is None:
            print(
                "[WARN] retrieve_relevant_from_history called with "
                "None prev_comm; skip retrieval."
            )
            return

        # A valid seed-centered prediction should always contain the seed.
        if self.oldSeed_raw not in {int(node) for node in prev_comm_raw}:
            print(
                f"[WARN] previous prediction does not contain seed "
                f"{self.oldSeed_raw}; skip warm-start retrieval."
            )
            return

        without_cc = bool(getattr(self.args, "ablate_cc", False))
        label = "previous-result warm-start retrieval" if without_cc else "current-snapshot warm-start retrieval"
        try:
            if without_cc:
                if prev_graph is None:
                    raise ValueError("w/o cc requires the previous snapshot graph")
                g_ref = induce_community_graph(prev_graph, prev_comm_raw)
            else:
                # C_ref^t(s) = CC(G^t[C_pred^{t-1}(s) ∩ V^t]).
                g_ref = induce_community_graph(
                    self.graph,
                    prev_comm_raw,
                    seed_raw=self.oldSeed_raw,
                )
        except Exception as exc:
            print(
                f"[WARN] failed to construct the {label} reference graph: "
                f"{exc}; skip relevant-community retrieval."
            )
            return

        # Known-community candidates are always induced from G^t.
        candidate_graphs = []
        for community in self.raw_known_comms:
            try:
                candidate_graphs.append(
                    induce_community_graph(self.graph, community)
                )
            except Exception:
                candidate_graphs.append(empty_grakel_graph())

        selected = retrieve_topk(
            g_ref,
            candidate_graphs,
            self.known_comms_subgraph,
            self.args.k,
            label,
        )
        self._apply_retrieval(selected)

    def retrieve_relevant_from_reference(self, com):
        reference_graph = build_grakel_graph(self.known_seed_subgraph, com)
        candidate_graphs = [
            build_grakel_graph(self.known_seed_subgraph, community)
            for community in self.known_comms_subgraph
        ]
        self._apply_retrieval(retrieve_topk(
            reference_graph, candidate_graphs, self.known_comms_subgraph,
            self.args.k, "relevant community retrieval"
        ))
