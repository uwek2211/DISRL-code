from typing import Optional, List
import numpy as np
from scipy import sparse as sp

import torch
from torch import nn

from env import CommunityExplorationEnv
from graph import Graph
from policy import ExplorerPolicy
from utils import load_params, overlap_scores


class IncrementalGraphPointerNetwork:
    def __init__(self, graph: Graph, k: int = 3, alpha: float = 0.85):
        self.graph = graph
        self.k = k
        self.alpha = alpha
        self.normlized_adj_mat = self._normalize_adj(graph.adj_mat).astype(np.float32)

    def __call__(self, x: sp.spmatrix):
        init_val = x
        for _ in range(self.k):
            x = self.alpha * (self.normlized_adj_mat @ x) + (1 - self.alpha) * init_val
        return x

    def updateGraph(self, graph: Graph):
        self.graph = graph
        self.normlized_adj_mat = self._normalize_adj(graph.adj_mat).astype(np.float32)

    @staticmethod
    def _normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class CommunityExplorer:
    def __init__(
        self,
        args,
        graph: Graph,
        model: ExplorerPolicy,
        optimizer,
        device: Optional[torch.device] = None,
        max_size: int = 100,
        k: int = 3,
        alpha: float = 0.85,
        gamma: float = 0.99,
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.n_nodes = self.graph.n_nodes
        self.max_size = max_size
        self.conv = IncrementalGraphPointerNetwork(graph, k, alpha)
        self.gamma = gamma
        self.args = args
        self.device = torch.device("cpu") if device is None else device

    def _seed_has_boundary(self, seed) -> bool:
        try:
            return len(self.graph.outer_boundary([seed])) > 0
        except Exception:
            return False

    def generate_community(self, seeds: List[list[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size
        if len(seeds) == 1 and (
            len(seeds[0]) >= max_size or not self.graph.outer_boundary(seeds[0])
        ):
            print("[Inference] No expansion is possible; returning the starting nodes.", flush=True)
            return [list(seeds[0])]
        env = CommunityExplorationEnv(self.graph, seeds, max_size)
        self.model.eval()
        with torch.no_grad():
            episodes, _ = self._sample_trajectories(env, False)
        return episodes

    def pad_rewards(self, rewards, logps):
        padded = np.zeros(tuple(logps.shape), dtype=np.float64)
        for row, values in enumerate(rewards):
            padded[row, : len(values)] = values
        return padded

    def policy_loss(self, seeds: List[int], true_coms):
        valid = [(s, c) for s, c in zip(seeds, true_coms) if self._seed_has_boundary(s)]
        if not valid:
            return None

        seeds, true_coms = map(list, zip(*valid))
        env = CommunityExplorationEnv(self.graph, [[seed] for seed in seeds], self.max_size)
        selected_nodes, logps = self._sample_trajectories(env, True)
        lengths = torch.tensor([len(x) for x in selected_nodes], device=self.device)
        rewards = []

        for community, target in zip(selected_nodes, true_coms):
            deltas, partial = [], [community[0]]
            for node in community[1:]:
                if node == "EOS":
                    continue
                before = round(overlap_scores(partial, target, epsilon=1e-9)[2], 4)
                partial.append(node)
                deltas.append(round(overlap_scores(partial, target, epsilon=1e-9)[2], 4) - before)
            rewards.append([np.sum(deltas[i] * (self.gamma ** np.arange(i, len(deltas)))) for i in range(len(deltas))])

        rewards_t = torch.from_numpy(self.pad_rewards(rewards, logps)).float().to(self.device)
        mask = (
            torch.arange(rewards_t.size(1), device=self.device).expand(len(seeds), -1) < (lengths - 1).unsqueeze(1)
        ).float()
        return -(rewards_t * logps * mask).sum()

    def train_policy(self, seeds: List[int], true_coms):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.policy_loss(seeds, true_coms)
        if loss is None:
            return
        loss.backward()
        self.optimizer.step()

    def policy_gradients(self, seeds, communities, params):
        load_params(self.model, params)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss = self.policy_loss(seeds, communities)
        if loss is None:
            return [torch.zeros_like(param) for param in self.model.parameters()]
        loss.backward()
        gradients = [
            param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
            for param in self.model.parameters()
        ]
        self.model.zero_grad(set_to_none=True)
        return gradients

    def train_from_sets(self, episodes: List[List[int]], max_size: Optional[int] = None):
        max_size = self.max_size if max_size is None else max_size

        episodes = [ep for ep in episodes if ep and self._seed_has_boundary(ep[0])]
        if not episodes:
            return

        self.model.train()
        self.optimizer.zero_grad()
        env = CommunityExplorationEnv(self.graph, [[x[0]] for x in episodes], max_size)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()

        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]

        while not env.done:
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits = self.model(*model_inputs)
            logps = []
            actions = []
            for logits, candidates, i in zip(batch_logits, batch_candidates, valid_index):
                valid_candidates = set(candidates) & (set(episodes[i]) - set(env.trajectories[i]))
                if len(valid_candidates) == 0:
                    action = len(candidates)
                else:
                    sub_idx = [idx for idx, v in enumerate(candidates) if v in valid_candidates]
                    action = sub_idx[logits[sub_idx].argmax().item()]
                actions.append(action)
                logps.append(logits[action])

            new_nodes = [x[i] if i < len(x) else "EOS" for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step(new_nodes, valid_index)
            for i, v1 in zip(valid_index, logps):
                episode_logps[i].append(v1)

        logps = nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_logps], batch_first=True)
        lengths = torch.LongTensor([len(x) for x in env.trajectories]).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device, dtype=torch.int64).expand(bs, -1) < (
            lengths - 1
        ).unsqueeze(1)
        mask = mask.float()
        n = mask.sum()
        policy_loss = -(logps * mask).sum() / n
        policy_loss.backward()
        self.optimizer.step()

    def update_graph_and_features(self, z_nodes, bs, new_nodes):
        if self.graph.expand_parent_neighborhood(new_nodes[0], self.args.k_ego_subG):
            extended_matrix = sp.lil_matrix((self.graph.n_nodes, bs), dtype=np.float32)
            extended_matrix[: self.n_nodes, :] = z_nodes
            z_nodes = extended_matrix.tocsc()
            self.n_nodes = self.graph.n_nodes
            self.conv.updateGraph(self.graph)
        return z_nodes

    def _sample_trajectories(self, env: CommunityExplorationEnv, is_train):
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()

        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        new_nodes = []

        while not env.done:
            z_nodes = z_nodes + self.conv(delta_x_nodes)

            if not is_train and new_nodes:
                z_nodes = self.update_graph_and_features(z_nodes, bs, new_nodes)
                env.updateGraph(self.graph)
                seeds = [env.data[i][0] for i in range(bs)]
                x_seeds = env.make_single_node_encoding(seeds)
                z_seeds = self.conv(x_seeds)

            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits = self.model(*model_inputs)
            actions, logps = self._sample_actions(batch_logits)
            new_nodes = [x[i] if i < len(x) else "EOS" for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step(new_nodes, valid_index)

            for i, v1 in zip(valid_index, logps):
                episode_logps[i].append(v1)

        logps = nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_logps], batch_first=True)
        return env.trajectories, logps

    def _prepare_inputs(
        self,
        valid_index: List[int],
        trajectories: List[List[int]],
        z_nodes: sp.csc_matrix,
        z_seeds: sp.csc_matrix,
    ):
        """Prepare state features for the community explorer."""
        vals_seed = []
        vals_node = []
        vals_abs_diff = []
        indptr = []
        offset = 0
        batch_candidates = []

        z_seeds_dense = z_seeds.T.toarray().astype(np.float32, copy=False)
        z_nodes_dense = z_nodes.T.toarray().astype(np.float32, copy=False)

        for i in valid_index:
            boundary_nodes = self.graph.outer_boundary(trajectories[i])
            candidate_nodes = list(boundary_nodes)
            involved_nodes = candidate_nodes + trajectories[i]
            batch_candidates.append(candidate_nodes)

            seed_row = z_seeds_dense[i]
            node_row = z_nodes_dense[i]

            involved_idx = np.asarray(involved_nodes, dtype=np.int64)
            seed_vals = seed_row[involved_idx]
            node_vals = node_row[involved_idx]

            vals_seed.append(seed_vals.astype(np.float32, copy=False))
            vals_node.append(node_vals.astype(np.float32, copy=False))

            traj_idx = np.asarray(trajectories[i], dtype=np.int64)
            comm_mean = float(node_row[traj_idx].mean())
            abs_diff_vals = np.abs(node_vals - comm_mean).astype(np.float32, copy=False)
            vals_abs_diff.append(abs_diff_vals)

            indptr.append((offset, offset + len(involved_nodes), offset + len(candidate_nodes)))
            offset += len(involved_nodes)

        vals_seed = torch.from_numpy(np.concatenate(vals_seed, axis=0)).to(self.device)
        vals_node = torch.from_numpy(np.concatenate(vals_node, axis=0)).to(self.device)

        vals_abs_diff = torch.from_numpy(np.concatenate(vals_abs_diff, axis=0)).to(self.device)
        return vals_seed, vals_node, vals_abs_diff, np.array(indptr), batch_candidates

    def _sample_actions(self, batch_logits: List) -> (List, List, List):
        actions, logps = [], []
        for logits in batch_logits:
            ps = torch.exp(logits) + 1e-8
            try:
                action = torch.multinomial(ps, 1).item()
            except Exception as e:
                print(f"Caught an exception: {e}")
                print(f"ps: {ps}")
                print(f"logits: {logits}")
                raise
            actions.append(action)
            logps.append(logits[action])
        return np.asarray(actions), logps
