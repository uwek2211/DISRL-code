import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def make_linear_block(in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
    return LinearBlock(in_size, out_size, act_cls, norm_type, bias, residual, dropout)


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
        super().__init__()
        self.residual = residual and (in_size == out_size)
        layers = []
        if norm_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(in_size))
        elif norm_type == 'layer_norm':
            layers.append(nn.LayerNorm(in_size))
        elif norm_type is not None:
            raise NotImplementedError
        if act_cls is not None:
            layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_size, out_size, bias))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        z = self.f(x)
        if self.residual:
            z += x
        return z


class ExplorerPolicy(nn.Module):

    def __init__(self, hidden_size, norm_type=None):
        super().__init__()
        self.hidden_size = hidden_size

        self.seed_embedding = nn.Linear(1, hidden_size, bias=False)
        self.node_embedding = nn.Linear(1, hidden_size, bias=False)
        self.abs_diff_embedding = nn.Linear(1, hidden_size, bias=False)

        self.input_mapping = nn.Sequential(
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
            make_linear_block(hidden_size, hidden_size, Swish, norm_type)
        )
        self.node_score_layer = nn.Linear(hidden_size, 1, bias=False)
        self.stopping_score_layer = nn.Linear(hidden_size, 2, bias=False)

        nn.init.zeros_(self.node_score_layer.weight.data)
        nn.init.zeros_(self.stopping_score_layer.weight.data)

    def forward(self, x_seeds, x_nodes, x_abs_diff, indptr):
        """Score candidates with the community-deviation feature."""
        h = (
            self.seed_embedding(x_seeds.unsqueeze(1))
            + self.node_embedding(x_nodes.unsqueeze(1))
            + self.abs_diff_embedding(x_abs_diff.unsqueeze(1))
        )
        h = self.input_mapping(h)
        node_scores = self.node_score_layer(h).squeeze(1)

        segments = []
        stop_nodes = []

        for startpoint, endpoint, candidate_endpoint in indptr:
            startpoint = int(startpoint)
            endpoint = int(endpoint)
            candidate_endpoint = int(candidate_endpoint)

            if startpoint == endpoint:
                raise ValueError('Finished Episode!')

            segments.append((startpoint, endpoint, candidate_endpoint))
            stop_nodes.append(torch.mean(h[startpoint:endpoint], dim=0))

        stop_nodes = torch.stack(stop_nodes, dim=0)
        stopping_logits_all = torch.log_softmax(
            self.stopping_score_layer(stop_nodes), dim=1
        )

        batch = []
        for idx, (startpoint, endpoint, candidate_endpoint) in enumerate(segments):
            node_logits = torch.log_softmax(node_scores[startpoint:candidate_endpoint], dim=0)
            stopping_logits = stopping_logits_all[idx]
            logits = torch.cat(
                [node_logits + stopping_logits[0], stopping_logits[1:].contiguous()],
                dim=0
            )
            batch.append(logits)

        batch_logits = batch
        return batch_logits


def build_policy(hidden_size, device):
    return ExplorerPolicy(hidden_size).to(device)
