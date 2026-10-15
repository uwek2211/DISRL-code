import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(value):
    value = str(value)
    return torch.device(value if torch.cuda.is_available() and "cuda" in value else "cpu")


def overlap_scores(predicted, target, epsilon=0.0):
    predicted, target = set(predicted), set(target)
    intersection = len(predicted & target)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(target) if target else 0.0
    denominator = precision + recall + epsilon
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    union = len(predicted | target)
    return precision, recall, f1, intersection / union if union else 0.0


def clone_params(params):
    return [p.detach().clone() for p in params]


def model_params(model):
    return clone_params(model.parameters())


def load_params(model, params) -> None:
    with torch.no_grad():
        for target, source in zip(model.parameters(), params):
            target.copy_(source.to(device=target.device, non_blocking=True))


def gradient_step(params, grads, lr):
    return [
        p.detach() - lr * g.detach().to(device=p.device, non_blocking=True)
        for p, g in zip(params, grads)
    ]
