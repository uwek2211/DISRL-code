from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import os
import time
import torch

from detector import Detector
from policy import build_policy
from graph import Graph
from reporting import RunReporter, log_prefix, resolve_output_root

from data import build_snapshot_cache, list_snapshots, read_fixed_seeds, read_seed_cmty_index
from utils import clone_params, gradient_step, load_params, model_params, resolve_device, seed_all


@torch.no_grad()
def aggregate_window_gradients(
    window_grads: List[List[torch.Tensor]], decay: float, use_temporal_decay: bool = True
) -> List[torch.Tensor]:
    if not window_grads:
        return []
    aggregated = [torch.zeros_like(g) for g in window_grads[0]]
    device = aggregated[0].device
    if any(g.device != device for step in window_grads for g in step):
        window_grads = [[g.to(device, non_blocking=True) for g in step] for step in window_grads]
    weights = (
        [float(decay) ** (len(window_grads) - 1 - i) for i in range(len(window_grads))]
        if use_temporal_decay else [1.0] * len(window_grads)
    )
    total = float(sum(weights))
    weights = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)
    for weight, gradients in zip(weights, window_grads):
        for target, gradient in zip(aggregated, gradients):
            target.add_(weight * gradient)
    return aggregated


AB_MODE_DISRL = "disrl"
AB_MODE_WO_SIM = "wo_sim"
AB_MODE_WO_WIN = "wo_win"
AB_MODE_WO_CC = "wo_cc"
AB_MODE_WO_DECAY = "wo_decay"
AB_MODE_WO_RCS = "wo_rcs"
AB_MODE_NAMES = {
    AB_MODE_DISRL: "DISRL",
    AB_MODE_WO_SIM: "w/o sim",
    AB_MODE_WO_WIN: "w/o win",
    AB_MODE_WO_CC: "w/o cc",
    AB_MODE_WO_DECAY: "w/o decay",
    AB_MODE_WO_RCS: "w/o rcs",
}


def resolve_dynamic_ablation_mode(args) -> str:
    mode = AB_MODE_DISRL
    for flag, ablation in (
        ("ablate_sim", AB_MODE_WO_SIM),
        ("ablate_win", AB_MODE_WO_WIN),
        ("ablate_cc", AB_MODE_WO_CC),
        ("ablate_decay", AB_MODE_WO_DECAY),
        ("ablate_rcs", AB_MODE_WO_RCS),
    ):
        if bool(getattr(args, flag, False)):
            mode = ablation
            break
    args.ablation_mode = mode
    return mode


def ablation_mode_name(mode: str) -> str:
    return AB_MODE_NAMES.get(mode, str(mode))


def build_fixed_windows(T: int, size: int) -> List[Tuple[int, int]]:
    if T <= 0:
        return []
    size = max(1, int(size))
    return [(start, min(start + size, T)) for start in range(0, T, size)]


def pretrain_initial_policy(
    args,
    *,
    snapshot: str,
    snapshot_cache: Dict,
    fixed_seeds: List[int],
    seed2cmty: Dict[int, int],
    device: torch.device,
) -> List[torch.Tensor]:
    epochs = max(0, int(getattr(args, "init_pretrain_epochs", 50)))
    pretrain_seed = next((int(seed) for seed in fixed_seeds if int(seed) in seed2cmty), None)
    if epochs <= 0 or pretrain_seed is None:
        args.use_pretrained_init_policy = False
        reason = "init_pretrain_epochs <= 0" if epochs <= 0 else "no valid seed found in first snapshot"
        print(
            f"[Init Pretrain] {reason}; use random initial policy.",
            flush=True,
        )
        return model_params(build_policy(args.hidden_size, device))

    print(
        f"[Init Pretrain] Start generic policy pretraining on {snapshot} known communities | "
        f"seed_anchor={pretrain_seed} | epochs={epochs}",
        flush=True,
    )
    tic = time.perf_counter()

    # The initializer is shared by all evaluation seeds, so its supervision
    # pool must exclude every evaluation target community, not only the anchor.
    all_target_community_indices = {
        int(seed2cmty[int(raw_seed)]) for raw_seed in fixed_seeds if int(raw_seed) in seed2cmty
    }

    det = Detector.from_snapshot_cache(
        args,
        seed_raw=pretrain_seed,
        com_index=int(seed2cmty[pretrain_seed]),
        cache=snapshot_cache,
        snapshot=snapshot,
        excluded_com_indices=all_target_community_indices,
    )
    removed_count = len(set(det.base_known_community_indices).intersection(all_target_community_indices))
    print(
        f"[Init Pretrain] Supervision split | base_known="
        f"{len(det.base_known_community_indices)} | excluded_target_communities="
        f"{removed_count} | train_known={len(det.known_community_indices)}",
        flush=True,
    )
    det.relevant_comms = det.known_comms_subgraph
    det._refresh_dynamic_expand_max_size()

    det.train_epochs(epochs)

    elapsed = time.perf_counter() - tic
    print(f"[Init Pretrain] Done | elapsed={elapsed:.6f}s", flush=True)
    return model_params(det.explorer.model)


def _run_snapshot(
    args,
    mode: str,
    raw_seed: int,
    snapshot: str,
    seed_to_community: Dict[int, int],
    cache: Dict,
    initial_params: List[torch.Tensor],
    previous_community: Optional[List[int]],
    previous_graph: Optional[Graph],
):
    if raw_seed not in seed_to_community:
        return None
    community_index = int(seed_to_community[raw_seed])
    try:
        detector = Detector.from_snapshot_cache(args, raw_seed, community_index, cache, snapshot)
    except KeyError:
        return None

    started = time.perf_counter()
    load_params(detector.explorer.model, initial_params)
    result, _, updated_params = detector.detect(
        previous_community,
        previous_graph,
        direct_supervision=mode == AB_MODE_WO_SIM,
        return_intermediate=True,
    )
    return (
        detector,
        community_index,
        result[2],
        updated_params,
        time.perf_counter() - started,
    )


def run_dynamic(args) -> Dict:
    seed_all(getattr(args, "seed", 0))
    mode = resolve_dynamic_ablation_mode(args)

    dataset_root = args.dataset_root
    output_root = resolve_output_root(args, include_worker=True)
    base_root = os.path.dirname(dataset_root.rstrip("/"))
    base_dataset = os.path.basename(dataset_root.rstrip("/"))

    args.root = base_root
    args.dataset = base_dataset

    snapshots = list_snapshots(dataset_root)
    if not snapshots:
        raise RuntimeError(f"No snapshots under {dataset_root}/snapshots")

    fixed_seeds = read_fixed_seeds(dataset_root, snapshot0=snapshots[0])

    start = int(getattr(args, "start", 0))
    search_size = int(getattr(args, "search_size", len(fixed_seeds)))
    seeds_to_run = fixed_seeds[start : start + search_size]

    device = resolve_device(getattr(args, "device", "cuda:0"))

    T = len(snapshots)
    base_epochs = max(1, int(getattr(args, "epochs", 100)))
    window_start_epochs = max(1, int(getattr(args, "window_start_epochs", base_epochs)))
    window_inner_epochs = max(1, int(getattr(args, "window_inner_epochs", base_epochs)))
    args.window_start_epochs = window_start_epochs
    args.window_inner_epochs = window_inner_epochs

    reporter = RunReporter(
        args,
        output_root,
        snapshots,
        seeds_to_run,
        metadata={
            "k_ego_subG": int(getattr(args, "k_ego_subG", 3)),
            "L": int(getattr(args, "L", 3)),
            "ablation_mode": mode,
            "ablate_sim": mode == AB_MODE_WO_SIM,
            "ablate_rcs": mode == AB_MODE_WO_RCS,
            "ref_ego_k": int(getattr(args, "ref_ego_k", 3)),
            "ablate_win": mode == AB_MODE_WO_WIN,
            "ablate_cc": mode == AB_MODE_WO_CC,
            "ablate_decay": mode == AB_MODE_WO_DECAY,
            "use_pretrained_init_policy": None,
            "init_pretrain_epochs": int(getattr(args, "init_pretrain_epochs", 50)),
            "base_epochs": base_epochs,
            "window_start_epochs": window_start_epochs,
            "window_inner_epochs": window_inner_epochs,
        },
        clear_existing=bool(getattr(args, "clear_existing_outputs", True)),
    )
    metrics = reporter.metrics

    seed2cmty_by_t = {
        t: read_seed_cmty_index(dataset_root, snapshot, fixed_seeds) for t, snapshot in enumerate(snapshots)
    }
    snapshot_cache = {
        snapshot: build_snapshot_cache(args, args.root, args.dataset, snapshot) for snapshot in snapshots
    }

    use_master_policy = mode != AB_MODE_WO_WIN

    windows = build_fixed_windows(T, int(getattr(args, "window_size", 4))) if use_master_policy else []

    theta_init: Optional[List[torch.Tensor]] = None
    use_init_pretrain = use_master_policy and not bool(getattr(args, "disable_init_pretrain", False))
    args.use_pretrained_init_policy = bool(use_init_pretrain)
    metrics["use_pretrained_init_policy"] = bool(use_init_pretrain)
    if use_init_pretrain:
        theta_init = pretrain_initial_policy(
            args,
            snapshot=snapshots[0],
            snapshot_cache=snapshot_cache[snapshots[0]],
            fixed_seeds=fixed_seeds,
            seed2cmty=seed2cmty_by_t[0],
            device=device,
        )
        metrics["use_pretrained_init_policy"] = bool(getattr(args, "use_pretrained_init_policy", False))
    elif use_master_policy:
        print(
            "[Init Pretrain] Disabled; each seed uses random master initialization.",
            flush=True,
        )

    for raw_seed in seeds_to_run:
        raw_seed = int(raw_seed)
        print(
            f"\n{log_prefix(args)} ========== Seed {raw_seed} start ==========\n",
            flush=True,
        )
        print(
            f"{log_prefix(args)} Seed {raw_seed} | ablation mode = {ablation_mode_name(mode)}",
            flush=True,
        )

        if use_master_policy:
            master_params = clone_params(theta_init) if theta_init is not None else model_params(
                build_policy(args.hidden_size, device)
            )
            window_decay = float(getattr(args, "window_decay", 0.95))
            groups = windows
            aggregation = "uniform_average" if mode == AB_MODE_WO_DECAY else "temporal_decay_average"
            print(
                f"{log_prefix(args)} Seed {raw_seed} | windows = ON | "
                "master carry = ON | "
                f"aggregation={aggregation} | "
                f"decay={window_decay} | "
                f"snapshots={T} | windows={len(windows)} | epoch schedule="
                f"{window_start_epochs}/{window_inner_epochs}",
                flush=True,
            )
        else:
            reset_params = model_params(build_policy(args.hidden_size, device))
            groups = [(t, t + 1) for t in range(T)]
            print(
                f"{log_prefix(args)} Seed {raw_seed} | DISRL master policy = OFF | "
                "snapshot-wise parameter reset enabled",
                flush=True,
            )

        previous_community = previous_graph = None
        for group_index, (start_t, end_t) in enumerate(groups):
            snapshot_ids = list(range(start_t, end_t))
            if use_master_policy:
                current_params = clone_params(master_params)
                window_buffer: List[List[torch.Tensor]] = []
                print(
                    f"\n{log_prefix(args)} Window {group_index + 1}/{len(groups)} | snapshots={snapshot_ids}",
                    flush=True,
                )
            else:
                current_params = reset_params

            for position, t in enumerate(snapshot_ids):
                snapshot = snapshots[t]
                args.epochs = (
                    window_start_epochs if position == 0 else window_inner_epochs
                ) if use_master_policy else base_epochs
                print(
                    f"{log_prefix(args)} Snapshot t{t:03d} ({snapshot}) | epochs={args.epochs} | "
                    f"policy_init={'window_fast' if use_master_policy else 'snapshot_reset'}",
                    flush=True,
                )
                result = _run_snapshot(
                    args, mode, raw_seed, snapshot, seed2cmty_by_t[t], snapshot_cache[snapshot],
                    current_params, previous_community, previous_graph,
                )
                if result is None:
                    continue
                detector, community_index, prediction, updated_params, elapsed = result
                previous_community, previous_graph = prediction, detector.graph
                if use_master_policy:
                    current_params = clone_params(updated_params)
                    if position < len(snapshot_ids) - 1:
                        next_t = snapshot_ids[position + 1]
                        next_mapping = seed2cmty_by_t[next_t]
                        if raw_seed in next_mapping:
                            try:
                                next_detector = Detector.from_snapshot_cache(
                                    args, raw_seed, int(next_mapping[raw_seed]),
                                    snapshot_cache[snapshots[next_t]], snapshots[next_t],
                                )
                                window_buffer.append(next_detector.transition_gradient(
                                    updated_params,
                                    previous_community=prediction,
                                    previous_graph=detector.graph,
                                    direct_supervision=mode == AB_MODE_WO_SIM,
                                ))
                            except KeyError:
                                pass
                reporter.record(detector, raw_seed, t, community_index, prediction, elapsed)

            if use_master_policy and window_buffer:
                master_params = gradient_step(
                    master_params,
                    aggregate_window_gradients(
                        window_buffer, window_decay, use_temporal_decay=mode != AB_MODE_WO_DECAY
                    ),
                    float(getattr(args, "g_lr", 1e-2)),
                )

        args.epochs = base_epochs

        print(
            f"{log_prefix(args)} ========== Seed {raw_seed} end ==========\n",
            flush=True,
        )

    return reporter.finalize()
