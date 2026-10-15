import argparse
import warnings
import os
import copy
import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "16"
warnings.filterwarnings("ignore", message=".*loky.*")
warnings.filterwarnings("ignore", message=".*CreateProcess.*")

# These imports intentionally follow the thread-limit environment setup above.
from trainer import (  # noqa: E402
    run_dynamic,
    resolve_dynamic_ablation_mode,
    ablation_mode_name,
)
from data import list_snapshots  # noqa: E402
from reporting import merge_worker_outputs, resolve_output_root  # noqa: E402


def _parse_gpu_ids(gpus_arg: str) -> list[str]:
    return [item.strip() for item in str(gpus_arg).split(",") if item.strip()]


def _resolve_launch_mode(args) -> bool:
    gpu_ids = _parse_gpu_ids(args.gpus)
    auto_multi = len(gpu_ids) > 1
    args.multi_gpu = bool(args.multi_gpu or auto_multi)
    return args.multi_gpu


def _chunk_seed_range(start: int, search_size: int, num_workers: int) -> list[tuple[int, int]]:
    if search_size <= 0 or num_workers <= 0:
        return []
    num_workers = min(num_workers, search_size)
    base = search_size // num_workers
    rem = search_size % num_workers
    chunks = []
    cur = start
    for i in range(num_workers):
        sz = base + (1 if i < rem else 0)
        if sz <= 0:
            continue
        chunks.append((cur, sz))
        cur += sz
    return chunks


def _print_configuration(args, prefix: str, include_window: bool = True) -> None:
    print(f"{prefix} Ablation mode = {ablation_mode_name(resolve_dynamic_ablation_mode(args))}", flush=True)
    print(
        f"{prefix} Graph setting: k_ego_subG={args.k_ego_subG}, "
        f"iGPN propagation layers L={getattr(args, 'L', 3)}",
        flush=True,
    )
    if getattr(args, "ablate_rcs", False):
        print(f"{prefix} Reference: current-snapshot seed neighborhood, k={args.ref_ego_k}", flush=True)
    if include_window:
        aggregation = "uniform_average" if args.ablate_decay else "temporal_decay_average"
        print(
            f"{prefix} Window setting: fixed size={args.window_size}, decay={args.window_decay}, "
            f"aggregation={aggregation}, "
            f"start_epochs={args.window_start_epochs}, inner_epochs={args.window_inner_epochs}",
            flush=True,
        )


def _worker_main(worker_args):
    worker_tag = getattr(worker_args, "worker_tag", "worker")
    device = getattr(worker_args, "device", "cpu")
    start = int(getattr(worker_args, "start", 0))
    search_size = int(getattr(worker_args, "search_size", 0))
    output_root = resolve_output_root(worker_args)

    print(
        f"[Worker {worker_tag}] start | device={device} | seed_range=[{start}, {start + search_size - 1}] | output_root={output_root}",
        flush=True,
    )
    _print_configuration(worker_args, f"[Worker {worker_tag}]", include_window=False)
    metrics = run_dynamic(worker_args)
    by_seed = metrics.get("by_seed", {})
    num_seed = len(by_seed)
    total_pairs = sum(len(v) for v in by_seed.values())
    print(
        f"[Worker {worker_tag}] done | device={device} | finished_seeds={num_seed} | evaluated_pairs={total_pairs}",
        flush=True,
    )


def _run_multi_gpu(args):
    gpu_ids = _parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("No valid GPU ids found in --gpus")

    snapshots = list_snapshots(args.dataset_root)
    if not snapshots:
        raise RuntimeError(f"No snapshots under {args.dataset_root}/snapshots")

    output_root = resolve_output_root(args)
    os.makedirs(output_root, exist_ok=True)

    chunks = _chunk_seed_range(int(args.start), int(args.search_size), len(gpu_ids))
    if not chunks:
        print("[Launch] search_size <= 0, nothing to run.", flush=True)
        return

    used_gpu_ids = gpu_ids[:len(chunks)]
    worker_tags = []

    print(f"[Launch] Multi-GPU mode enabled automatically. gpus={','.join(used_gpu_ids)}", flush=True)
    print(f"[Launch] Seed split: {chunks}", flush=True)
    _print_configuration(args, "[Launch]")

    ctx = mp.get_context("spawn")
    processes = []

    for idx, ((chunk_start, chunk_size), gpu_id) in enumerate(zip(chunks, used_gpu_ids)):
        worker_args = copy.deepcopy(args)
        worker_args.start = int(chunk_start)
        worker_args.search_size = int(chunk_size)
        worker_args.device = f"cuda:{gpu_id}"
        worker_args.output_root = output_root
        worker_args.worker_tag = f"worker_{idx:02d}_gpu{gpu_id}"
        worker_args.clear_existing_outputs = True
        worker_tags.append(worker_args.worker_tag)

        p = ctx.Process(target=_worker_main, args=(worker_args,), name=worker_args.worker_tag)
        p.start()
        processes.append(p)

    failed = []
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append((p.name, p.exitcode))

    if failed:
        raise RuntimeError(f"Some workers failed: {failed}")

    merged = merge_worker_outputs(
        output_root=output_root,
        worker_tags=worker_tags,
        num_snapshots=len(snapshots),
        clean_workers=bool(getattr(args, "clean_worker_outputs", False)),
    )
    print("[Merge] Worker outputs merged successfully.", flush=True)
    print(f"[Merge] nmi_log: {merged['nmi_log']}", flush=True)
    print(f"[Merge] seed_time_log: {merged['seed_time_log']}", flush=True)
    print(f"[Merge] seed_avg_time_log: {merged['seed_avg_time_log']}", flush=True)
    print(f"[Merge] pred_dir: {merged['pred_dir']}", flush=True)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset_root", type=str, default="datasets/dblp_expandshrink")
    p.add_argument("--start", type=int, default=0, help="start index in seed list")
    p.add_argument("--search_size", type=int, default=100, help="how many seeds to run")
    p.add_argument("--seed", type=int, default=0, help="random seed")

    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--gpus", type=str, default="1",
                   help="GPU ids, e.g. 0 or 0,7. If multiple ids are provided, multi-GPU is enabled automatically.")
    p.add_argument("--multi_gpu", action="store_true",
                   help="Optional compatibility switch. Multi-GPU is now auto-enabled when --gpus contains multiple ids.")
    p.add_argument("--output_root", type=str, default="",
                   help="Final output root. Default: dataset_root")
    p.add_argument("--clean_worker_outputs", action="store_true",
                   help="Delete worker_xx directories after merge")

    p.add_argument("--train_size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--g_lr", type=float, default=1e-2)
    p.add_argument("--g_batch_size", type=int, default=16)
    p.add_argument("--k", type=int, default=50,
                   help="Top-K known communities selected by similarity to the reference community.")
    p.add_argument("--k_ego_subG", type=int, default=3,
                   help="Initial working-subgraph and inference neighborhood radius; independent of --L.")
    p.add_argument("--L", type=int, default=3,
                   help="Number of iGPN propagation layers used during training and inference.")

    p.add_argument("--window_size", type=int, default=4,
                   help="Fixed window size used by the window-level master/fast-params framework.")
    p.add_argument("--window_start_epochs", type=int, default=50,
                   help="Epochs for the first snapshot in each window.")
    p.add_argument("--window_inner_epochs", type=int, default=50,
                   help="Epochs for non-first snapshots inside each window.")
    p.add_argument("--init_pretrain_epochs", type=int, default=50,
                   help="Epochs for one-time generic policy pretraining on known communities in the first snapshot.")
    p.add_argument("--disable_init_pretrain", action="store_true",
                   help="Disable generic initial policy pretraining and use random master initialization.")
    p.add_argument("--window_decay", type=float, default=0.95)

    ablations = p.add_mutually_exclusive_group()
    ablations.add_argument("--ablate_sim", action="store_true",
                           help="w/o sim: supervise directly with known communities, without reference community retrieval.")
    ablations.add_argument("--ablate_win", action="store_true",
                           help="w/o win: disable window-level master policy and reset parameters each snapshot.")
    ablations.add_argument("--ablate_cc", action="store_true",
                           help="w/o cc: use the previous result directly as the warm-start reference.")
    ablations.add_argument("--ablate_decay", action="store_true",
                           help="w/o decay: average window gradients without temporal-decay weights.")
    ablations.add_argument("--ablate_rcs", action="store_true",
                           help="w/o rcs: use the current seed's local neighborhood as reference.")
    p.add_argument("--ref_ego_k", type=int, default=3,
                   help="Seed-neighborhood reference radius for --ablate_rcs; independent of k_ego_subG and L.")
    args = p.parse_args()
    if args.ref_ego_k < 0:
        p.error("--ref_ego_k must be nonnegative")
    if args.L < 1:
        p.error("--L must be at least 1")

    _ = resolve_dynamic_ablation_mode(args)

    using_multi_gpu = _resolve_launch_mode(args)

    if using_multi_gpu:
        _run_multi_gpu(args)
    else:
        print(f"[Launch] Single-GPU mode. device={args.device}", flush=True)
        _print_configuration(args, "[Launch]")
        metrics = run_dynamic(args)

        scores = [score for items in metrics["by_seed"].values() for _, score in items]
        total = len(scores)
        if total > 0:
            f1 = sum(score["f1"] for score in scores) / total
            nmi = sum(score["nmi"] for score in scores) / total
            print(f"\nOverall avg over {total} (seed,snapshot) pairs: F1={f1:.4f}, NMI={nmi:.4f}")
        else:
            print("\nNo evaluated pairs. Check seed_cmty_index files and snapshot structure.")


if __name__ == "__main__":
    main()
