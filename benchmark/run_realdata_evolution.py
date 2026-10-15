"""Run dynamic community evolution on real static datasets."""

import os

from config import (
    DATA_ROOT,
    DATASETS,
    EVOLUTION_MODE,
    HEAD_EVOLVE_K,
    OUTPUT_ROOT,
    SEED_EVOLVE_K,
    TAIL_EVOLVE_K,
    TIMESTEPS,
    USE_SEEDS,
)
from data_io import (
    load_communities,
    load_graph,
    load_seed_context,
)
from evolution import (
    AddNodesRule,
    ExpandRule,
    RemoveEdgesRule,
    RemoveNodesRule,
    ShrinkRule,
)
from events import ParallelPairwiseMergeEvent, RandomRecordedSplitEvent
from scheduler import Scheduler
from simulation import GraphMovementSimulation


def build_scheduler(mode, timesteps):
    """Create the evolution schedule for all snapshots."""
    scheduler = Scheduler()

    if mode == "expand_shrink":
        expand_rule = ExpandRule(
            comm_frac=0.5,
            add_ratio=0.5,
            max_add_per_comm=3,
            edge_frac=0.5,
            max_edges_per_new=5,
            force_head_k=HEAD_EVOLVE_K,
            force_tail_k=TAIL_EVOLVE_K,
        )
        shrink_rule = ShrinkRule(
            comm_frac=0.5,
            shrink_frac=0.2,
            max_rm_per_comm=3,
            min_comm_size=3,
            remove_isolated=True,
            force_head_k=HEAD_EVOLVE_K,
            force_tail_k=TAIL_EVOLVE_K,
        )
        scheduler.add_rule_window(1, 7, [expand_rule])
        scheduler.add_rule_window(7, timesteps, [shrink_rule])
    elif mode == "merge_split":
        scheduler.add_event(
            ParallelPairwiseMergeEvent(
                start=1,
                duration=4,
                m_frac=0.8,
                min_comm_size=3,
                tail_merge_frac=0.4,
                bridge_frac=0.05,
                bridge_cap=2000,
            )
        )
        scheduler.add_event(
            RandomRecordedSplitEvent(
                start=5,
                duration=4,
            )
        )
    else:
        raise ValueError(
            f"Unsupported EVOLUTION_MODE={mode!r}; use 'expand_shrink' "
            "or 'merge_split'."
        )

    scheduler.add_rule_window(
        0,
        timesteps,
        [
            AddNodesRule(protect_seeds=USE_SEEDS),
            RemoveNodesRule(protect_seeds=USE_SEEDS),
            RemoveEdgesRule(protect_seeds=USE_SEEDS),
        ],
    )

    return scheduler


def run_dataset(name):
    print(f"\n========== Processing dataset: {name} ==========")

    dataset_root = os.path.join(DATA_ROOT, name)
    ungraph_path = os.path.join(dataset_root, "t000_ungraph.txt")
    cmty_path = os.path.join(dataset_root, "t000_cmty.txt")

    out_root = os.path.join(OUTPUT_ROOT, f"{name}_{EVOLUTION_MODE}")
    os.makedirs(out_root, exist_ok=True)

    graph = load_graph(ungraph_path)
    multi_labels = load_communities(cmty_path)

    seeds, seed2cid = load_seed_context(
        dataset_root,
        enabled=USE_SEEDS,
        limit=SEED_EVOLVE_K,
    )

    if seeds:
        print(f"  mode: local seed-aware evolution ({len(seeds)} seeds)")
    else:
        print("  mode: global community evolution (no seeds)")

    sim = GraphMovementSimulation(
        graph,
        multi_labels,
        seed2cid=seed2cid,
        seeds=seeds,
        tail_evolve_k=TAIL_EVOLVE_K,
    )

    scheduler = build_scheduler(EVOLUTION_MODE, TIMESTEPS)

    sim.run(
        graph,
        timesteps=TIMESTEPS,
        scheduler=scheduler,
        out_root=out_root,
        seeds=seeds,
        write_seed_files=bool(seeds),
    )

    print(f"  Done. snapshots saved under: {os.path.join(out_root, 'snapshots')}")


def main():
    if not DATASETS:
        print("No datasets configured. Edit DATASETS in config.py before running.")
        return

    print(f"Evolution mode: {EVOLUTION_MODE}; timesteps: {TIMESTEPS}")

    for name in DATASETS:
        run_dataset(name)


if __name__ == "__main__":
    main()
