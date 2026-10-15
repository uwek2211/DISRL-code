"""User-editable settings for the dynamic benchmark generator."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Main settings
# Modes: "expand_shrink" or "merge_split".
EVOLUTION_MODE = "expand_shrink"

# Available datasets: amazon, dblp, youtube, twitter, and psd.
# Merge-split is intended for Amazon and DBLP.
DATASETS = ["amazon"]

# Input and output directories. Replace these Path values when using custom data.
DATA_ROOT = PACKAGE_ROOT / "initial_datasets"
OUTPUT_ROOT = PACKAGE_ROOT / "generated_datasets"

# Seed-aware evolution
USE_SEEDS = True
# None keeps all seeds listed for the dataset.
SEED_EVOLVE_K = None

# Protect the final t=0 communities as training anchors.
TAIL_EVOLVE_K = 100
HEAD_EVOLVE_K = 0

# Protocol-derived settings
TIMESTEPS_BY_MODE = {"expand_shrink": 13, "merge_split": 9}
if EVOLUTION_MODE not in TIMESTEPS_BY_MODE:
    raise ValueError(f"Unsupported EVOLUTION_MODE: {EVOLUTION_MODE!r}")
TIMESTEPS = TIMESTEPS_BY_MODE[EVOLUTION_MODE]
