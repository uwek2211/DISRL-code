# DISRL: source code

This repository contains the code for the dynamic local community detection method and the synthetic dynamic benchmark generator.

## Layout

- `method/`: training, inference, and evaluation code.
- `benchmark/`: dynamic graph generation from initial graph and community files.

## Environment

Use Python 3.10 with the packages in `requirements.txt`. Install a PyTorch build compatible with your CUDA installation if using a GPU.

## Run the method

The dataset root must contain `snapshots/t000/`, `snapshots/t001/`, and so on. Each snapshot uses files named `tXXX_ungraph.txt`, `tXXX_cmty.txt`, `tXXX_seed.txt`, `tXXX_seed_cmty_index.txt`, and `tXXX_row2cid.txt`.

From `method/`, run:

```bash
python run.py --dataset_root /path/to/dataset --output_root /path/to/output --start 0 --search_size 100 --seed 0 --device cuda:0 --gpus 0
```

Use `python run.py --help` for the available parameters and ablation flags. The evaluation results are written under the selected output root.

## Generate a benchmark

Edit `benchmark/config.py` to choose a dataset, evolution mode, and input/output roots, then run:

```bash
cd benchmark
python run_realdata_evolution.py
```

Initial datasets and generated graph snapshots are not included in this source repository.
