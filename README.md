# DISRL: source code

This repository contains the code for the dynamic local community detection method and the synthetic dynamic benchmark generator.

## Layout

- `method/`: training, inference, and evaluation code.
- `benchmark/`: dynamic graph generation from initial graph and community files.

## Environment

Use Python 3.10 with the packages in `requirements.txt`. Install a PyTorch build compatible with your CUDA installation if using a GPU.

## Datasets

[Download the datasets](./dataset_9e.tar.xz)

Extract the archive from the repository root:

```bash
tar -xJf dataset_9e.tar.xz
```

The archive extracts into `dataset/` and contains `amazon_expandshrink`, `dblp_expandshrink`, `amazon_mergesplit`, `dblp_mergesplit`, `twitter_expandshrink`, `youtube_expandshrink`, and `football`.

## Run the method

The dataset root must contain `snapshots/t000/`, `snapshots/t001/`, and so on. Each snapshot uses files named `tXXX_ungraph.txt`, `tXXX_cmty.txt`, `tXXX_seed.txt`, `tXXX_seed_cmty_index.txt`, and `tXXX_row2cid.txt`.

For example, after extracting the datasets, run from `method/`:

```bash
python run.py --dataset_root ../dataset/amazon_expandshrink --output_root ../outputs/amazon_expandshrink --start 0 --search_size 100 --seed 0 --device cuda:0 --gpus 0
```

Use `python run.py --help` for the available parameters and ablation flags. The evaluation results are written under the selected output root.

## Generate a benchmark

Edit `benchmark/config.py` to choose a dataset, evolution mode, and input/output roots, then run:

```bash
cd benchmark
python run_realdata_evolution.py
```

The downloadable archive contains the generated graph snapshots. Initial input files for generating new benchmarks must be provided separately.
