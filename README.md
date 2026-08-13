# Economic Damages of Delayed Climate Action

This repository contains the code, inputs, run outputs, and analysis notebooks
for the paper ``Economic Damages of Delayed Climate Action.''

## Repository layout

- `src/` contains the model and reusable analysis code.
- `scripts/` contains local entry points, cluster wrappers, diagnostics, and
  postprocessing utilities.
- `data/` contains model inputs. Current paper outputs are in
  `data/new_outputs/`.
- `notebooks/paper_facing_plots.ipynb` regenerates the paper figures and
  tables. `notebooks/frontier_tree_metrics.ipynb` produces the decision trees.
- `aux_notebooks/` contains supporting input-preparation notebooks.
- `tests/` contains regression and validation tests.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate econ
```

## Current paper outputs

The paper-facing notebook reads the following current run folders:

```text
paper-august-main-frontier-v2
paper-august-robustness-frontier-v2
paper-august-partial-mitigation-array-v2
paper-august-tree-array-v2
paper-august-preference-array-v2
paper-august-technology-array-v2
paper-august-damage-array-v2
paper-august-mac-shift-array-v2
paper-august-gaussian-ensemble-array-v2
```

The figure and table artifacts are written to:

```text
data/new_outputs/paper_facing_plots/figures
data/new_outputs/paper_facing_plots/tables
```

## Regenerating paper figures

To regenerate the figures, run ``notebooks/paper_facing_plots.ipynb``.

## Running model scripts

Run local entry points from the repository root, for example:

```bash
python scripts/main.py
python scripts/main_delayed.py
```

Cluster wrappers are the `scripts/run_*_array_job.sh` files. They assume an
SGE-style environment and may require site-specific paths, scheduler options,
and a conda environment name. Set `EZDELAY_CONDA_ENV` when a cluster uses a
different environment name.
