# Economic Damages of Delayed Climate Action

This repository contains the model runs and analysis for the delayed climate
action paper.

## Repository Architecture

- `src/` contains the EZClimate model code and reusable analysis utilities.
- `scripts/` contains the local entry points and SGE cluster array wrappers used
  to recreate the paper runs.
- `data/` contains model inputs and `data/new_outputs/` contains the current
  paper run outputs used by the analysis notebook.
- `notebooks/paper_facing_plots.ipynb` regenerates the paper-facing tables and
  figures from the run outputs.
- `aux_notebooks/` contains the SSP baseline notebook and supporting data used
  for input preparation.

## Reproducing the Paper Results

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate EZClimate
```

The paper-facing analysis reads run outputs from `data/new_outputs/`. The
current input folders are:

```text
damage-robustness-BY2025-cons2025-samegrid-run0-v1
delay-frontier-BY2025-fiveyear-cons2025-robustness-v2
delay-frontier-BY2025-fiveyear-cons2025-run0-v2
ensemble-BY2025-cons2025-samegrid-run0gauss-eisfix-v1
partial-mitigation-BY2025-cons2025-samegrid-run0-cap-v1
preference-grid-BY2025-cons2025-samegrid-run0-v1
technology-grid-BY2025-cons2025-samegrid-run0-v1
tree-robustness-BY2025-cons2025-samegrid-run0-v1
```

To print the exact local and SGE cluster commands for those folders:

```bash
python scripts/reproduce_main_spec.py
```

Cluster submission assumes an SGE environment with `grid_run` available. Those commands will likely have to be adjusted.

After the run outputs are present, regenerate the paper-facing tables and
figures using `notebooks/paper_facing_plots.ipynb`

The resulting paper tables and figures are written to:

```text
data/new_outputs/paper_facing_plots/tables
data/new_outputs/paper_facing_plots/figures
```
