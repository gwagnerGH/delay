# Economic Damages of Delayed Climate Action

This repository contains the model runs and analysis for the paper `The welfare cost of delayed climate policy`.

## Repository Structure

- `src/`: core model code, calibration utilities, simulation logic, and analysis helpers.
- `scripts/`: executable run scripts for local and cluster jobs. The `main_*` files run individual experiment families; the `run_*_array_job.sh` files are SGE array-job wrappers.
- `data/`: model inputs and output data.
- `data/new_outputs/`: paper run outputs used by the notebook.
- `notebooks/paper_facing_plots.ipynb`: final paper-facing analysis, tables, and figures.
- `aux_notebooks/`: auxiliary data preparation, including SSP baseline construction.

## Reproducing the Paper Results

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate EZClimate
```

Data output folders:

```text
damage-robustness-BY2025-samegrid-run0-v3
delay-frontier-BY2025-fiveyear-robustness-v1
delay-frontier-BY2025-fiveyear-run0-v1
delay-frontier-BY2025-robustness-v2
delay-frontier-BY2025-samegrid-run0-v4
ensemble-BY2025-samegrid-run0gauss-v5
mean-parameter-BY2025-samegrid-run0-v3
partial-mitigation-BY2025-samegrid-run0-cap-v3
preference-grid-BY2025-samegrid-run0-v3
technology-grid-BY2025-samegrid-run0-v3
tree-robustness-BY2025-samegrid-run0-v3
```

To print the exact local and SGE cluster commands (have to be adjusted for most HPCs) for those folders:

```bash
python scripts/reproduce_main_spec.py
```

Cluster submission assumes an SGE environment with `grid_run` available.

After the run outputs are present, regenerate the paper-facing tables and
figures using `notebooks/paper_facing_plots.ipynb`.

The resulting paper tables and figures are written to:

```text
data/new_outputs/paper_facing_plots/tables
data/new_outputs/paper_facing_plots/figures
```
