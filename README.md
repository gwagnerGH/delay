# Economic Damages of Delayed Climate Action

This repository contains the model runs and analysis for the delayed climate
action paper.

## Reproducing the Paper Results

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate EZClimate
```

The paper-facing analysis reads run outputs from `data/new_outputs/`. The
current input folders are:

```text
mean-parameter-BY2025-samegrid-run0-v3
partial-mitigation-BY2025-samegrid-run0-cap-v3
tree-robustness-BY2025-samegrid-run0-v3
preference-grid-BY2025-samegrid-run0-v3
technology-grid-BY2025-samegrid-run0-v3
damage-robustness-BY2025-samegrid-run0-v3
ensemble-BY2025-samegrid-run0gauss-v5
```

To print the exact local and SGE cluster commands for those folders:

```bash
python scripts/reproduce_main_spec.py
```

To run the local benchmarks:

```bash
python scripts/reproduce_main_spec.py --execute-local
```

To submit the cluster array jobs:

```bash
python scripts/reproduce_main_spec.py --submit-cluster
```

Cluster submission assumes an SGE environment with `grid_run` available. The
wrapper scripts activate the `EZClimate` conda environment by default; set
`EZDELAY_CONDA_ENV=<name>` before submission to use a different environment.

The cluster wrapper scripts are `scripts/run_*_array_job.sh`; each calls the
matching `scripts/main_*_cluster.py` script and writes results under
`data/new_outputs/$OUTPUT_FOLDER`. Use the `OUTPUT_FOLDER` names above when
rerunning individual jobs by hand. The paper-facing frontier is built from
period-length-5 robustness outputs; annual period-length-1 frontier outputs are
not included in this release. The full Gaussian ensemble rerun is `1-9000` for
`ensemble-BY2025-samegrid-run0gauss-v5`.

After the run outputs are present, regenerate the paper-facing tables and
figures with:

```bash
jupyter nbconvert --to notebook --execute notebooks/paper_facing_plots.ipynb --inplace
```

The resulting paper tables and figures are written to:

```text
data/new_outputs/paper_facing_plots/tables
data/new_outputs/paper_facing_plots/figures
```
