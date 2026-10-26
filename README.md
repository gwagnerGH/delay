# The welfare cost of delayed climate policy: reproduction package

This is the paper-facing snapshot of the corrected EZDelay model, run scripts, certified R&R result tables, and figure code for *The welfare cost of delayed climate policy* (Kent D. Daniel, Robert B. Litterman, Theo Moers, and Gernot Wagner; [gwagner.com/delay](https://gwagner.com/delay)).

## What is included

- `src/` contains the revised EZDelay model using the adjoint objective. The baseline is RA 10, EIS 1.5, annual PRTP 0.5%, consumption growth 1.5%, exogenous abatement technology progress 1.5%, no endogenous learning, and IID post-2250 log-growth with mean 1.5% and volatility 2%. See `src/config/revised_baseline.py` and `src/config/terminal_growth.py` for the executable defaults.
- `scripts/` contains the model entry points, SGE array wrappers, figure regeneration, and supplementary sensitivity runners. `python scripts/reproduce_main_spec.py` prints the paper's main run suite without submitting cluster jobs.
- `data/revised_default/` contains the saved damage grids and warm starts, although they can be regnereated from each main script; `data/*.csv` contains emissions and calibration inputs.
- `data/R_and_R/` contains 11 consolidated result bundles for the main frontier, partial mitigation, tree, preference, technology, damage, backstop, growth, ensemble, and MAC-shift exhibits. These are the numerical inputs to `notebooks/paper_facing_plots.ipynb`. Larger CSVs are stored as `.csv.gz`; the notebook reads either format automatically. Each bundle's `README.json` and `certified_case_sources.csv` give provenance and case counts. The cluster's individual raw job folders are not included but can be regenerated using the python wrappers under `scripts/`.
- `data/si_sensitivities/` has the learning-measure and long-run cost-floor results underlying the SI tables, including their saved run diagnostics.
- `paper_outputs/` is the figure and table snapshot used for the manuscript and SI. The nine `figures/` assets named in the supplied manuscript and SI are present. `notebooks/paper_facing_plots.ipynb` regenerates them.
- `tests/` contains model and regression checks.

The paper exhibits map to the retained array runners as follows:

| Paper result | Runner in `scripts/` | Saved bundle in `data/R_and_R/` |
| --- | --- | --- |
| Main delay frontier | `run_delay_frontier_array_job.sh` | `download-delay-frontier-main` |
| Frontier preference and learning checks | `run_delay_frontier_array_job.sh` | `download-delay-frontier-robustness` |
| Partial mitigation | `run_partial_array_job.sh` | `download-partial` |
| Tree and damage checks | `run_tree_robustness_array_job.sh`, `run_damage_robustness_array_job.sh` | `download-tree`, `download-damage` |
| Preference, technology, and removal checks | `run_preference_grid_array_job.sh`, `run_technology_grid_array_job.sh`, `run_backstop_delay_array_job.sh` | `download-preference`, `download-technology`, `download-backstop` |
| Long-run growth and MAC shifts | `run_growth_sensitivity_array_job.sh`, `run_mac_shift_grid_array_job.sh` | `download-future-growth`, `download-mac-shift` |
| Gaussian ensemble | `run_ensemble_delayed_array_job.sh` | `download-ensemble` |

The separate `run_mean_parameter_array_job.sh` checks the baseline parameter
vector. SI learning-measure and cost-floor cases are specified by
`scripts/reproduce_si.py` and saved under `data/si_sensitivities/`.

## Verify and regenerate figures

Create the environment from `environment.yaml`, then run from this directory:

```bash
conda env create -f environment.yaml
conda activate econ
python scripts/verify_package.py
python scripts/render_paper.py
```

The figure runner executes the notebook's code cells with this repository as its working directory. It writes regenerated figures, tables, and a render manifest to `data/new_outputs/paper_facing_plots/`. 

For the model's active tests:

```bash
python -m unittest discover -s tests
```

## Recompute model outcomes

To solve the revised full-tree baseline and a ten-year delay locally:

```bash
python -m scripts.run_revised_adjoint
python -m scripts.run_revised_adjoint --fixed-years 10
```

Most results can be generated on a mobile computer and require less than 1 GB of RAM and 1 core. Other results, such as the ensemble, are recommended to be run on a computing cluster to take advantage of the built-in parallelization. For the complete paper suite, inspect `python scripts/reproduce_main_spec.py`; its `--submit-cluster` flag requires the `grid_run` scheduler used for the original SGE arrays, which has to be adapted given the HPC environment. The saved damage grids avoid repeating the climate Monte Carlo step, although this can be disabled. 
