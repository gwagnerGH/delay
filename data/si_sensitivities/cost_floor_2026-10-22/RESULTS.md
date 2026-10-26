# Long-run abatement-cost floor sensitivity (2026-10-22)

The revised EZDelay full tree was reoptimized under four non-compressible shares of marginal abatement and removal costs. For each floor, one no-delay policy and separate 5-, 10-, and 15-year delay policies were solved. All other economics use the current baseline: RA 10, EIS 1.5, annual PRTP 0.5%, modeled consumption growth 1.5%, uncertain post-2250 log growth, SSP2 emissions held constant after 2100, smooth S-shaped removal costs, PCHIP damages, five-year cost accounting, and no endogenous learning.

At five-year consumption date `u`, the technology multiplier is the inherited `L_u` before 2030 and `floor + (1-floor)*L_u` from 2030 onward. The 2030 calibration and the shape of the marginal cost curve are unchanged. The floor applies to ordinary abatement and net removal. The 2030 marginal cost at `m=1` is $528.58/tCO2, so a 40% floor implies an asymptotic $211.43/tCO2 at `m=1` (and $235.72/tCO2 in 2200). These shares are illustrative sensitivities, not estimates.

| Floor | 2025 no-delay interval price ($/t) | 10-year re-entry price ($/t) | 5-year DWL (%/year) | 10-year DWL (%/year) | 15-year DWL (%/year) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 188.64 | 241.24 | 3.681 | 4.117 | 4.609 |
| 25% | 218.48 | 286.29 | 4.751 | 5.350 | 6.018 |
| 40% | 233.30 | 308.75 | 5.315 | 6.006 | 6.766 |
| 60% | 250.95 | 335.58 | 6.019 | 6.828 | 7.699 |

Mean 2150 removal under the no-delay policy falls from 9.05 GtCO2/year at zero floor to 5.40 with a 40% floor and 4.31 with a 60% floor. Under a ten-year delay it falls from 10.53 to 6.48 and 5.31 GtCO2/year. The corresponding mean 2200 interval carbon prices in the no-delay policies are $1.36, $53.84, and $113.75/t. In the zero-floor case, 62 of 128 decision nodes in 2200 choose the lower mitigation bound; with the 40% and 60% floors, all 128 are interior.

Validation: the zero-floor fixed-policy objective reproduces the saved baseline welfare exactly. All 16 newly solved policies pass the local stationarity certification, with maximum feasible marginal-condition gap below $1/t, positive consumption, no numerical mitigation guard binding, and maximum independent finite-difference gradient discrepancy below 8.1e-8. The focused model and derivative tests pass (21 tests). These are local numerical checks, not a guarantee of a global optimum. The sensitivity changes future abatement costs only; it does not change terminal growth uncertainty or introduce a separate tipped climate state.

Reproduce with `python scripts/run_cost_floor_sensitivity.py --output data/new_outputs/revised_cluster/cost_floor_2026-10-22` from the EZDelay repository, then `python scripts/plot_cost_floor_sensitivity.py data/new_outputs/revised_cluster/cost_floor_2026-10-22`. The runner skips completed solves and preserves their manifests. `summary.csv` gives the paired welfare and price results; `trajectories.csv` gives probability-weighted means and 10th/90th percentiles. Each policy directory contains node-level prices, climate benefits, emissions, mitigation, removal, consumption, solver diagnostics, and copied model inputs.
