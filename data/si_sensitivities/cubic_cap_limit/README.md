# Smooth cubic removal costs approaching a no-removal constraint

Question: Does the large welfare cost of a hard no-removal constraint (`m <= 1`) emerge as the limit of an increasingly expensive but smooth removal technology? This implements Bob Litterman's proposed check without altering EZDelay source code.

## Model and comparison

The experiment uses the full original delay tree and the current CKI/EZDelay revised specification: RA 10, EIS 1.5, PRTP 0.5%, deterministic growth 1.5% through 2250, post-2250 IID log-growth mean 1.5% and volatility 2%, exogenous technology improvement 1.5%, no endogenous learning, flat SSP2 BAU emissions after 2100, PCHIP damages, and five-year emissions/consumption/technology cost accounting. Both the no-delay optimum and ten-year delay are **reoptimized** for every row. The hard-cap comparator is independently optimized with `m <= 1` and the same ordinary abatement cost curve.

The smooth extension is the existing `CubicBackstopCost`:

```text
MAC_g(m,t) = L_t [58.9 (exp(2.3m)-1) + g max(m-1,0)^3].
```

For `m <= 1`, every smooth case has identical marginal cost. For `m > 1`, the incremental integrated cost is `L_t g(m-1)^4/4` per baseline ton. Therefore any fixed amount of removal becomes arbitrarily expensive as `g` increases; with positive finite-horizon baseline emissions and technology multipliers, the limiting feasible policy has `m <= 1`.

| Cubic coefficient `g` | Ten-year delay DWL, percent of consumption in each delay year | Maximum `m`, optimal | Maximum `m`, delayed | Optimal welfare above hard cap | Delayed welfare above hard cap |
|---:|---:|---:|---:|---:|---:|
| 200,000 | 6.8707 | 1.37719 | 1.43374 | 0.201094 | 3.048534 |
| 2,000,000 | 9.2758 | 1.27450 | 1.42685 | 0.138331 | 2.892106 |
| 20,000,000 | 15.3381 | 1.19795 | 1.30596 | 0.092812 | 2.616457 |
| 2,000,000,000 | 59.7319 | 1.05955 | 1.12216 | 0.035782 | 1.081287 |
| 200,000,000,000 | 87.7627 | 1.01414 | 1.02859 | 0.009806 | 0.260415 |
| 20,000,000,000,000 | 95.1337 | 1.00312 | 1.00624 | 0.002250 | 0.057239 |
| Hard cap `m <= 1` | 97.2401 | 1.00000 | 1.00000 | 0 | 0 |

The primary requested five-step sequence ends at `g=200,000,000,000`. The sixth, higher coefficient checks that the remaining gap keeps shrinking. Welfare is the model's recursive utility index; differences in the last two columns are utility-index units, not consumption percentages. The DWL is the uniform proportional increase in consumption during the ten-year delay needed to match the corresponding no-delay optimum.

The sequence supports convergence of policy and welfare to the hard-cap solution. It does not establish that the no-removal result is empirically plausible. In particular, the 97.24% compensation depends on a world in which removal is effectively impossible even late in the horizon. Nor does a finite sequence prove global optimality in a nonconvex tree problem. Each saved candidate passed positive-consumption, finite-difference-gradient, independent-start, and local feasible-marginal-condition checks; the largest final feasible condition discrepancy in any row is below $1/tCO2. The highest-coefficient candidates required local coordinate polishing after the adjoint L-BFGS-B solve, and both original and polished policies are retained.

The marginal **resource** cost at a node bound by `m=1` need not equal the marginal climate benefit there. A positive cap shadow value explains the gap. As `g` grows, the smooth solution can have `m` arbitrarily close to one while its marginal cost above one approaches that shadow value. Thus policy and welfare should converge to the cap outcome, but the smooth marginal price need not converge to the cap's ordinary resource MAC.

## Reproduction files

- `run_ladder.py` runs the initial five smooth levels for optimal and ten-year delay; `cap1_*` contain independently solved hard-cap comparators.
- The sixth level used the polished `g=200,000,000,000` policy as its start, then the same revised adjoint runner.
- `polish_coordinates.py` corrects residuals at a small number of late nodes while retaining welfare and reports its own feasible-condition check. The source output folders remain intact.
- `summarize.py` evaluates saved policies, verifies that the hard-cap policies have the same utility under every smooth curve, solves the delay-window consumption-equivalent compensation, and writes `summary.csv`.
- `plot_convergence.py` generates `convergence.png` from that CSV.

The CKI/EZDelay source and damage simulation are external to this record. `manifest.json`, `policy.npz`, gradient audits, and `interval_nodes.csv` are stored separately for every solve in this directory.
