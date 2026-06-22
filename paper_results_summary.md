# Results Summary for the Delay Analysis

Updated for the refreshed deterministic outputs and the `ensemble-BY2025-cons2025-samegrid-run0gauss-eisfix-v1` Gaussian ensemble in `data/new_outputs`. The raw ensemble has 2,998 sample-delay rows: 998 at the 5-year delay and 1,000 at the 10- and 15-year delays. The paper-facing Gaussian figure and summaries use 2,985 valid rows after excluding 13 failed or obviously erroneous sample-delay rows, leaving 995 valid draws at each delay.

Important status update: the annual-frontier DWL values currently in the downloaded output folders use a one-annual-subperiod compensation equivalent. They are therefore not comparable to the five-year first-period DWL used in the mean-parameter, partial-mitigation, and robustness runs. The code now writes a comparable five-year first-period compensation metric (`delta_c_5yr_pct`, `delta_c_5yr_billions`, and `delta_c_5yr_total_billions`). The annual frontier and annual frontier robustness runs should be rerun before citing annual-frontier DWL levels or compensation amounts from this summary.

## Executive Summary

The model starts in 2025, uses the 2025 carbon-cycle and concentration initial conditions, and uses endogenous learning in the baseline specification. The central welfare object is the consumption-equivalent deadweight loss of delay, denoted here as DWL. For paper-facing comparisons, DWL should be reported as the proportional increase in consumption over the first five calendar years required to make the delayed-policy path as valuable as the unconstrained policy path. On the standard five-year grid this equals the old first-period object; on the annual delay frontier it is the new `delta_c_5yr_pct` object.

The refreshed annual delay frontier is substantially steeper than the archived v1 frontier. In the current `samegrid-run0-v3` output, a one-year delay implies a DWL of about 4.2 percent of first-period consumption. A five-year delay raises this to about 32.4 percent, a ten-year delay to 82.6 percent, a fifteen-year delay to 157.8 percent, and a twenty-year delay to 286.6 percent. The corresponding first-period compensation rises from roughly $2.5 trillion after one year to about $171.6 trillion after twenty years.

The emissions channel remains intuitive and large. A five-year delay adds about 164 GtCO2 by policy re-entry; a ten-year delay adds about 326 GtCO2; a fifteen-year delay adds about 499 GtCO2; and a twenty-year delay adds about 671 GtCO2. Atmospheric CO2 at re-entry rises from 425.6 ppm at the start to about 447.8 ppm after five years, 465.5 ppm after ten years, 481.5 ppm after fifteen years, and 496.3 ppm after twenty years. The model temperature anomaly rises from about 1.42 C to about 1.87 C by twenty-year re-entry.

The marginal value of avoiding one more year of delay is also large. In the baseline annual frontier, the first year of delay costs about 4.2 percentage points of first-period consumption. The fifth, tenth, fifteenth, and twentieth years add about 8.1, 13.0, 18.4, and 33.1 percentage points, respectively. These are discrete differences on the annual frontier, not smooth derivatives.

Partial early mitigation still sharply reduces welfare losses. In the refreshed partial-mitigation run, allowing 50 percent early mitigation avoids about 71 percent of the five-year no-action DWL, 76 percent of the ten-year no-action DWL, and 78 percent of the fifteen-year no-action DWL. Allowing 75 percent early mitigation avoids roughly 93 to 97 percent of DWL across the tested delays. This is one of the clearest policy-facing results: the cost of delay depends strongly on whether delay means complete inaction or constrained but meaningful early action.

Robustness exercises preserve the qualitative result that delay is costly, but the magnitude varies enormously. Tree structure and preference parameters produce the largest variation. Technology assumptions and damage specifications also matter, but their central message is more stable: stronger endogenous learning makes delay more costly because delayed deployment also delays learning; more exogenous technological progress makes delay less costly because future mitigation becomes cheaper.

The `eisfix-v1` Gaussian ensemble removes the previous EIS-near-one numerical failure and leaves a strongly right-skewed uncertainty distribution after filtering failed rows. The median paper-facing DWL is zero at the 5-, 10-, and 15-year delays, while the means are 13.4 percent, 24.5 percent, and 61.0 percent. The 5th-to-95th percentile ranges are 0.0-46.9 percent, 0.0-91.1 percent, and 0.0-162.3 percent. The probability that DWL exceeds 10 percent rises from 15.3 percent at five years to 21.3 percent at ten years and 27.9 percent at fifteen years.

Preference parameters remain the clearest drivers within the filtered Gaussian ensemble, but the rank correlations are much weaker than in the previous run because many valid draws have zero paper-facing DWL. Across the three delays, the Spearman correlation between DWL and PRTP is about -0.16 to -0.20, while the correlation with EIS is about +0.15 to +0.16. Correlations with risk aversion, technology, backstop cost, and growth remain smaller in magnitude. These are prior-uncertainty results from the specified truncated Gaussian sampling distributions, not posterior estimates informed by data.

There are two important interpretation cautions. First, the zero-delay row should be treated as an economic normalization, not as a mechanism observation. The raw zero-delay comparison has a small numerical DWL and an anomalous re-entry-price comparison. Second, the `High EIS (1.86)` annual robustness run in the refreshed output is numerically degenerate, with near-zero DWL throughout and extreme price-comparison values. It should not be used as a substantive paper line unless rerun or diagnosed. The paper-facing annual frontier panel currently uses baseline, low EIS, high RA, and no endogenous learning.

## Output Status

| Run | Rows |
| --- | --- |
| mean | 3 |
| delay_frontier | 21 |
| delay_frontier_robustness | 105 |
| partial | 63 |
| tree | 15 |
| preference | 81 |
| technology | 27 |
| damage | 24 |
| Gaussian ensemble (`cons2025`, raw `eisfix-v1`) | 2,998 |
| Gaussian ensemble (`cons2025`, paper-facing valid) | 2,985 |


The raw `eisfix-v1` Gaussian ensemble is missing two 5-year tasks, sample 112 task 337 and sample 824 task 2473. The paper-facing notebook additionally excludes 13 failed sample-delay rows with non-finite utilities, negative compensation artifacts, or implausible re-entry price jumps. After this filter there are 995 valid draws at each delay. The uncertainty bands and parameter-interaction results should be described as Gaussian prior-uncertainty or ensemble results, not as posterior estimates.

## Welfare Metric

The consumption-equivalent DWL solves

```text
U_0^* = [ (1 - beta) ((1 + phi)c_0^delay)^rho
          + beta (CE_1^delay)^rho ]^(1/rho).
```

For `rho != 0`,

```text
phi = [ ((U_0^*)^rho - beta (CE_1^delay)^rho)
        / ((1 - beta)(c_0^delay)^rho) ]^(1/rho) - 1.
```

A DWL value of 10 means that the representative agent would need a 10 percent increase in first-period consumption to be indifferent between the delayed path and the unconstrained path. In partial-mitigation figures, remaining losses are shown using the nonnegative convention `max(DWL, 0)`, and avoided-loss shares are capped at 100 percent. Raw negative values at high allowed mitigation mean the constrained path more than offsets the comparison benchmark; they should not be described as negative delay costs.

## Parameter Priors for the Gaussian Ensemble

The Gaussian ensemble now uses truncated Gaussian distributions whose peaks, or raw normal locations, equal the main run-0 calibration. These are structured uncertainty distributions, not posterior estimates.

| Parameter | Lower | Upper | Mode | Std. dev. |
| --- | --- | --- | --- | --- |
| RA | 3 | 15 | 10 | 3 |
| EIS | 0.55 | 1.86 | 0.833 | 0.3275 |
| tech_chg | 0 | 3 | 1.5 | 0.75 |
| tech_scale | 0 | 3 | 1.5 | 0.75 |
| PRTP | 0.001 | 0.024 | 0.002 | 0.00575 |
| bs_premium | 5000 | 20000 | 10000 | 3750 |
| growth | 0.01 | 0.025 | 0.015 | 0.00375 |


## Gaussian Ensemble Results (`cons2025`, `eisfix-v1`)

The `eisfix-v1` ensemble combines 1,000 truncated-Gaussian parameter draws with the 5-, 10-, and 15-year delay experiments. The raw output has 2,998 rows. The paper-facing convention uses nonnegative five-year consumption-equivalent DWL, `max(delta_c_5yr_pct, 0)`, after excluding failed rows according to the notebook filter.

| Delay | Valid draws | Median DWL | IQR | 5th-95th percentile | Mean DWL | P(DWL > 10%) | P(DWL > 50%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 995 | 0.0% | 0.0-4.4% | 0.0-46.9% | 13.4% | 15.3% | 4.7% |
| 10 | 995 | 0.0% | 0.0-7.7% | 0.0-91.1% | 24.5% | 21.3% | 8.0% |
| 15 | 995 | 0.0% | 0.0-12.3% | 0.0-162.3% | 61.0% | 27.9% | 10.4% |

The mean is far above the median because many draws have zero paper-facing DWL while a smaller set of high-DWL draws produces a long upper tail. Quantile bands and upper-tail probabilities are therefore more informative than the mean alone. The upper tail expands with delay: the probability of DWL above 100 percent is 2.3 percent at five years, 4.7 percent at ten years, and 6.6 percent at fifteen years. The probability of DWL above 250 percent is 1.2 percent, 2.0 percent, and 3.4 percent, respectively.

Median extra cumulative emissions at re-entry are about 176 GtCO2 after five years, 335 GtCO2 after ten years, and 500 GtCO2 after fifteen years. Median re-entry carbon-price increases are $3.8/tCO2, $10.3/tCO2, and $14.7/tCO2, respectively. The price-increase IQRs are $1.1-$9.0/tCO2, $3.7-$19.1/tCO2, and $4.9-$28.1/tCO2.

The ensemble price means should not be reported from the raw output. A small number of failed rows have implausibly large positive or negative price differences. The paper-facing notebook excludes those rows before constructing the price summaries and price-path fan.

The preference interaction is still economically large among positive-DWL ten-year-delay draws. Across EIS quintiles, median DWL rises from 5.9 percent to 222.3 percent within the low-PRTP tercile. Within the high-PRTP tercile, it rises from 0.8 percent to 8.7 percent. This supports presenting EIS and PRTP jointly rather than interpreting either parameter in isolation.

## Central Annual Delay Frontier

The annual delay frontier is the headline object for integer delay lengths from zero to twenty years. It asks: what is the welfare cost of postponing unconstrained mitigation for `d` years?

| Delay | DWL (%) | First-period compensation | Deadweight cost | Extra emissions (GtCO2) | Re-entry price increase | CO2 (ppm) | Temp. (C) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0 (raw 0.043) | $0.0 tn | $0 | 0 | not used | 425.6 | 1.42 |
| 1 | 4.2 | $2.5 tn | $87 | 29.0 | $1.6 | 431.0 | 1.44 |
| 5 | 32.4 | $19.4 tn | $601 | 163.9 | $5.2 | 447.8 | 1.53 |
| 10 | 82.6 | $49.5 tn | $1,559 | 326.0 | $11.5 | 465.5 | 1.64 |
| 15 | 157.8 | $94.5 tn | $2,936 | 499.2 | $19.3 | 481.5 | 1.76 |
| 20 | 286.6 | $171.6 tn | $5,306 | 670.9 | $27.2 | 496.3 | 1.87 |


Main interpretation: the refreshed annual frontier is steep and convex in the policy-relevant range. The twenty-year delay requires about $171.6 trillion in first-period compensation and has a deadweight cost above $5,300 per tCO2. The scale is much larger than the archived v1 frontier, and this difference comes from the refreshed output data rather than a plotting multiplier.

The re-entry price increase should be read as a mechanism signal, not as the welfare loss itself. The welfare loss includes the entire period in which optimal policy is unavailable, the forgone early mitigation, the lost learning, and the worse climate state at re-entry.

## Annual Frontier Robustness

The annual robustness frontier uses the same integer delay lengths as the baseline annual frontier. Selected DWL values are:

| Specification | 0 years | 1 year | 5 years | 10 years | 15 years | 20 years |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline annual frontier | 0.0 | 4.2 | 32.4 | 82.6 | 157.8 | 286.6 |
| Low EIS (0.55) | <0.01 | 0.6 | 2.7 | 5.7 | 9.1 | 12.8 |
| High EIS (1.86) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |
| Low RA (3) | <0.01 | 3.2 | 25.3 | 62.5 | 112.5 | 185.0 |
| High RA (15) | 0.1 | 4.4 | 37.8 | 104.3 | 209.6 | 403.7 |
| No endogenous learning | 0.0 | 7.9 | 28.5 | 64.4 | 116.0 | 196.8 |


High risk aversion greatly amplifies delay costs: the twenty-year DWL rises to about 403.7 percent. No endogenous learning lowers the frontier relative to the baseline at long horizons, with a twenty-year DWL of about 196.8 percent rather than 286.6 percent. This confirms that endogenous learning is an important channel through which early action has value.

Low EIS produces much smaller DWLs. It should be treated as a preference stress test: low intertemporal substitutability changes how utility losses translate into first-period compensation. The high-EIS annual robustness output is currently degenerate and should be excluded from paper claims unless repaired.

## Marginal Value of One Year Earlier Action

The marginal value table reports discrete differences in DWL between delay `d` and delay `d - 1`.

| Specification | Delay year reached | Marginal DWL | Marginal compensation |
| --- | --- | --- | --- |
| Baseline annual frontier | 1 | 4.2 | $2.5 tn |
| Baseline annual frontier | 5 | 8.1 | $4.8 tn |
| Baseline annual frontier | 10 | 13.0 | $7.8 tn |
| Baseline annual frontier | 15 | 18.4 | $11.0 tn |
| Baseline annual frontier | 20 | 33.1 | $19.8 tn |
| Low EIS (0.55) | 1 | 0.5 | $0.3 tn |
| Low EIS (0.55) | 5 | 0.6 | $0.3 tn |
| Low EIS (0.55) | 10 | 0.6 | $0.4 tn |
| Low EIS (0.55) | 15 | 0.8 | $0.5 tn |
| Low EIS (0.55) | 20 | 0.7 | $0.4 tn |
| High RA (15) | 1 | 4.3 | $2.6 tn |
| High RA (15) | 5 | 9.6 | $5.7 tn |
| High RA (15) | 10 | 17.2 | $10.3 tn |
| High RA (15) | 15 | 24.4 | $14.6 tn |
| High RA (15) | 20 | 46.9 | $28.1 tn |
| No endogenous learning | 1 | 7.9 | $4.7 tn |
| No endogenous learning | 5 | 5.9 | $3.5 tn |
| No endogenous learning | 10 | 8.4 | $5.1 tn |
| No endogenous learning | 15 | 11.3 | $6.8 tn |
| No endogenous learning | 20 | 19.6 | $11.8 tn |


The first year is a boundary movement from the no-delay optimum to a constrained delayed path. After that, the increments measure additional years of waiting conditional on already being delayed. In the refreshed output, the marginal cost generally grows with the delay window and becomes very large near twenty years.

## Five-Year Mean-Parameter Benchmark

The mean-parameter benchmark is the coarser-grid comparison at 5-, 10-, and 15-year delays. It is useful because most robustness grids use this same common spacing.

| Delay | DWL (%) | First-period compensation | Deadweight cost | Extra emissions (GtCO2) | Re-entry price increase |
| --- | --- | --- | --- | --- | --- |
| 5 | 14.2 | $8.5 tn | $211 | 205.0 | $8.4 |
| 10 | 25.7 | $15.4 tn | $404 | 391.0 | $12.7 |
| 15 | 39.7 | $23.8 tn | $635 | 581.2 | $26.7 |


The coarser benchmark is not numerically identical to the annual frontier, but it gives the common reference for the tree, preference, technology, and damage robustness blocks. In the refreshed output, the five-year mean-parameter DWL is about 14.2 percent; the fifteen-year value is about 39.7 percent.

## Partial Early Mitigation Frontier

The partial-mitigation experiment asks how much welfare loss remains when policy delay does not mean complete inaction. The early mitigation variable is an upper bound on first-period mitigation during the delay window.

| Delay | Allowed early mitigation | Remaining DWL (%) | DWL avoided | Utility loss avoided | Re-entry price increase |
| --- | --- | --- | --- | --- | --- |
| 5 | 0% | 14.0 | 0% | 0% | $15.3 |
| 5 | 25% | 8.5 | 39% | 37% | $9.9 |
| 5 | 50% | 4.0 | 71% | 69% | $10.9 |
| 5 | 75% | 1.0 | 93% | 92% | $7.3 |
| 5 | 100% | 0.0 | 100% | 100% | $-2.4 |
| 10 | 0% | 25.9 | 0% | 0% | $21.8 |
| 10 | 25% | 14.8 | 43% | 39% | $17.6 |
| 10 | 50% | 6.1 | 76% | 74% | $12.1 |
| 10 | 75% | 1.2 | 95% | 95% | $-3.9 |
| 10 | 100% | 0.1 | 100% | 100% | $2.4 |
| 15 | 0% | 39.5 | 0% | 0% | $21.8 |
| 15 | 25% | 21.3 | 46% | 41% | $24.3 |
| 15 | 50% | 8.8 | 78% | 74% | $11.8 |
| 15 | 75% | 1.4 | 97% | 96% | $4.9 |
| 15 | 100% | 0.0 | 100% | 100% | $1.5 |


Policy implication: partial action is highly valuable. A 50 percent early-mitigation cap avoids most of the no-action welfare loss in all three tested delay cases. A 75 percent cap nearly eliminates the welfare loss. The re-entry price gap generally declines as more early mitigation is allowed, though individual points can be nonmonotone because the model reoptimizes over a nonlinear stochastic tree.

At 100 percent allowed early mitigation, raw DWL can be slightly negative in the five- and fifteen-year cases. The plotted and paper-facing convention floors remaining loss at zero and caps avoided shares at 100 percent.

## Tree-Structure Robustness

The tree-structure run changes decision timing and fragility probability weighting while comparing delayed and baseline cases on aligned grids.

| Delay | Tree specification | DWL (%) | Deadweight cost | Extra emissions (GtCO2) | Re-entry price increase |
| --- | --- | --- | --- | --- | --- |
| 5 | Back-loaded | 13.1 | $198 | 200.8 | $11.8 |
| 5 | Baseline tree | 14.3 | $221 | 197.3 | $12.4 |
| 5 | Front-loaded | 14.4 | $215 | 202.9 | $19.3 |
| 5 | p=0.75 (more fragile) | 49.1 | $701 | 493.3 | $5.8 |
| 5 | p=1.5 (less fragile) | 4.4 | $89 | 41.3 | $0.8 |
| 10 | Back-loaded | 24.4 | $391 | 384.0 | $19.7 |
| 10 | Baseline tree | 25.8 | $410 | 387.4 | $21.0 |
| 10 | Front-loaded | 25.1 | $391 | 395.6 | $22.5 |
| 10 | p=0.75 (more fragile) | 105.5 | $1,512 | 993.1 | $45.7 |
| 10 | p=1.5 (less fragile) | 7.5 | $163 | 77.7 | $1.6 |
| 15 | Back-loaded | 37.2 | $611 | 565.6 | $23.9 |
| 15 | Baseline tree | 39.5 | $635 | 577.4 | $27.4 |
| 15 | Front-loaded | 38.3 | $615 | 579.1 | $34.2 |
| 15 | p=0.75 (more fragile) | 195.3 | $2,781 | 1508.9 | $83.7 |
| 15 | p=1.5 (less fragile) | 10.9 | $238 | 116.6 | $3.2 |


The baseline, front-loaded, and back-loaded decision-time structures all imply large positive delay costs. The probability-scale experiments matter even more. In this output, `p=0.75` is the more fragile specification and produces much larger DWLs; `p=1.5` is less fragile and produces much lower DWLs. This is now clearer than in the older summary: higher fragility raises, rather than lowers, delay costs in the refreshed run.

## Technology Robustness

Technology robustness varies exogenous technological change and endogenous learning. The table reports the fifteen-year delay case.

| Exog. tech | Learning scale | DWL (%) | Deadweight cost | Optimal first carbon price | Utility loss |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0 | 39.0 | $601 | $216.0 | 0.14 |
| 0.0 | 1.5 | 48.7 | $731 | $230.9 | 0.17 |
| 0.0 | 3.0 | 54.8 | $822 | $231.2 | 0.19 |
| 1.5 | 0.0 | 32.9 | $545 | $196.0 | 0.13 |
| 1.5 | 1.5 | 39.3 | $625 | $215.7 | 0.15 |
| 1.5 | 3.0 | 43.7 | $704 | $210.0 | 0.16 |
| 3.0 | 0.0 | 29.1 | $504 | $191.4 | 0.11 |
| 3.0 | 1.5 | 33.8 | $572 | $202.2 | 0.13 |
| 3.0 | 3.0 | 37.5 | $625 | $209.0 | 0.14 |


Two patterns are central. More exogenous technology progress lowers delay costs because mitigation becomes cheaper over calendar time. Stronger endogenous learning raises delay costs because delayed mitigation also delays learning-by-doing. This mechanism is economically important for the paper: learning strengthens the value of early action.

## Damage-Function Robustness

Damage robustness varies the damage-function index and tipping-point switch.

| Delay | Damage specification | DWL (%) | Deadweight cost | Utility loss | Extra emissions (GtCO2) |
| --- | --- | --- | --- | --- | --- |
| 5 | D=0, TP off | 10.9 | $173 | 0.05 | 192.2 |
| 5 | D=0, TP on | 14.3 | $213 | 0.06 | 204.8 |
| 5 | D=1, TP off | 28.8 | $413 | 0.11 | 212.1 |
| 5 | D=1, TP on | 33.4 | $478 | 0.12 | 212.9 |
| 5 | D=2, TP off | 1.1 | $35 | 0.01 | 97.5 |
| 5 | D=2, TP on | 3.3 | $73 | 0.02 | 136.2 |
| 5 | D=3, TP off | 3.6 | $78 | 0.02 | 140.0 |
| 5 | D=3, TP on | 5.9 | $111 | 0.03 | 163.6 |
| 10 | D=0, TP off | 19.4 | $323 | 0.08 | 370.7 |
| 10 | D=0, TP on | 26.0 | $408 | 0.10 | 392.0 |
| 10 | D=1, TP off | 55.6 | $795 | 0.19 | 430.3 |
| 10 | D=1, TP on | 65.3 | $930 | 0.21 | 432.0 |
| 10 | D=2, TP off | 1.8 | $60 | 0.01 | 180.4 |
| 10 | D=2, TP on | 5.4 | $130 | 0.02 | 258.3 |
| 10 | D=3, TP off | 6.0 | $140 | 0.03 | 265.6 |
| 10 | D=3, TP on | 10.2 | $203 | 0.05 | 307.9 |
| 15 | D=0, TP off | 29.1 | $502 | 0.11 | 537.3 |
| 15 | D=0, TP on | 39.4 | $621 | 0.15 | 589.2 |
| 15 | D=1, TP off | 89.1 | $1,274 | 0.26 | 649.3 |
| 15 | D=1, TP on | 109.6 | $1,560 | 0.30 | 652.4 |
| 15 | D=2, TP off | 2.4 | $88 | 0.01 | 259.3 |
| 15 | D=2, TP on | 7.7 | $190 | 0.03 | 379.4 |
| 15 | D=3, TP off | 8.7 | $206 | 0.04 | 390.5 |
| 15 | D=3, TP on | 15.0 | $306 | 0.06 | 454.1 |


Delay is costly under every tested damage specification, but the level varies sharply. Damage function 1 with tipping points gives the largest losses; damage function 2 gives the smallest. The main-paper message should be that the sign of the delay cost is robust, while its scale is sensitive to the damage function and tipping assumptions.

## Preference Robustness

Preference robustness varies risk aversion, EIS, and the pure rate of time preference. The table first reports the central risk-aversion and EIS combination while varying PRTP.

| Delay | PRTP | DWL (%) | Deadweight cost | Utility loss | Extra emissions (GtCO2) | Re-entry price increase |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.001 | 27.8 | $401 | 0.25 | 211.1 | $11.6 |
| 5 | 0.002 | 14.0 | $210 | 0.06 | 203.5 | $18.5 |
| 5 | 0.024 | 0.1 | $8 | 0.00 | 36.0 | $0.1 |
| 10 | 0.001 | 55.3 | $794 | 0.45 | 428.3 | $29.5 |
| 10 | 0.002 | 25.7 | $404 | 0.10 | 391.2 | $21.5 |
| 10 | 0.024 | 0.1 | $14 | 0.00 | 66.8 | $0.2 |
| 15 | 0.001 | 90.4 | $1,294 | 0.64 | 648.4 | $41.9 |
| 15 | 0.002 | 39.3 | $626 | 0.15 | 582.1 | $27.7 |
| 15 | 0.024 | 0.2 | $20 | 0.00 | 101.4 | $0.1 |


The discount-rate channel is extremely strong. Holding RA and EIS at their central values, moving PRTP from 0.2 percent to 2.4 percent nearly eliminates the consumption-equivalent DWL. Moving PRTP from 0.2 percent to 0.1 percent roughly doubles or triples the central DWL, depending on delay length.

The most extreme preference-grid cases are much larger:

| Delay | Max-DWL preference case | DWL (%) | Deadweight cost |
| --- | --- | --- | --- |
| 5 | RA=15, EIS=1.86, PRTP=0.002 | 1057.7 | $10,048 |
| 10 | RA=15, EIS=1.86, PRTP=0.002 | 2396.6 | $22,740 |
| 15 | RA=15, EIS=1.86, PRTP=0.002 | 4348.1 | $41,258 |


These preference-grid extremes should be described as stress tests, not central estimates. They are useful because they show how sensitive consumption-equivalent welfare statements are to deep preference parameters.

## Robustness Ranges

| Run family | Delay | Min DWL (%) | Max DWL (%) | Mean DWL (%) |
| --- | --- | --- | --- | --- |
| damage | 5 | 1.1 | 33.4 | 12.7 |
| damage | 10 | 1.8 | 65.3 | 23.7 |
| damage | 15 | 2.4 | 109.6 | 37.6 |
| preference | 5 | 0.0 | 1057.7 | 90.7 |
| preference | 10 | 0.0 | 2396.6 | 230.6 |
| preference | 15 | 0.0 | 4348.1 | 433.5 |
| technology | 5 | 11.8 | 17.3 | 14.3 |
| technology | 10 | 19.4 | 34.3 | 26.0 |
| technology | 15 | 29.1 | 54.8 | 39.9 |
| tree | 5 | 4.4 | 49.1 | 19.1 |
| tree | 10 | 7.5 | 105.5 | 37.7 |
| tree | 15 | 10.9 | 195.3 | 64.3 |


The preference range dominates the scale and should generally be plotted separately or transformed. Otherwise, the tree, technology, and damage ranges become visually unreadable.

## Paper-Facing Figures

The regenerated paper-facing plots are:

| Figure | Purpose |
| --- | --- |
| annual_delay_frontier_four_panel.pdf / `.eps` | DWL with compensation axis, marginal one-year DWL, emissions, and re-entry price by delay length. |
| policy_reentry_paths_first_panel.pdf / `.eps` | Expected carbon-price paths over the full decision horizon. |
| partial_mitigation_four_panel.pdf / `.eps` | Remaining DWL, avoided losses, and re-entry price under partial early mitigation. |
| uncertainty_main_text_panel.pdf / `.eps` | Gaussian input distributions, DWL bands, upper-tail risk, price effects, price paths, and the EIS-PRTP interaction. |
| robustness_ranges_four_panel.pdf / `.eps` | Structured deterministic robustness results for the supplementary information. |
| appendix_carbon_price_trees_optimal_delayed.pdf / `.eps` | Carbon-price trees for the supplementary information. |


The Gaussian uncertainty figure is now supported by the filtered `cons2025` `eisfix-v1` ensemble. Its bands summarize the specified parameter uncertainty distribution and should not be labeled as posterior credible intervals.

## Paper Claims Supported Now

1. Delay is costly in consumption-equivalent welfare terms, and the cost rises rapidly with the length of delay.
2. The refreshed annual frontier is much steeper than the archived v1 output; the paper must use the updated numbers consistently.
3. The emissions mechanism is direct: delayed mitigation accumulates extra emissions, raises concentration and temperature at re-entry, and worsens the state in which policy resumes.
4. Carbon-price gaps at re-entry are useful mechanism indicators but do not fully measure the welfare cost of delay.
5. Partial early mitigation avoids a large share of the welfare cost, so delay is not an all-or-nothing policy concept.
6. Endogenous learning raises the value of early mitigation because delayed deployment also delays cost reductions from learning-by-doing.
7. Tree structure, fragility weighting, damage functions, and technology assumptions change levels but do not erase the cost of delay.
8. Preference assumptions can dominate consumption-equivalent welfare magnitudes. Preference stress tests should be presented separately from central robustness ranges.
9. The filtered Gaussian ensemble shows a right-skewed distribution of delay costs with a zero median and an upper tail that expands with delay.
10. PRTP and EIS remain the clearest preference drivers in the ensemble and should be presented jointly when discussing preference uncertainty.

## Caveats Before Writing Final Paper Text

1. Do not use the archived v1 annual-frontier numbers in prose. The current `samegrid-run0-v3` annual frontier is much larger.
2. Treat the zero-delay row as a normalization. Do not interpret its raw price gap as a substantive re-entry-price effect.
3. Exclude or repair the degenerate high-EIS annual robustness run before making claims about high EIS in the annual frontier.
4. Keep annual-frontier results separate from coarser five-year robustness results unless the text explicitly notes the different grid.
5. Describe Gaussian bands as prior-uncertainty or ensemble intervals, not posterior intervals.
6. Use median and quantile summaries for ensemble re-entry-price effects; extreme numerical price outliers in failed rows make the raw mean unreliable.
7. Note that the raw `eisfix-v1` Gaussian run is missing two 5-year tasks and that the paper-facing notebook excludes 13 failed sample-delay rows, leaving 995 valid draws at each delay.
