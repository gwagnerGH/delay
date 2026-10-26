# Checked inputs for the revised EZDelay default

These are copies of the already validated simulated damage grids and
converged warm-start policies from the EZC-growth research records of
September 26–27, 2026. They make the default CKI/EZDelay run independent of
another repository at runtime. The `.npz` files are inputs, not newly
estimated calibrations; the solver reoptimizes from each starting policy.

| Local file | Source research record | SHA-256 |
|---|---|---|
| `full_damages.npz` | `full_tree_smooth_pchip_flat_2026-09-26/damages.npz` | `3434c3439ee0e8fb66caa7cb892cf772b7b8d6b2859bfaaf96234d8bba19c46d` |
| `full_start.npz` | `cki_revised_adjoint_full_final_2026-09-27/policy.npz` | `bd9f0a9cd0308c29077054326a6167576708aebec693d522bb92d58c74f80927` |
| `short_damages.npz` | `flat_short_nolearning_premia_2026-09-25/damages.npz` | `c45700087708bf7e011f364639116697422ef45c6be8228c0ca5127d97d554f6` |
| `short_start.npz` | `cki_revised_adjoint_short_2026-09-27/policy.npz` | `b9adc64e723c6ea0a31b0d1ad7563d4185fe6ccce26b02cf8da3a5c754c98f8e` |

The full tree has decision times `[0,5,10,15,35,75,125,175,225]`; the
short tree has `[0,10,35,75,125,175,225]`. The code checks tree compatibility
when loading a grid or policy. To change the climate calibration, regenerate
the damage grid explicitly and pass `--damage` and a matching `--start`.

## Warm starts retained from the previous baseline (September 28, 2026)

The default runner uses these policies only as numerical warm starts, with the same saved damage grids. They were solved under the previous RA 15/EIS 1.5 learning specification, so the current RA 10/EIS 2.2/no-learning model always reoptimizes them. Using a warm start is not a substitute for checking the final marginal conditions.

| Local file | Description | SHA-256 |
|---|---|---|
| `full_baseline_start.npz` | Previous RA 15/EIS 1.5 learning baseline; current-model warm start only | `b6b363b039ca862af52b9bb040bc2d7a2deb514d3b3ebbf5b7933d6c4cbbe041` |
| `short_baseline_start.npz` | Previous RA 15/EIS 1.5 learning baseline; current-model warm start only | `7ca74c5dfc69ed51de32d7c2b03528501aaf96debd856c959fe5826002858a2c` |

The quoted certification statistics for these files apply to the previous model specification, not to the current baseline.
