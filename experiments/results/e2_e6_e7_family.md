# E2 / E6 / E7, Distributional family, exposure specification, zero inflation

Three A/B experiments against the production SPF spec in `pipeline/fit_risk_model.py`.
Every number below was produced by code in `experiments/ab/` that was actually run;
nothing is estimated, carried over, or reconstructed from memory. Where a comparison
is inconclusive it is labelled inconclusive.

Nothing under `pipeline/` or `data/` was modified. `pipeline.fit_risk_model.main()` was
never called; only `load_and_join`, `prepare`, `MODES` and `SHARED_PREDICTORS` were imported.

## Commands run

```bash
cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e2_nb_vs_poisson.py
cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e6_exposure_offset_vs_covariate.py
cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e7_nb_vs_zinb.py
cd C:/Users/jfbaa/project-cycle-group && PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e2_e6_e7_merge_results.py
```

Machine-readable output: `experiments/results/e2_e6_e7_results.json`
(plus per-experiment `e2_results.json`, `e6_results.json`, `e7_results.json`).

## Environment

Python 3.13.2 · numpy 2.1.3 · pandas 2.2.3 · scipy 1.15.1 · statsmodels 0.14.6 ·
patsy 1.0.2 · **scikit-learn 1.6.1 (checked, present. So `sklearn.model_selection.KFold`
was used, not the numpy fallback)** · Windows 11.

## Data

`prepare(load_and_join())` → **346 arterial rows**. `bike_total` = 169 events,
`ped_total` = 266, `vehicle_only_total` = 1295. `years_observed == 6` for every row,
so `offset == log(6) == 1.791759469228055` is constant across all 346 rows.

## Cross-cutting method notes

* **CV**: `KFold(n_splits=5, shuffle=True, random_state=0)`. Reported as mean ± SD
  across folds *and* as the pooled out-of-fold metric over all 346 held-out predictions.
* **Design matrices for CV** are built once with patsy on the full frame and then
  row-subset per fold, so every fold and every family sees identical columns. The rare
  levels `legs_cat == 5` (n=13) and `arterial_class == 5` (n=19) would otherwise make a
  fold-local patsy build silently change the design. Only the *column set* crosses the
  fold boundary, never a response value.
* **AIC/BIC** are recomputed by hand as `-2·LL + 2k` and `-2·LL + k·log(n)` with
  `k = len(result.params)` (so the NB dispersion parameter is counted). statsmodels'
  own `.aic`/`.bic` were also recorded and agree with the manual values for these fits.
* **Convergence**: statsmodels' NB default is `bfgs, maxiter=35`, and it **fails to
  converge for all three modes** on the full data (`ConvergenceWarning: Maximum
  Likelihood optimization failed to converge`). Production handles this by retrying
  `method="bfgs", maxiter=200`, which does converge. That is the fit reported here.
  In E6/E7 an early return on the *first* rung reporting `converged=True` proved unsafe
  (see E6 Part B note), so those scripts run every rung and keep the converged rung with
  the **highest log-likelihood**. All attempts are recorded in the JSON.

---

# E2, Negative Binomial vs Poisson

## Why this experiment exists

`pipeline/fit_risk_model.py:363` declares the family correct on the strength of a single
threshold:

```python
verdict = "overdispersed (NB correct)" if alpha > 0.05 else "near-Poisson"
```

A Poisson model was never fit. There is no side-by-side likelihood, no information
criterion comparison, no formal test of H0: α = 0, and no out-of-sample number anywhere
in the repo. This converts the assertion into a measurement.

## Specs compared

Identical formula and identical offset in both arms; only the family differs.

| arm | call |
|---|---|
| (a) NB2, production | `smf.negativebinomial(formula, data=df, offset=df["offset"].values)` |
| (b) Poisson | `smf.poisson(formula, data=df, offset=df["offset"].values)` |

`formula = "{target} ~ is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class)"`
plus `+ log_bike_centrality` (bike) or `+ log_aadt` (ped, vehicle).

## Overdispersion in the raw targets

Poisson assumes variance = mean. It does not hold for any mode.

| mode | mean | variance | variance/mean | zero sites |
|---|---|---|---|---|
| bike | 0.4884 | 1.3578 | **2.780** | 256 / 346 |
| ped | 0.7688 | 1.5928 | **2.072** | 204 / 346 |
| vehicle | 3.7428 | 28.7539 | **7.683** | 89 / 346 |

## In-sample fit

Every fit below converged. NB converged on the production `bfgs, maxiter=200` retry;
Poisson converged on the statsmodels default (`newton, maxiter=35`) for all three modes.

| mode | LL (NB) | LL (Poisson) | AIC NB | AIC Pois | BIC NB | BIC Pois | α̂ (SE) |
|---|---|---|---|---|---|---|---|
| bike | −275.656 | −301.194 | **575.31** | 624.39 | **621.47** | 666.70 | 1.2382 (0.3287) |
| ped | −346.402 | −352.390 | **716.80** | 726.78 | **762.96** | 769.09 | 0.3049 (0.1194) |
| vehicle | −745.202 | −944.485 | **1514.40** | 1910.97 | **1560.56** | 1953.28 | 0.6585 (0.0832) |

k = 12 for NB (11 regression parameters + α), k = 11 for Poisson, n = 346.
NB wins AIC and BIC in all three modes, by 49.1 / 10.0 / 396.6 AIC points respectively.

## Formal test of H0: α = 0

`LR = 2·(LL_NB − LL_Poisson)`.

| mode | LR | naive χ²(1) p | **boundary-corrected p** |
|---|---|---|---|
| bike | 51.075 | 8.892e-13 | **4.446e-13** |
| ped | 11.977 | 5.386e-04 | **2.693e-04** |
| vehicle | 398.566 | 1.130e-88 | **5.651e-89** |

**The difference in one sentence:** α = 0 sits on the boundary of the parameter space,
so under the null the LR statistic follows a 50:50 mixture of χ²(0) (a point mass at
zero) and χ²(1) rather than a plain χ²(1); the naive p-value is therefore exactly twice
the correct one, i.e. conservative. It understates the evidence against Poisson.
Here the correction changes nothing practical: Poisson is rejected overwhelmingly either
way, and the smallest margin (ped, p ≈ 2.7e-4) is nowhere near any decision threshold.

## Predictive calibration, 90% intervals

Poisson: `poisson.ppf(0.05/0.95, mu)`. NB: `nbinom.ppf(0.05/0.95, n=1/α, p=1/(1+α·mu))`,
matching `pipeline/tests/test_calibration.py`.

| mode | NB coverage | NB mean width | Poisson coverage | Pois mean width | width ratio NB/Pois | sites outside NB / outside Pois |
|---|---|---|---|---|---|---|
| bike | 96.5% | 2.13 | 94.8% | 1.64 | 1.300 | 12 / 18 |
| ped | 97.1% | 2.58 | 96.5% | 2.26 | 1.143 | 10 / 12 |
| vehicle | **94.8%** | 10.62 | **76.6%** | 5.65 | **1.880** | 18 / **81** |

Out-of-fold coverage (fold-specific parameters, held-out rows) tells the same story:
bike NB 96.2% vs Poisson 94.8%; ped NB 96.0% vs 94.8%; **vehicle NB 93.6% vs Poisson 74.9%**.

**Quantifying "Poisson's intervals are too narrow":** the claim holds decisively for the
vehicle mode and only weakly for the two sparse modes. For vehicle, Poisson intervals are
**47% narrower** than NB's (5.65 vs 10.62 counts) and miss the nominal 90% target in the
wrong direction by **13.4 percentage points** (76.6% vs 90%), 81 of 346 sites fall outside
a nominally-90% interval instead of the expected ~35. For bike and ped, both families
*over*-cover (94.8 to 97.1% against a 90% target) because the counts are so small that
integer-valued interval endpoints are coarse; Poisson is narrower but not yet
mis-calibrated at these means.

## Out-of-sample point accuracy (5-fold CV). The null result

All 5 folds converged for both families in all three modes.

| mode | family | CV MAE (mean ± SD) | CV RMSE (mean ± SD) | pooled OOF MAE | pooled OOF RMSE |
|---|---|---|---|---|---|
| bike | NB | 0.5945 ± 0.1067 | 1.0498 ± 0.3134 | 0.5946 | 1.0866 |
| bike | Poisson | 0.5910 ± 0.1151 | 1.0429 ± 0.3154 | 0.5911 | 1.0804 |
| ped | NB | 0.7269 ± 0.0685 | 1.0968 ± 0.1386 | 0.7271 | 1.1044 |
| ped | Poisson | 0.7243 ± 0.0707 | 1.0933 ± 0.1443 | 0.7245 | 1.1015 |
| vehicle | NB | 2.8867 ± 0.2225 | 4.4548 ± 0.4145 | 2.8861 | 4.4717 |
| vehicle | Poisson | 2.8955 ± 0.1541 | 4.4796 ± 0.3529 | 2.8949 | 4.4924 |

Paired per-fold MAE difference (NB − Poisson): bike **+0.0035 ± 0.0099** (paired t p = 0.478),
ped **+0.0026 ± 0.0043** (p = 0.256), vehicle **−0.0087 ± 0.0837** (p = 0.827).

**This is squarely within noise, and the sign is not even consistent**, Poisson is
trivially *better* on CV MAE for bike and ped, NB trivially better for vehicle, and no
difference approaches significance. This is expected rather than surprising: NB2 and
Poisson share the same log-link mean function, and α affects the *variance*, not the
fitted mean, so a mean-error metric like MAE/RMSE is close to blind to the family choice.

## Verdict, E2

**NB is decisively the right family, but for variance reasons only, and the repo's
stated justification is thinner than the real evidence.** The boundary-corrected LR test
rejects Poisson at p = 4.4e-13 (bike), 2.7e-4 (ped) and 5.7e-89 (vehicle); NB wins AIC and
BIC in every mode; and for the vehicle model Poisson's 90% intervals cover only 76.6%
in-sample and 74.9% out-of-fold. **However, out-of-sample point accuracy is identical
between the two families to within noise (paired MAE differences of ±0.009 with p ≥ 0.26)
so anything downstream that consumes only the point prediction would be unaffected by
this choice, and only the uncertainty-consuming parts (predictive intervals, the
Empirical-Bayes weighting in `score_risk.py`) actually depend on it.**

---

# E6, Exposure as `offset = log(years_observed)` vs as a free covariate

## Why this experiment exists

An audit flagged this comparison as untestable on the real data because
`years_observed == 6` for all 346 rows, making a free exposure covariate perfectly
collinear with the intercept. The goal was to (i) *demonstrate* that non-identifiability
rather than assert it, and (ii) run an actual test on data where exposure varies.

## Part A, non-identifiability on the real data, demonstrated

`df["years_observed"].unique()` → **`[6]`**. `df["offset"].unique()` → `[1.791759469228055]`.

Six degenerate specs were built (three modes × {`years_observed`, `np.log(years_observed)`}),
each **with an intercept and with the offset deliberately omitted**, e.g.

```
bike_total ~ years_observed + is_signalized + C(legs_cat, Treatment(reference=4))
             + max_speed_limit + bike_facility + C(arterial_class) + log_bike_centrality
```

### Design-matrix rank

`numpy.linalg.matrix_rank` on the patsy design matrix, all six specs:

| spec | design columns | rank | rank deficient | condition number | smallest singular value |
|---|---|---|---|---|---|
| bike :: years_observed | 12 | **11** | yes | 1.54e+16 | 3.26e-14 |
| bike :: log(years_observed) | 12 | **11** | yes | 7.19e+16 | 6.82e-15 |
| ped :: years_observed | 12 | **11** | yes | 1.61e+16 | 3.26e-14 |
| ped :: log(years_observed) | 12 | **11** | yes | 7.74e+16 | 6.64e-15 |
| vehicle :: years_observed | 12 | **11** | yes | 1.61e+16 | 3.26e-14 |
| vehicle :: log(years_observed) | 12 | **11** | yes | 7.74e+16 | 6.64e-15 |

Control. The production offset spec on the same 346 rows: **11 columns, rank 11, full rank**
for all three modes.

The exposure column is literally a constant (`years_observed` = 6.0 everywhere,
`np.log(years_observed)` = 1.791759469228055 everywhere), i.e. an exact scalar multiple
of the intercept column.

### What statsmodels actually does. And this is the concerning part

**It does not raise, and it does not drop the term.** In all six specs
`smf.negativebinomial(...).fit(method="bfgs", maxiter=200)` returned a result object with
all 13 parameters present, no exception, and `mle_retvals["converged"] == True`.

The standard errors are where behaviour diverges, **inconsistently**:

| spec | standard errors | `cov_params()` |
|---|---|---|
| bike :: years_observed | **all 13 finite** | finite |
| bike :: log(years_observed) | all 13 NaN | raises `ValueError: need covariance of parameters for computing (unnormalized) covariances` |
| ped :: years_observed | **all 13 finite** | finite |
| ped :: log(years_observed) | **all 13 finite** | finite |
| vehicle :: years_observed | **all 13 finite** | finite |
| vehicle :: log(years_observed) | all 13 NaN | raises `ValueError` (same) |

So in **4 of 6 cases statsmodels silently returns usable-looking standard errors for a
rank-deficient design**, with a converged flag and no warning. There is no reliable
automatic signal that the fit is meaningless.

### The decisive demonstration, likelihood ridge probe

If the parameters were identified, restarting the optimiser at a shifted point along the
collinear direction would return to the same solution. It does not. Restarting at
`(exposure coef + δ, intercept − 6δ)` for δ ∈ {−2, −0.5, 0, +0.5, +2}, bike ::
`years_observed`:

| start shift δ | intercept | exposure coef | intercept + 6·coef | log-likelihood |
|---|---|---|---|---|
| −2.0 | +11.336104 | −1.683097 | **1.2375191897** | −275.65644707887003 |
| −0.5 | +2.336104 | −0.183097 | **1.2375191897** | −275.65644707887003 |
| 0.0 | −0.663896 | +0.316903 | **1.2375191897** | −275.65644707887003 |
| +0.5 | −3.663896 | +0.816903 | **1.2375191897** | −275.65644707887000 |
| +2.0 | −12.663896 | +2.316903 | **1.2375191897** | −275.65644707887003 |

Across all six specs: the exposure coefficient moves over a range of **4.000**, the
intercept over 24.000 (or 7.167 for the log version), while the **log-likelihood spread is
between 0.0 and 1.1e-13** and the spread of the identified combination
`intercept + x·coef` is between **0.0 and 1.8e-15**. The likelihood is exactly flat along
that direction: the optimiser reports whatever point it happens to land on, and the
"estimate" of the exposure coefficient is an artefact of the starting value.

**Conclusion, plainly: the offset-vs-covariate comparison is NOT identifiable on the real
346-row dataset. Any number statsmodels prints for a free exposure coefficient here is
arbitrary. The audit's claim is correct, and now demonstrated rather than asserted.**

## Part B, test on data where exposure genuinely varies

### ⚠ This is a SYNTHETIC EXPOSURE DESIGN

It tests the *modelling assumption* (is the log-exposure coefficient 1?) on a construction
where the true coefficient is 1 by design. **It does not and cannot change what is knowable
from the real, uniform 6-year observation window.** Part A remains the operative fact about
the production dataset.

### Construction (exact)

Source: `data/intermediate/crashes_by_intersection_year.parquet`, 3906 rows = 651
intersections × 6 years (2018 to 2023), columns `intersection_id`, `year`, `crash_count`.
**Inspected first, and it carries only the TOTAL crash count per intersection-year; there
is no mode breakdown in it.** Per-mode 6-year totals come from
`crashes_by_intersection.parquet`.

Verified before use: every site has exactly 6 year-rows; per-site `sum(crash_count) ==
total_crashes` for 651/651 sites; each mode total ≤ `total_crashes` at every site; every
site with `total_crashes == 0` has all three mode totals == 0.

With `rng = numpy.random.default_rng(seed)`, iterating sites in sorted `intersection_id`
order:

1. **Exposure draw.** `k = rng.integers(2, 7)` → 2 to 6 inclusive;
   `S = rng.choice(the 6 years, size=k, replace=False)`. Set `years_observed := k`.
2. **Crash-year allocation.** Because the grid has no mode split, each mode's 6-year total
   `c_m` is allocated to years by `rng.choice(years, size=c_m, replace=True, p=p_y)` where
   `p_y = crash_count_y / total_crashes` is that site's own empirical year distribution.
3. **Thinning.** The synthetic observed count is the number of allocated mode-m crash-years
   falling inside `S`.

Since `S` is drawn independently of the crash years, `E[y_m | site] = mu_m · k/6`, so the
**true log-exposure coefficient in this design is exactly 1.0**. Which is what makes it a
usable test bed. Features are taken unchanged from the 346-row production frame.

**Supplementary arm** (`total_supplementary`): the same window `S` applied to the site's
**real** per-year total crash counts. No allocation step, so the crash-to-year assignment
is real data and only the observation window is synthetic.

### Resulting dataset (seed 0)

346 rows. `years_observed` distribution: 2→75 sites, 3→51, 4→67, 5→76, 6→77.
Targets after thinning: bike 124 (from 169), ped 185 (from 266), vehicle 839 (from 1295),
total 1165 (from 1720).

### Specs compared

| spec | description |
|---|---|
| (a) | `offset = log(years_observed)`, coefficient constrained to exactly 1 |
| (b) | `log_years_observed` as a free covariate, **no** offset |
| (c) | offset **plus** a free `log_years_observed` term, estimates the deviation from 1 |

(b) and (c) are the same model reparametrised, so `coef_b − coef_c` must be exactly 1.0 and
`LL_b − LL_c` exactly 0. Measured: `coef_b − coef_c = 1.000000` in all four arms, and
`|LL_b − LL_c| ≤ 2.3e-13`. The nested check `LL_b ≥ LL_a` passes in all four arms.

> **Optimiser note. A real trap that was hit and fixed.** With an early return on the
> first rung reporting `converged=True`, the bike spec (b) landed on a local optimum with
> α collapsed to 0 and LL = −234.038, *below* the nested spec (a)'s LL = −221.064, which
> is mathematically impossible and would have produced a bogus coefficient of +1.7024 with
> CI [1.0836, 2.3212] (i.e. a spurious rejection of the offset constraint). The scripts now
> run every rung and select the highest converged log-likelihood, with warm starts from the
> nested spec (a) solution. All reported numbers come from the corrected procedure.

### Results (seed 0). All fits converged

| target | spec | LL | AIC | CV MAE (mean ± SD) | coef on log(exposure) | 95% CI |
|---|---|---|---|---|---|---|
| bike | (a) offset | −221.064 | 466.13 | 0.4601 ± 0.0812 | fixed at 1 | n/a |
| bike | (b) free | −219.684 | 465.37 | 0.4525 ± 0.0843 | **+1.6230** | **[+0.8677, +2.3783]** |
| bike | (c) offset+free | −219.684 | 465.37 | 0.4525 ± 0.0843 | +0.6230 (deviation) | [−0.1323, +1.3783] |
| ped | (a) offset | −271.364 | 566.73 | 0.5365 ± 0.0694 | fixed at 1 | n/a |
| ped | (b) free | −271.112 | 568.22 | 0.5350 ± 0.0706 | **+1.1772** | **[+0.6841, +1.6703]** |
| ped | (c) offset+free | −271.112 | 568.22 | 0.5350 ± 0.0706 | +0.1772 (deviation) | [−0.3159, +0.6703] |
| vehicle | (a) offset | −615.528 | 1255.06 | 2.0648 ± 0.2111 | fixed at 1 | n/a |
| vehicle | (b) free | −615.415 | 1256.83 | 2.0720 ± 0.2070 | **+1.0798** | **[+0.7503, +1.4092]** |
| vehicle | (c) offset+free | −615.415 | 1256.83 | 2.0720 ± 0.2070 | +0.0798 (deviation) | [−0.2497, +0.4092] |
| total (real years) | (a) offset | −681.770 | 1387.54 | 2.4091 ± 0.3575 | fixed at 1 | n/a |
| total (real years) | (b) free | −681.662 | 1389.32 | 2.4157 ± 0.3498 | **+1.0662** | **[+0.7869, +1.3454]** |
| total (real years) | (c) offset+free | −681.662 | 1389.32 | 2.4157 ± 0.3498 | +0.0662 (deviation) | [−0.2131, +0.3454] |

### The key question: does the CI contain 1.0?

**Yes, in all four arms.**

| target | coef | z vs 1.0 | CI contains 1.0 | LR test of the offset constraint | p (χ²(1)) |
|---|---|---|---|---|---|
| bike | +1.6230 | +1.617 | **yes** | 2.760 | 0.0966 |
| ped | +1.1772 | +0.704 | **yes** | 0.505 | 0.4773 |
| vehicle | +1.0798 | +0.475 | **yes** | 0.226 | 0.6346 |
| total (real crash-years) | +1.0662 | +0.464 | **yes** | 0.216 | 0.6420 |

AIC prefers the constrained offset spec (a) for ped (566.73 vs 568.22), vehicle
(1255.06 vs 1256.83) and total (1387.54 vs 1389.32). The free parameter does not pay for
itself. For bike, AIC marginally prefers the free spec (465.37 vs 466.13, Δ = 0.76), which
is not a meaningful margin. CV MAE differences between specs are ≤ 0.008 counts everywhere,
i.e. negligible.

### Seed sensitivity (10 independent replicates, seeds 0 to 9)

The whole construction, window and allocation, is regenerated per seed.

| target | mean coef | range | between-seed SD | mean nominal SE | CI covers 1.0 |
|---|---|---|---|---|---|
| bike | +0.8452 | +0.260 … +1.623 | 0.4961 | 0.3522 | **8/10** |
| ped | +0.9443 | +0.431 … +1.470 | 0.2871 | 0.2466 | **9/10** |
| vehicle | +0.9654 | +0.789 … +1.277 | 0.1411 | 0.1754 | **10/10** |
| total (real crash-years) | +0.9405 | +0.713 … +1.290 | 0.1624 | 0.1447 | **8/10** |

The mean estimate is within 0.16 of 1.0 in every arm, and the coverage rate (8 to 10 of 10)
is consistent with a nominal 95% CI at this replicate count (observing ≤8/10 successes
when the true rate is 0.95 has probability ≈ 0.086, so 8/10 is unremarkable).

Caveat worth stating: for bike the between-replicate spread (SD 0.496) **exceeds** the mean
nominal standard error (0.352), i.e. the bike estimate is visibly unstable across
replicates and its reported SE looks optimistic. For ped, vehicle and total the two are
comparable. That instability tracks event count (124 thinned bike events vs 839 vehicle).

## Verdict, E6

**Part A: not identifiable, demonstrated, full stop**. The design matrix is rank 11 of 12
columns for all six degenerate specs, the exposure column is an exact constant multiple of
the intercept, and restarting the optimiser along the collinear direction moves the exposure
coefficient by 4.0 while changing the log-likelihood by ≤ 1.1e-13. The alarming operational
finding is that statsmodels raises nothing, drops nothing, reports `converged=True`, and in
**4 of 6 cases hands back finite, entirely usable-looking standard errors**. So this failure
mode is silent and would not be caught by any convergence check.
**Part B: on a synthetic variable-exposure design the offset constraint is empirically
justified and never rejected**. The 95% CI for the free log-exposure coefficient contains
1.0 in all four arms at seed 0 (bike +1.62 [0.87, 2.38]; ped +1.18 [0.68, 1.67]; vehicle
+1.08 [0.75, 1.41]; real-crash-year total +1.07 [0.79, 1.35]) and in 8 to 10 of 10 seeds, with
AIC preferring the constrained offset in 3 of 4 arms. **The bike arm is the least
informative and is genuinely inconclusive on its own** (CI width 1.51, spread across seeds
exceeding its own SE). It fails to reject 1.0 but would also fail to reject 1.6.
Neither part licenses any claim about the real 6-year window, where the question stays
unanswerable.

---

# E7, Negative Binomial vs Zero-Inflated NB

## Why this experiment exists, premise confirmed, plus a second breakage found

The stated motivation was that `pipeline/evaluate_models.py` contains a zero-inflation check
that has never run because "the script is broken. It imports a name that doesn't exist."

**That premise is correct, and it was verified against the committed code rather than
assumed.** The version at `HEAD` begins:

```python
from pipeline.fit_risk_model import (
    DESIGN_PREDICTORS,   # <- this name does not exist
    MODES, ModeSpec, load_and_join, prepare,
)
```

`DESIGN_PREDICTORS` occurs **0 times** in `pipeline/fit_risk_model.py` (which is itself
unmodified), and loading the `HEAD` version of the module in isolation gives:

```
ImportError: cannot import name 'DESIGN_PREDICTORS' from 'pipeline.fit_risk_model'
```

**A timing note that matters for reproducing this.** Two other agents were working in this
repo concurrently. Partway through this session one of them modified
`pipeline/evaluate_models.py`, dropping the `DESIGN_PREDICTORS` import and rewriting
`_design_matrix` to use `mode.predictors`. That edit is not mine; nothing under `pipeline/`
or `data/` was written by these experiments. It means a checkout of `HEAD` and the current
working tree give **different** failures.

**With the import bug fixed, the script still does not run**. It fails one step later, on a
second and independent breakage. `python -m pipeline.evaluate_models` against the working
tree gives:

```
File "C:\Users\jfbaa\project-cycle-group\pipeline\evaluate_models.py", line 60, in _load_pkl
    return pickle.load(f)
TypeError: StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given
```

It fails unpickling `data/model/nb_v3_bike.pkl`. The stored pickles were written under a
different pandas version than the installed 2.2.3. So `print_zero_prediction_check`
(`evaluate_models.py:164`) is unreachable **both** at `HEAD` (ImportError) and after the
concurrent fix (pickle TypeError), and the zero-inflation check has never produced a number.
The pickle failure also silently disables `pipeline/tests/test_calibration.py`, which loads
the same three pkl files.

Bike crashes are sparse, 169 events over 346 sites, 256 of them zero, so excess zeros are
plausible and were untested.

## API choice (stated, as required)

ZINB was fit with the statsmodels **array API**:
`ZeroInflatedNegativeBinomialP(y, X, exog_infl=None, offset=off, inflation="logit", p=2)`
on a patsy design matrix built from the production formula. `exog_infl=None` gives a
single constant column, i.e. a **constant excess-zero probability**. `p=2` makes the main
part NB2, matching production. Reasons for the array API over the formula API: the
inflation design is explicit, the identical design matrix is reused for the NB comparator
and every CV fold, and warm starts can be passed as plain arrays.

Rungs tried per fit: default (`bfgs, maxiter=35`), `bfgs maxiter=500` (the task-specified
retry), `bfgs maxiter=2000`, `nelder-mead maxiter=20000`, plus four warm-started rungs
seeded from the converged NB parameters with logit intercept ∈ {−3, −1}. The converged rung
with the highest log-likelihood is reported.

## Observed vs NB-implied zeros. The check that never ran

NB-implied expected zeros = `sum((1/α) / ((1/α) + mu))**(1/α)`.

| mode | observed zero sites | NB-implied expected zeros | gap (obs − exp) |
|---|---|---|---|
| bike | 256 / 346 | 256.45 | **−0.45** |
| ped | 204 / 346 | 206.02 | **−2.02** |
| vehicle | 89 / 346 | 90.86 | **−1.86** |

Every gap is **negative**. The NB predicts slightly *more* zeros than were observed, which
is the opposite of excess zeros. To put those gaps on a sampling-noise scale, 4000 datasets
were simulated from each fitted NB (`numpy.random.default_rng(0)`) and the zero count
recounted:

| mode | observed | simulated zeros (mean ± SD) | 90% band | observed inside band |
|---|---|---|---|---|
| bike | 256 | 256.5 ± 7.6 | [244, 269] | **yes** |
| ped | 204 | 205.9 ± 8.0 | [193, 219] | **yes** |
| vehicle | 89 | 91.0 ± 7.4 | [79, 104] | **yes** |

Every observed zero count sits comfortably inside the NB's own predictive band, near the centre.

## NB vs ZINB

All fits converged. ZINB converged on a warm-started Nelder-Mead rung in all three modes.

| mode | LL NB | LL ZINB | ΔLL | AIC NB | AIC ZINB | ΔAIC | BIC NB | BIC ZINB | ΔBIC |
|---|---|---|---|---|---|---|---|---|---|
| bike | −275.656 | −275.656 | +1.05e-09 | 575.31 | 577.31 | **+2.000** | 621.47 | 627.32 | **+5.846** |
| ped | −346.402 | −346.402 | +3.00e-09 | 716.80 | 718.80 | **+2.000** | 762.96 | 768.81 | **+5.846** |
| vehicle | −745.202 | −745.202 | +6.03e-12 | 1514.40 | 1516.40 | **+2.000** | 1560.56 | 1566.41 | **+5.846** |

(Positive Δ = ZINB worse. k = 12 for NB, 13 for ZINB, n = 346.)

**The ZINB collapses exactly onto the NB.** The estimated inflation probability goes to the
boundary in every mode:

| mode | logit inflation intercept | SE | π̂ | ZINB expected zeros |
|---|---|---|---|---|
| bike | −29.641 | 30352.77 | **1.34e-13** | 256.45 |
| ped | −30.635 | **NaN** | **4.96e-14** | 206.02 |
| vehicle | −31.673 | **NaN** | **1.76e-14** | 90.86 |

The huge or NaN standard errors on the inflation intercept are the expected signature of a
parameter pinned at a boundary (π → 0 corresponds to the logit intercept → −∞, which is
not attainable in the interior). The log-likelihood gain is ~1e-9, so ZINB pays 2 AIC points
and 5.85 BIC points for a parameter that buys nothing.

Sanity checks: the explicit formulas used here for the ZINB marginal mean and P(Y=0) were
cross-checked against statsmodels' own `predict(which="mean")` and `predict(which="prob")`
max absolute differences 0.0 and ≤ 1.4e-15 respectively.

## Out-of-sample CV, 0 folds failed for either family in any mode

| mode | family | CV MAE (mean ± SD) | CV RMSE (mean ± SD) | pooled OOF MAE |
|---|---|---|---|---|
| bike | NB | 0.5945 ± 0.1067 | 1.0498 ± 0.3134 | 0.5946 |
| bike | ZINB | 0.5945 ± 0.1067 | 1.0498 ± 0.3134 | 0.5946 |
| ped | NB | 0.7269 ± 0.0686 | 1.0968 ± 0.1386 | 0.7271 |
| ped | ZINB | 0.7269 ± 0.0686 | 1.0968 ± 0.1386 | 0.7271 |
| vehicle | NB | 2.8868 ± 0.2225 | 4.4548 ± 0.4145 | 2.8861 |
| vehicle | ZINB | 2.8867 ± 0.2225 | 4.4548 ± 0.4145 | 2.8861 |

Identical to 3 to 4 decimal places, which follows directly from π̂ ≈ 0.

## Vuong test

The statistic *was* implemented from the per-observation log-likelihoods
(`res.model.loglikeobs(res.params)`), and the reconciliation check passes exactly:
`sum(loglikeobs) − llf = 0.0` for both models in all three modes.

**But the result is degenerate and is reported as such rather than dressed up.** Because
π̂ collapsed to ~1e-14, the two models produce the same per-observation log-likelihoods
(max |difference| = 1.9e-05 bike, 2.6e-05 ped, 9.0e-07 vehicle), so the raw statistic is
`V = 2.9e-05` (bike), `3.9e-05` (ped), `1.9e-06` (vehicle), i.e. **zero to within optimiser
tolerance, two-sided p ≈ 1.0**, meaning "these are the same model", not "these are two
distinguishable models that tie". The Akaike- and Schwarz-corrected Vuong variants are a
fixed constant divided by a numerical-noise standard deviation; they took absurd values
(order 1e4 to 1e6) and have been **suppressed in the JSON with an explicit reason** rather than
reported.

Two caveats stated on the record: (1) Vuong (1989) assumes strictly non-nested models,
whereas NB is the limit of ZINB as the inflation probability → 0, so the null distribution
is not exactly standard normal in this application (cf. Wilson 2015). The statistic is
suggestive at best; (2) for the same nesting-at-a-boundary reason, **no likelihood-ratio
test of NB vs ZINB is reported**, since π = 0 corresponds to a logit intercept at −∞ rather
than a finite boundary point and the usual mixture correction does not straightforwardly apply.
The verdict below therefore rests on AIC/BIC, the observed-vs-expected zero counts with their
bootstrap bands, and out-of-sample CV. Not on Vuong.

## Verdict, E7

**Zero-inflation is not warranted for any of the three modes, and the result is about as
decisive as this kind of comparison gets.** The ZINB's inflation probability collapses to the
NB boundary in every mode (π̂ = 1.3e-13 / 5.0e-14 / 1.8e-14, log-likelihood gain ≤ 3e-09), so
ZINB is strictly worse by AIC (+2.000) and BIC (+5.846) everywhere while making numerically
identical predictions (CV MAE differs by ≤ 0.0001). The plain NB already reproduces the
observed zero counts almost exactly, 256 observed vs 256.45 expected (bike), 204 vs 206.02
(ped), 89 vs 90.86 (vehicle), all comfortably inside a 4000-draw parametric bootstrap band
and every gap is *negative*, meaning the NB slightly over-predicts zeros rather than
under-predicting them. **Despite the sparsity that motivated the question, bike crashes show
no excess-zero problem whatsoever.** The one genuine surprise is procedural, not statistical:
the repo's zero-inflation check is broken *twice over*. The `DESIGN_PREDICTORS` ImportError
on record at `HEAD`, and behind it a pandas-version pickle incompatibility that also silently
disables `pipeline/tests/test_calibration.py`. Fixing only the import would not have made the
check run.

---

# What these experiments do NOT establish

* E2 compares NB2 against Poisson only. Quasi-Poisson, NB1, Conway-Maxwell-Poisson and
  hurdle models were not fit.
* E2's CV verdict concerns point accuracy (MAE/RMSE). CV *likelihood*. Which would be
  sensitive to the family, was not computed.
* E6 Part B's per-mode arms depend on a mode-to-year allocation step that is synthetic,
  because the year grid has no mode breakdown. Only the `total_supplementary` arm uses real
  crash-year assignments, and even there the observation window is synthetic. **Nothing in
  Part B changes the Part A conclusion for the production dataset.**
* E7 fits a constant-only (intercept-only) inflation equation. An inflation model with
  covariates was not fit; given π̂ ≈ 1e-14 with a constant term it is unlikely to change the
  verdict, but it was not tested.
* All 346 rows are Capitol Hill arterial intersections with a single 6-year window; nothing
  here speaks to other facility types, other cities, or other time periods.
