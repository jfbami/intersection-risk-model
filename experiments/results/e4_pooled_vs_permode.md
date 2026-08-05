# E4, Pooled all-crash model + share scaling vs three per-mode models

Run `2026-08-05T05:15:39Z` · Python 3.13.2, numpy 2.1.3, pandas 2.2.3 · 346 arterial intersections

## The decision under test

The project's model went from **v2** (`nb_v2_arterial_aadt`: ONE pooled Negative
Binomial on `total_crashes`, with mode-specific risk obtained by scaling that
single all-crash prediction by a citywide share) to **v3** (`nb_v3_bike`/`ped`/
`vehicle`: THREE separate NB fits, one per crash mode).

The v2→v3 rewrite was squashed into a single commit (`c72f1ec`) whose message
describes what v3 is but never says why the pooled model was replaced. There is
no recorded comparison anywhere in the repo. This experiment reconstructs one.

It matters because the project's headline output is **expected bike KSI per
year**: under v2 that was a scaled all-crash prediction, under v3 it comes from
a dedicated bike model.

## Commands run

```bash
cd C:/Users/jfbaa/project-cycle-group && \
  PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e4_pooled_vs_permode.py
cd C:/Users/jfbaa/project-cycle-group && \
  PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e4_report.py
```

Nothing under `pipeline/` or `data/` was modified. `pipeline.fit_risk_model.main()`
was never called; only `load_and_join`, `prepare`, `MODES` and `SHARED_PREDICTORS`
were imported. All artifacts written go to `experiments/results/`.

## Data, and the 1720-vs-1730 double-counting issue

`prepare(load_and_join())` yields **346** arterial rows; `offset` is a constant
`log(6)` for every row.

| target | events |
|---|---|
| `total_crashes` | 1720 |
| `bike_total` | 169 |
| `ped_total` | 266 |
| `vehicle_only_total` | 1295 |
| `bike_ksi_total` | 16 |
| `ped_ksi_total` | 32 |
| `vehicle_only_ksi_total` | 25 |
| `ksi_total` | 71 |

**The three mode targets sum to 1730, but
`total_crashes` is 1720. A discrepancy of
10 crashes.** The same applies to severity:
the three mode-KSI columns sum to 73 against
`ksi_total` = 71, a discrepancy of
2. The cause is crashes flagged as BOTH bike
and ped, which land in `bike_total` and `ped_total` simultaneously
(acknowledged at `pipeline/score_risk.py:14-18`).

**How this experiment handles it.** Every comparison in Parts A and B is made
*within a single mode*, Arm 1's scaled prediction and Arm 2's dedicated
prediction are scored against the *same* observed column, on the same 346 rows.
The double-counting is therefore identical on both sides of every comparison and
cancels out; it can bias neither arm. It does have two real consequences,
flagged where they arise:

1. The three per-mode shares sum to
   1.005814, i.e.
   0.58%
   above 1.0, so Arm 1's three scaled predictions also over-total by that amount.
   This is a property of the *targets*, not of the pooling choice.
2. It is one of the reasons summing likelihoods across the three per-mode fits
   and comparing to the pooled fit is not legitimate (Part C).

## Arm specifications

`SHARED_PREDICTORS` = `is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class)`

| arm | formula | fit |
|---|---|---|
| **Arm 1**, pooled + share scaling (v2 approach, v3 leg encoding) | `total_crashes ~ is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class) + log_aadt` | 1 NB |
| **Arm 1b**, pooled, *true historical v2 spec* | `total_crashes ~ is_signalized + num_legs + max_speed_limit + bike_facility + C(arterial_class) + log_aadt` | 1 NB |
| **Arm 2**, per-mode (bike, current v3) | `bike_total ~ is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class) + log_bike_centrality` | 1 of 3 NB |
| **Arm 2**, per-mode (ped, current v3) | `ped_total ~ is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class) + log_aadt` | 1 of 3 NB |
| **Arm 2**, per-mode (vehicle, current v3) | `vehicle_only_total ~ is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class) + log_aadt` | 1 of 3 NB |

All fits: `smf.negativebinomial(formula, data=df, offset=df['offset'].values).fit(disp=False)`,
prediction via `.predict(df, offset=df['offset'].values)`, matching production exactly.

Arm 1 predicts mode-*m* crashes as `pred_total × share_m`. **`share_m` is computed
on the training rows only inside each CV fold** (`sum(mode_m) / sum(total_crashes)`
over the training split). Computing it on the full data would leak held-out
information into the prediction; that was explicitly avoided. Per-fold training
shares are recorded in `e4_results.json → cv.fold_summary`.

### Why there are two pooled arms

Arm 1 holds the predictor encoding fixed at v3's (top-coded `legs_cat`) so the A/B
isolates *pooling vs splitting*. But the historical v2 used `num_legs` as a
continuous slope. Running both separates the two changes that the rewrite made
at once. The reconstruction of the historical spec is exact:

| all-crash fit | sum_pred | MAE | RMSE | calibration |
|---|---|---|---|---|
| Arm 1 spec (`legs_cat`) | 1705.7 | 3.188 | 4.838 | -0.83% |
| Arm 1b spec (`num_legs`) | 1772.7 | 3.474 | 5.425 | +3.06% |
| **v2 as recorded in README.md:162** | **1772.7** | **3.47** | n/a | **+3.1%** |

Arm 1b reproduces the recorded v2 numbers to the printed precision (sum_pred 1772.7 vs 1772.7; MAE 3.474 vs 3.47). **Arm 1b is the model that actually shipped as v2.**

Arm 2 is likewise verified against the shipped v3: the refit alphas
(1.2382 / 0.3049 / 0.6585 for bike/ped/vehicle) and in-sample MAEs
(0.577 / 0.698 / 2.764) reproduce the values recorded in
`MODEL_NOTES.md` §1 exactly. Both arms are faithful reconstructions.

## Convergence ledger

Production's ladder is newton → bfgs(maxiter=200) → `sys.exit`. This experiment
uses the identical ladder, then adds a Nelder-Mead attempt **as a diagnostic only**,
to distinguish optimiser fragility from genuine non-identifiability. Any fit that
needed the diagnostic step is recorded as a production failure.

| full-data fit | converged | method | logLik | AIC | BIC | alpha | k |
|---|---|---|---|---|---|---|---|
| `pooled` | yes | bfgs | -810.999 | 1645.997 | 1692.155 | 0.4972 | 12 |
| `permode_bike` | yes | bfgs | -275.656 | 575.313 | 621.470 | 1.2382 | 12 |
| `permode_ped` | yes | bfgs | -346.402 | 716.803 | 762.961 | 0.3049 | 12 |
| `permode_vehicle` | yes | bfgs | -745.202 | 1514.404 | 1560.562 | 0.6585 | 12 |
| `pooled_v2_legs_sensitivity` | yes | bfgs | -835.036 | 1690.072 | 1728.536 | 0.6401 | 10 |

Every full-data fit converged, all via the bfgs retry (newton alone failed for all
five, including the three production v3 fits. This is normal for this dataset and
is what `fit_for_mode`'s retry ladder exists for).

- Primary 5-fold CV: **25 fits, all converged** (`all_fits_converged = True`).
- Repeated CV (10 fold seeds): **200 fits, 1 failed** production's ladder.
  - `permode_bike_seed3_fold0` (n_train = 276): newton and bfgs both emitted
    *"Maximum Likelihood optimization failed to converge"*. Production
    `fit_for_mode` would have called `sys.exit` here. Nelder-Mead converged
    after 2792 iterations, so this is optimiser
    fragility rather than an unidentified model.

**This asymmetry is itself a result.** The failure was a *per-mode bike* fit on a
276-row training split. The bike model carries
169 events across 12 parameters
(~14 events/parameter);
the pooled model carries 1720 events across the same
12 parameters
(~143 events/parameter).
Across the whole experiment **230 NB fits** were run (5 full-data + 25 primary-CV + 200 repeated-CV), of which **1 failed** production's ladder. All 56 pooled fits
converged; the single failure was 1 of the 56 per-mode bike fits.
Splitting the target by mode divides the events but not the parameters.

## Part A, predicting each mode's crash count

### A.1 In-sample (what the repo has always measured)

| mode | arm | MAE | RMSE | Spearman ρ | sum_pred | calibration | NB log-score |
|---|---|---|---|---|---|---|---|
| bike | Arm 1 pooled×share | 0.5929 | 1.0619 | +0.4252 | 167.6 | -0.83% | -0.8234 |
| bike | **Arm 2 per-mode** | **0.5774** | **1.0458** | +0.4254 | 166.7 | -1.37% | **-0.7967** |
| ped | Arm 1 pooled×share | 0.7161 | 1.0645 | +0.5696 | 263.8 | -0.83% | -1.0178 |
| ped | **Arm 2 per-mode** | **0.6976** | **1.0501** | +0.5904 | 267.8 | +0.67% | **-1.0012** |
| vehicle | Arm 1 pooled×share | 2.7700 | 4.3197 | +0.6130 | 1284.2 | -0.83% | -2.1636 |
| vehicle | **Arm 2 per-mode** | **2.7639** | **4.2937** | +0.6090 | 1282.8 | -0.94% | **-2.1538** |

In-sample, Arm 2 wins on MAE and RMSE for all three modes. That is exactly what
36 parameters should do against 12 on the data they were fitted to.

Note the Spearman ρ for Arm 1 is *identical across all three modes by
construction*. The share is a positive constant, so scaling cannot change ranks.
Arm 1's mode ranking is always just the all-crash ranking.

### A.2 Out-of-sample, 5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=0)`)

Pooled out-of-fold metrics (all 346 held-out predictions concatenated):

| mode | arm | MAE | RMSE | Spearman ρ | calibration | NB log-score |
|---|---|---|---|---|---|---|
| bike | Arm 1 pooled×share | 0.6013 | 1.0749 | +0.4014 | -0.38% | -0.8327 |
| bike | Arm 2 per-mode | 0.5946 | 1.0866 | +0.3917 | -2.77% | -0.8302 |
| ped | Arm 1 pooled×share | 0.7290 | 1.0893 | +0.5536 | -0.33% | -1.0292 |
| ped | Arm 2 per-mode | 0.7271 | 1.1044 | +0.5415 | +0.10% | -1.0549 |
| vehicle | Arm 1 pooled×share | 2.8713 | 4.4620 | +0.5895 | -0.34% | -2.2072 |
| vehicle | Arm 2 per-mode | 2.8861 | 4.4717 | +0.5795 | -0.28% | -2.1957 |

Per-fold mean ± SD across the 5 folds:

| mode | arm | MAE | RMSE | Spearman ρ |
|---|---|---|---|---|
| bike | Arm 1 pooled×share | 0.6011 ± 0.1121 | 1.0302 ± 0.3422 | +0.4054 ± 0.0896 |
| bike | Arm 2 per-mode | 0.5945 ± 0.1067 | 1.0498 ± 0.3134 | +0.3820 ± 0.0984 |
| ped | Arm 1 pooled×share | 0.7287 ± 0.0606 | 1.0792 ± 0.1597 | +0.5539 ± 0.0899 |
| ped | Arm 2 per-mode | 0.7269 ± 0.0685 | 1.0968 ± 0.1386 | +0.5381 ± 0.0718 |
| vehicle | Arm 1 pooled×share | 2.8720 ± 0.2082 | 4.4467 ± 0.3940 | +0.6036 ± 0.0496 |
| vehicle | Arm 2 per-mode | 2.8867 ± 0.2225 | 4.4548 ± 0.4145 | +0.5945 ± 0.0529 |

Difference (Arm 2 − Arm 1) on the pooled out-of-fold metrics:

| mode | ΔMAE | ΔMAE % | ΔRMSE | ΔRMSE % | Δρ | bootstrap 95% CI on Δρ |
|---|---|---|---|---|---|---|
| bike | -0.00669 | -1.11% | +0.01170 | +1.09% | -0.0098 | [-0.0427, +0.0224] |
| ped | -0.00184 | -0.25% | +0.01506 | +1.38% | -0.0122 | [-0.0419, +0.0159] |
| vehicle | +0.01481 | +0.52% | +0.00978 | +0.22% | -0.0100 | [-0.0213, +0.0015] |

Negative ΔMAE/ΔRMSE favours Arm 2; positive Δρ favours Arm 2.

**The in-sample advantage largely evaporates out of sample.** On `random_state=0`,
Arm 2 keeps a small MAE edge for bike and ped but loses on RMSE for all three
modes and loses on ρ for all three.

### A.3 Repeated CV over 10 fold seeds

The Part A effects are on the order of 1% of MAE, which a single fold assignment
cannot resolve. The whole CV was therefore repeated with `random_state` 0 to 9
(200 fits total). Seeds containing a fit that failed production's
ladder are excluded from the aggregates (per-seed values for all 10 retained in
the JSON).

| mode | metric | Arm 1 mean ± SD | Arm 2 mean ± SD | Δ (A2−A1) mean ± SD | Δ range | seeds Arm 2 wins |
|---|---|---|---|---|---|---|
| bike | MAE | 0.59930 ± 0.00406 | 0.60711 ± 0.00936 | +0.00780 ± 0.00865 | [-0.00669, +0.01765] | **2/9** |
| bike | RMSE | 1.07266 ± 0.00484 | 1.09853 ± 0.01614 | +0.02587 ± 0.01359 | [-0.00116, +0.04133] | **1/9** |
| bike | ρ | 0.40510 ± 0.00919 | 0.36767 ± 0.01610 | -0.03743 ± 0.01506 | [-0.06554, -0.00976] | **0/9** |
| ped | MAE | 0.72961 ± 0.00657 | 0.72695 ± 0.00592 | -0.00266 ± 0.00542 | [-0.01033, +0.00685] | **7/10** |
| ped | RMSE | 1.08835 ± 0.01132 | 1.10158 ± 0.01015 | +0.01323 ± 0.00791 | [+0.00056, +0.02434] | **0/10** |
| ped | ρ | 0.55153 ± 0.00668 | 0.55138 ± 0.00784 | -0.00015 ± 0.00868 | [-0.01216, +0.01388] | **5/10** |
| vehicle | MAE | 2.87078 ± 0.02538 | 2.88410 ± 0.02446 | +0.01332 ± 0.00930 | [+0.00402, +0.03651] | **0/10** |
| vehicle | RMSE | 4.47753 ± 0.04022 | 4.48625 ± 0.04140 | +0.00872 ± 0.01133 | [-0.00720, +0.03390] | **1/10** |
| vehicle | ρ | 0.58479 ± 0.00816 | 0.57281 ± 0.01016 | -0.01198 ± 0.00319 | [-0.01783, -0.00861] | **0/10** |

**This overturns the `random_state=0` reading for bike.** Averaged over seeds:

- **bike**: the dedicated model is *worse* out of sample, MAE +0.00780 (2/9 seeds won), RMSE +0.02587 (1/9), ρ -0.03743 (0/9). The seed-0 MAE win was the exception, not the rule.
- **ped**: a wash, MAE -0.00266 (7/10 seeds), but RMSE +0.01323 (0/10) and ρ -0.00015 (5/10).
- **vehicle**: the dedicated model is worse on every metric, MAE +0.01332 (0/10), ρ -0.01198 (0/10).

### A.4 Headline answer

**No. The dedicated bike model does not beat scaling a pooled prediction.**
Out of sample it is worse by +0.00780 MAE (+1.30%) and -0.0374 Spearman ρ, losing on ρ in 9/9 fold seeds.
The per-mode models win in-sample and lose out-of-sample: the textbook signature
of extra parameters buying fit rather than signal.

## Part B, predicting bike KSI, the project's headline metric

Observed: **16 bike-KSI events across 346 sites**, concentrated in 15 sites.

Routes to a bike-KSI prior, matching `pipeline/score_risk.py:207`
(`mu_ksi = predictions[mode.crash_predicted] * city_share`, with `city_share` from
`citywide_mode_ksi_share` = `ksi_actual.sum() / crash_actual.sum()`):

- **Arm 1 (v2 route)**: `pred_total_pooled × 0.009302` (= 16/1720)
- **Arm 2 (v3 route)**: `pred_bike_permode × 0.094675` (= 16/169)

90% predictive intervals use `nbinom(n=1/alpha, p=1/(1+alpha*mu))`, the identical
estimator to `pipeline/tests/test_calibration.py:33`, with each arm borrowing its
own model's α (as `test_calibration.py` does for the KSI proxy).

### B.1 Results

| basis | arm | MAE | RMSE | Spearman ρ | calibration | 90% coverage |
|---|---|---|---|---|---|---|
| in-sample | Arm 1 (v2 route) | 0.08410 | 0.21729 | +0.2218 | -0.83% | 99.13% |
| in-sample | Arm 2 (v3 route) | 0.08306 | 0.21663 | +0.2281 | -1.37% | 99.13% |
| **out-of-fold** | Arm 1 (v2 route) | 0.08492 | 0.21929 | +0.2043 | -0.75% | 98.55% |
| **out-of-fold** | Arm 2 (v3 route) | 0.08409 | 0.22195 | +0.1681 | -3.29% | 98.55% |
| out-of-fold | Arm 1b (*true* v2 spec) | 0.08668 | 0.21815 | +0.1797 | +7.30% | 98.55% |

Per-fold mean ± SD:

| arm | MAE | RMSE | Spearman ρ | coverage |
|---|---|---|---|---|
| Arm 1 (v2 route) | 0.08493 ± 0.02042 | 0.20837 ± 0.07653 | +0.2381 ± 0.1401 | 98.56 ± 1.44% |
| Arm 2 (v3 route) | 0.08412 ± 0.02180 | 0.21245 ± 0.07200 | +0.2010 ± 0.1338 | 98.56 ± 1.44% |

Only **4 of 5 folds** yield a defined Spearman ρ: fold 4 contains zero
bike-KSI events in its held-out set, so ρ is undefined there. That alone shows how
thin this target is.

Both arms' 90% predictive intervals are heavily over-covering
(98.55% and 98.55% against a nominal 90%),
identically. With a mean predicted KSI count near 0.05 the interval is `[0, 1]` for
almost every site, so coverage carries essentially no discriminating information
here. It is reported because it was asked for, not because it separates the arms.

### B.2 Statistical power: bootstrapped Spearman

Paired site-level bootstrap, 2000 resamples,
seed 0. Sites are resampled with replacement and *both* arms are re-scored on the
same resample, so the difference is paired. Percentile 95% CIs.

| basis | Arm 1 ρ [95% CI] | Arm 2 ρ [95% CI] | **Δρ (A2−A1) [95% CI]** | P(Δ>0) |
|---|---|---|---|---|
| in-sample | +0.2208 [+0.1290, +0.3042] | +0.2270 [+0.1282, +0.3172] | **+0.0063 [-0.0163, +0.0314]** | 0.699 |
| out-of-fold | +0.2035 [+0.1196, +0.2839] | +0.1678 [+0.0693, +0.2572] | **-0.0357 [-0.0764, -0.0011]** | 0.021 |
| out-of-fold, vs **Arm 1b** (true v2) | +0.1790 [+0.0816, +0.2698] | +0.1678 [+0.0693, +0.2572] | **-0.0112 [-0.0469, +0.0240]** | 0.273 |

Cross-checked against the 10-seed repeated CV (bike-KSI ρ):

| Arm 1 ρ (mean ± SD) | Arm 2 ρ (mean ± SD) | Δρ (A2−A1) mean ± SD | Δρ range | seeds Arm 2 wins |
|---|---|---|---|---|
| 0.2054 ± 0.0039 | 0.1514 ± 0.0208 | -0.0540 ± 0.0195 | [-0.0867, -0.0227] | **0/9** |

### B.3 Reading this honestly

**In-sample the two are indistinguishable.** Δρ = +0.0063, 95% CI [-0.0163, +0.0314], comfortably
containing zero. On the evidence the repo actually has (in-sample only), **this
dataset cannot distinguish the two approaches.**

**Out of sample the comparison does resolve, and it does not favour v3.** Δρ = -0.0357, 95% CI [-0.0764, -0.0011], excluding zero,
with P(Δ>0) = 0.021. The repeated CV
agrees and is stronger: Arm 2 loses on ρ in **9/9** fold seeds,
mean Δρ -0.0540 ± 0.0195. The *sign* is stable.

Three caveats that must travel with that statement:

1. **The magnitude is small and both arms are weak.** The arms' own ρ CIs
   ([+0.1196, +0.2839] and [+0.0693, +0.2572]) overlap almost
   entirely. The defensible claim is *"per-mode does not beat pooled"*, **not**
   *"pooled predicts bike KSI well"*. Neither does.
2. **The paired bootstrap holds predictions fixed.** It captures sampling noise in
   the evaluation set, not parameter-estimation uncertainty, so its CI is narrower
   than a full accounting would be. The 10-seed repeated CV covers fold-assignment
   uncertainty; neither covers the fact that all 346 sites are one city, one
   6-year window.
3. **Against the *true* v2 (Arm 1b) the bike-KSI comparison is a genuine tie**:
   Δρ -0.0112, 95% CI [-0.0469, +0.0240]
   contains zero.

### B.4 Event capture. The decision-relevant view

With 16 events, "how many real KSI events land in the top-k prioritised sites" is
more interpretable than ρ.

| k | Arm 1 in-sample | Arm 2 in-sample | Arm 1 out-of-fold | Arm 2 out-of-fold | random |
|---|---|---|---|---|---|
| 10 | 2/16 (12.5%) | 2/16 (12.5%) | 1/16 (6.2%) | 0/16 (0.0%) | 2.9% |
| 20 | 5/16 (31.2%) | 5/16 (31.2%) | 2/16 (12.5%) | 3/16 (18.8%) | 5.8% |
| 50 | 7/16 (43.8%) | 10/16 (62.5%) | 6/16 (37.5%) | 6/16 (37.5%) | 14.5% |
| 100 | 14/16 (87.5%) | 13/16 (81.2%) | 14/16 (87.5%) | 12/16 (75.0%) | 28.9% |

Both arms beat random by a wide margin and are near-identical to each other. Out
of fold the differences are 0 to 2 events, well inside what 16 events can resolve.

## Part C, model complexity and parsimony

- **Arm 1**: 1 NB model, **12 parameters**
  (11 regression coefficients + α), plus 1 estimated share per mode.
- **Arm 2**: 3 NB models, **36 parameters** (12 + 12 + 12).

Arm 2 spends **3× the parameters**.

### Is summing AIC across the three fits legitimate? No.

| quantity | Arm 1 | Arm 2 (sum of 3) |
|---|---|---|
| logLik | -810.999 | -1367.260 |
| AIC | 1645.997 | 2806.521 |
| BIC | 1692.155 | 2944.993 |

**These two columns must not be compared, and the apparent 1160-point AIC gap is
meaningless.** Reasons:

1. **Different random variables.** Arm 1's likelihood is over `total_crashes`;
   Arm 2's is over three different targets. AIC/BIC are only comparable across
   models fitted to *the same* observations. There is no shared reference measure.
2. **Different event totals.** 1720 vs 1730,
   the latter double-counting 10 bike+ped crashes. Arm 2's
   summed likelihood is over data that partly counts the same crash twice.
3. **Arm 2's sum is a likelihood of three independent fits**, which is not the
   joint likelihood of the modes. They are correlated at a site by construction.

Anyone quoting "pooled AIC 1646 beats per-mode AIC 2807" would be quoting a
number with no inferential content. It is recorded here only to say plainly that
it should not be used.

### What *is* legitimate: same-target NB log-score

Both arms produce a predictive distribution for the *same* mode count, so they can
be scored with the same proper scoring rule (mean NB log predictive density,
higher is better). In-sample:

| mode | Arm 1 (α borrowed) | Arm 1 (α refit) | Arm 2 | free params A1 / A2 |
|---|---|---|---|---|
| bike | -0.8234 | -0.8057 | -0.7967 | 1 / 12 |
| ped | -1.0178 | -1.0161 | -1.0012 | 1 / 12 |
| vehicle | -2.1636 | -2.1564 | -2.1538 | 1 / 12 |

Out of fold (weighted by fold size):

| mode | Arm 1 | Arm 2 | Δ (A2−A1) |
|---|---|---|---|
| bike | -0.8327 | -0.8302 | +0.0026 |
| ped | -1.0292 | -1.0549 | -0.0258 |
| vehicle | -2.2072 | -2.1957 | +0.0115 |

Arm 2's log-score edge is a few thousandths of a nat per site, bought with 24 extra
parameters, and it does not translate into better MAE/RMSE/ρ out of sample.

**Parsimony verdict, judged on out-of-sample error as instructed:** Arm 1 is
strictly preferable. It is 3× smaller, never failed to converge in 56 fits, and
is at least as accurate out of sample on every mode.

## Part D. Where the two approaches disagree

All 346 sites ranked by predicted bike KSI under each arm (full-data fits, which
is what both v2 and v3 actually shipped).

- Spearman between the two arms' **prior** rankings: **0.9601** (Kendall τ 0.8233)
- Spearman between the two arms' **EB-posterior** rankings (mirroring
  `score_risk.compute_mode_ksi_eb`, which is what `risk_rank` actually uses): **0.9621** (Kendall τ 0.8298)

| overlap | prior ranking | EB ranking |
|---|---|---|
| top-10 | 4/10 | 7/10 |
| top-20 | 12/20 | 16/20 |
| top-50 | 42/50 | 44/50 |

| rank-shift statistic | prior | EB |
|---|---|---|
| mean | 21.2 | 20.4 |
| median | 16.0 | 15.0 |
| 90th pct | 46.0 | 45.2 |
| max | 121.0 | 120.0 |

### The 10 largest rank shifts

| intersection_id | bike crashes | bike KSI | total crashes | μ Arm 1 | μ Arm 2 | rank A1 | rank A2 | shift |
|---|---|---|---|---|---|---|---|---|
| `035bd7c03c9d` | 0 | 0 | 1 | 0.0494 | 0.0190 | 98 | 219 | 121 |
| `78f6e60e1693` | 0 | 0 | 2 | 0.0479 | 0.0251 | 101 | 195 | 94 |
| `62e1ef5c56ea` | 0 | 0 | 3 | 0.0458 | 0.0242 | 110 | 198 | 88 |
| `d2627b97845d` | 0 | 0 | 5 | 0.0721 | 0.0315 | 82 | 162 | 80 |
| `394633f5fc08` | 0 | 0 | 4 | 0.0342 | 0.0540 | 176 | 100 | 76 |
| `71ddc4d720dc` | 0 | 0 | 6 | 0.0731 | 0.0348 | 80 | 156 | 76 |
| `4cf57bebe8b0` | 0 | 0 | 0 | 0.0674 | 0.0318 | 89 | 161 | 72 |
| `a06f9bbfefb0` | 1 | 0 | 4 | 0.0381 | 0.0594 | 155 | 86 | 69 |
| `01699c227158` | 0 | 0 | 2 | 0.0716 | 0.0379 | 83 | 150 | 67 |
| `ab7174653e32` | 0 | 0 | 1 | 0.0627 | 0.0342 | 92 | 157 | 65 |

**Every one of the 10 largest shifts is a site with 0 observed bike KSI, and 9 of
10 have 0 or 1 observed bike crashes.** They sit in the flat middle of the
distribution (ranks ~80 to 220) where predicted KSI differs by hundredths of an
event. These moves are numerically large and practically irrelevant.

### The decision-relevant head of the ranking

Union of each arm's top 20 = **28 distinct sites** (perfect agreement would be 20). Mean rank shift within that set 17.0, median 16.0, max 46.0.

Union of each arm's top 50 = **58 distinct sites**; mean shift 17.8, max 46.0.

| intersection_id | bike | bike KSI | total | μ A1 | μ A2 | rank A1 | rank A2 | shift |
|---|---|---|---|---|---|---|---|---|
| `4efd4397bfbc` | 1 | 0 | 24 | 0.1845 | 0.1958 | 1 | 2 | 1 |
| `a69226a229ad` | 0 | 0 | 7 | 0.1574 | 0.1629 | 2 | 13 | 11 |
| `cdb875cde624` | 2 | 1 | 4 | 0.1447 | 0.1551 | 3 | 20 | 17 |
| `407803206118` | 0 | 0 | 9 | 0.1418 | 0.1463 | 4 | 22 | 18 |
| `a932fcb41bb1` | 0 | 0 | 40 | 0.1399 | 0.1668 | 5 | 9 | 4 |
| `05467827ef34` | 1 | 0 | 24 | 0.1382 | 0.0834 | 6 | 46 | 40 |
| `3dec2b9f9160` | 2 | 0 | 28 | 0.1379 | 0.1632 | 7 | 12 | 5 |
| `afbd562d0b58` | 0 | 0 | 8 | 0.1376 | 0.1756 | 8 | 6 | 2 |
| `a6b897eea101` | 2 | 1 | 21 | 0.1338 | 0.1740 | 9 | 7 | 2 |
| `5786cba5578d` | 3 | 0 | 17 | 0.1329 | 0.1443 | 10 | 27 | 17 |
| `33f5b8caa58f` | 0 | 0 | 0 | 0.1327 | 0.1764 | 11 | 5 | 6 |
| `6f39cc8b472f` | 9 | 1 | 40 | 0.1309 | 0.1648 | 12 | 10 | 2 |
| `ffed5006142f` | 9 | 1 | 19 | 0.1309 | 0.1601 | 13 | 15 | 2 |
| `ffc015bea08e` | 0 | 0 | 4 | 0.1293 | 0.1735 | 14 | 8 | 6 |
| `a20bf0bcdf38` | 3 | 1 | 27 | 0.1243 | 0.0791 | 15 | 50 | 35 |
| `1b69a47ecb58` | 1 | 0 | 5 | 0.1240 | 0.1439 | 17 | 28 | 11 |
| `107d328fd7ac` | 3 | 0 | 6 | 0.1240 | 0.1490 | 17 | 21 | 4 |
| `2d1a747f6a58` | 1 | 0 | 3 | 0.1240 | 0.1424 | 17 | 32 | 15 |
| `6a49a840502c` | 0 | 0 | 10 | 0.1234 | 0.0781 | 19 | 54 | 35 |
| `1aa39a558f89` | 0 | 0 | 13 | 0.1234 | 0.2007 | 20 | 1 | 19 |
| `fb5b5d87c308` | 5 | 1 | 20 | 0.1181 | 0.1643 | 27 | 11 | 16 |
| `10e4191de520` | 2 | 0 | 12 | 0.1097 | 0.1598 | 32 | 16 | 16 |
| `569aafe7605a` | 0 | 0 | 6 | 0.1087 | 0.1576 | 35 | 17 | 18 |
| `712128e34142` | 2 | 0 | 24 | 0.1072 | 0.1935 | 37 | 3 | 34 |
| `93eee812b2e2` | 1 | 0 | 7 | 0.1052 | 0.1566 | 40 | 18 | 22 |
| `27f43f41187b` | 0 | 0 | 23 | 0.1046 | 0.1624 | 44 | 14 | 30 |
| `aa90d1c8a2c3` | 0 | 0 | 4 | 0.0986 | 0.1766 | 50 | 4 | 46 |
| `293c589368c9` | 1 | 0 | 12 | 0.0933 | 0.1564 | 61 | 19 | 42 |

### Does the modelling choice change what gets built?

**Partly. And more than the ρ = 0.96 headline suggests.** The two rankings agree
almost perfectly overall, but the top-10 overlap is only 4/10 on the prior ranking and 7/10 after EB. A city funding its worst 10 intersections
would send crews to a materially different set depending on which model version
shipped. By top-50 the sets have largely reconverged
(42/50 prior, 44/50 EB).

The honest framing: the choice **does** move specific sites in and out of a short
priority list, but Parts A and B show there is **no evidence the v3 ordering is the
better one**, out of sample it is, if anything, slightly worse. The reshuffling is
churn, not improvement.

## Disentangling the rewrite: pooling vs leg encoding

The v2→v3 commit changed two things at once: it split one model into three, **and**
it changed `num_legs` from a continuous slope to a top-coded categorical. Arm 1b
(the exactly reconstructed historical v2) isolates that second change.

Out-of-fold, 5-fold CV, `random_state=0`:

| mode | Arm 1b (true v2) | Arm 1 (pooled + `legs_cat`) | Arm 2 (v3 per-mode) |
|---|---|---|---|
| bike MAE | 0.6380 | 0.6013 | **0.5946** |
| bike RMSE | 1.1379 | **1.0749** | 1.0866 |
| bike ρ | +0.3820 | **+0.4014** | +0.3917 |
| ped MAE | 0.8010 | 0.7290 | **0.7271** |
| ped RMSE | 1.2680 | **1.0893** | 1.1044 |
| ped ρ | +0.5224 | **+0.5536** | +0.5415 |
| vehicle MAE | 3.0824 | **2.8713** | 2.8861 |
| vehicle RMSE | 5.1422 | **4.4620** | 4.4717 |
| vehicle ρ | +0.5393 | **+0.5895** | +0.5795 |
| bike-KSI MAE | 0.08668 | 0.08492 | **0.08409** |
| bike-KSI ρ | +0.1797 | **+0.2043** | +0.1681 |

On this fold assignment Arm 1 beats Arm 2 in **8 of the 11** Arm-1-vs-Arm-2 cells above; Arm 1b (true v2) loses to both in every one.

**This is the most important table in the report.**

- **Arm 2 clearly beats Arm 1b.** Against what actually shipped as v2, the v3
  per-mode models improve bike MAE from 0.6380 to 0.5946, ped from 0.8010 to 0.7271, vehicle from 3.0824 to 2.8861, and ρ on all three modes. The rewrite genuinely
  improved the model.
- **But Arm 1 beats Arm 2 on most of the same metrics.** Keeping one pooled model
  and adopting *only* the categorical leg encoding is enough to match or beat the
  three per-mode models, at a third of the parameters. The 10-seed repeated CV
  (§A.3) is the stronger evidence here, and it favours Arm 1 for bike
  (9/9 seeds on ρ) and vehicle
  (10/10 seeds on MAE), with ped a wash.

**So the measured gain from the v2→v3 rewrite is attributable to the leg-encoding
change, not to splitting the model by mode.** The per-mode split came along for the
ride and, on this data, costs a little accuracy and a lot of robustness.

(Calibration corroborates: Arm 1b over-predicts out of fold by +7.36%, against -0.38% for Arm 1. The continuous
`num_legs` slope extrapolates badly to rare 5+/6-leg geometries, exactly the
failure mode the `legs_cat` docstring in `fit_risk_model.py:69-71` describes.)

Sensitivity note: the two pooled specs' predictions correlate at Spearman 0.9279, so they are genuinely different models,
not a re-parameterisation.

## Threats to validity

- **One city, one 6-year window, 346 sites.** CV resamples sites, not cities or
  years. Nothing here speaks to transfer.
- **16 bike-KSI events.** Part B is power-limited by construction. The out-of-fold
  difference resolves only because the paired design cancels most of the noise;
  the absolute predictive quality of *both* arms is poor.
- **The paired bootstrap does not refit models.** See B.3 caveat 2.
- **α is borrowed, not modelled, for the KSI proxy** in both arms. The same
  simplification `score_risk.py` and `test_calibration.py` already make. It is
  applied identically to both arms.
- **`years_observed` is constant at 6**, so the offset only shifts the intercept
  and plays no differentiating role between arms.
- **Arm 1's shares are estimated, adding 1 free parameter per mode** that the
  parameter counts in Part C attribute to it. Even counting generously, Arm 1 uses
  12 + 3 = 15 vs Arm 2's 36.

## Verdict

**1. Did the dedicated per-mode models beat scaling a pooled prediction? No.**
In-sample they win on MAE and RMSE for all three modes. And in-sample is all the
repo has ever measured. Out of sample, across 10 fold seeds, per-mode is worse for
bike (ΔMAE +0.00780, Δρ -0.0374, losing ρ in
9/9 seeds), a wash for ped, and worse for
vehicle (ΔMAE +0.01332, losing MAE in 10/10 seeds).

**2. Was the bike-KSI comparison decisive? Partly, and not in v3's favour.**
In-sample the arms are statistically indistinguishable (Δρ = +0.0063, 95% CI [-0.0163, +0.0314]). Out of fold, Δρ = -0.0357, 95% CI [-0.0764, -0.0011], excluding zero,
against v3, and confirmed by 9/9 seeds. But both arms predict bike KSI
poorly and their individual ρ CIs overlap almost entirely, so the correct claim is
*"per-mode does not beat pooled"*, not *"pooled is good"*.

**3. Do the site rankings materially differ?** Overall, barely, Spearman
0.9601 (prior), 0.9621 (EB). But at the sharp end
the top-10 lists share only 4/10 sites
(7/10 after EB), so a short priority list *would* differ.
Since no arm demonstrably ranks better, that difference is churn.

**4. Was the v2→v3 rewrite justified on measured evidence?**

**The rewrite improved the model, but not for the reason its structure implies,
and the per-mode split specifically is not supported.** Against the true v2 (Arm 1b)
v3 is clearly better out of sample on every mode. But a pooled model with v2's
structure and only the leg-encoding fixed (Arm 1) matches or beats v3 everywhere,
using 12 parameters instead of 36 and without the convergence fragility that made 1 of the 56 per-mode
bike fits fail production's retry ladder (all 56 pooled fits converged).

**The measured gain belongs to the leg-encoding change. The three-model split
delivered no measurable out-of-sample benefit and cost robustness.** No evidence
recorded in the repo supports it, and this experiment does not either. The
defensible engineering claim for v3 is *"it lets us report per-mode risk and
compare coefficients across modes"*. An interpretability and product argument,
which is legitimate but was never stated, and is not an accuracy argument.

---

Machine-readable results: `experiments/results/e4_results.json`.
Scripts: `experiments/ab/e4_pooled_vs_permode.py` (experiment), `experiments/ab/e4_report.py` (this report).
