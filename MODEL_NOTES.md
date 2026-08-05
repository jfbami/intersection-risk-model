# Model-selection evidence and current architecture

Generated 2026-08-04 against `main` @ `780efca`. Working tree was clean for all
`data/` and `pipeline/` paths at start and at finish.

**Headline for the portfolio rewrite:** the repo's own `README.md` is stale. It
documents `nb_v2_arterial_aadt`, a *single* all-crash model, as though it were
current. The code on `main` fits **three per-mode models** (`nb_v3_*`). Every
number in your claims (e) is a v2 number copied out of that stale README. There
is almost no recorded A/B evidence anywhere in this repo; most modeling
decisions are stated justifications, not measured comparisons.

---

## 0. What I ran, and what I had to do to get there

`pipeline/evaluate_models.py` **does not run on `main`.** It is broken code, not
a missing-input problem. Details in §1.

To recover current model numbers I instead ran `pipeline/fit_risk_model.py`,
which works. Two complications, both worth knowing:

1. **The committed `.pkl` files cannot be loaded by the environment in this
   repo.** `pickle.load` on all four models in `data/model/` raises
   `TypeError: StringDtype.__init__() takes from 1 to 2 positional arguments but
   3 were given`. They were serialized under a different (newer) pandas.
   Installed here: Python 3.13.2, pandas 2.2.3, numpy 2.1.3, statsmodels 0.14.6,
   scipy 1.15.1, patsy 1.0.2. `requirements.txt` pins nothing, so the
   environment that produced the committed pickles is not reproducible.
2. Therefore I **re-fit** the models (`python -m pipeline.fit_risk_model`),
   which overwrites `data/model/nb_v3_*.pkl` and
   `data/intermediate/intersection_predictions.parquet`. I backed those five
   tracked files up first and **restored them afterwards**; `git status` is back
   to its starting state (only the pre-existing `frontend/package-lock.json`
   modification and untracked `.claude/worktrees/`).

I did **not** need to run `assemble_features.py` or `snap_crashes.py`; their
parquet outputs were present and current.

**Caveat on every "current" number below:** they come from my re-fit under
pandas 2.2.3 / statsmodels 0.14.6, not from the committed pickles. They should
be identical (same data, same formula, deterministic MLE) but I could not verify
against the committed artifacts because those will not deserialize.

---

## 1. Verbatim `evaluate_models.py` output

Run as instructed:

```
$ python pipeline/evaluate_models.py
Traceback (most recent call last):
  File "C:\Users\jfbaa\project-cycle-group\pipeline\evaluate_models.py", line 32, in <module>
    from pipeline.fit_risk_model import (
    ...<5 lines>...
    )
ModuleNotFoundError: No module named 'pipeline'
```

The script uses absolute `pipeline.*` imports, so it must be run as a module.
Retried that way, and after regenerating the pickles, so this is not a
missing-artifact failure:

```
$ python -m pipeline.evaluate_models
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\jfbaa\project-cycle-group\pipeline\evaluate_models.py", line 32, in <module>
    from pipeline.fit_risk_model import (
    ...<5 lines>...
    )
ImportError: cannot import name 'DESIGN_PREDICTORS' from 'pipeline.fit_risk_model' (C:\Users\jfbaa\project-cycle-group\pipeline\fit_risk_model.py)
```

**Root cause.** [`evaluate_models.py:32-38`](pipeline/evaluate_models.py:32)
imports `DESIGN_PREDICTORS`; [`fit_risk_model.py:72`](pipeline/fit_risk_model.py:72)
defines the constant as `SHARED_PREDICTORS`. The name `DESIGN_PREDICTORS` exists
nowhere else in the repo, and `git log --all -S` shows **both** names were
introduced in the same commit `c72f1ec`. So `evaluate_models.py` has never
executed successfully against any committed state of this repository.

**Consequence for your case study:** there is no VIF table, no pseudo-R², no
AIC, no Spearman, no zero-inflation check, and no cross-mode residual
correlation on record. The script that would produce all of them has never run.
Do not cite anything as coming from it.

### What `fit_risk_model.py` produces instead (this ran clean)

Full stdout, verbatim:

```
Joined dataset: 651 rows (expect 651).

Scope filter:
  raw rows                   651
  dropped: non-arterial      286
  dropped: missing AADT      19
  rows used for fit          346

years_observed: all rows == 6.  Good.
max_speed_limit: fully populated (0 NaN). Good.

Fitting Bicycle model (target: bike_total, n_events: 169)...
  Convergence: OK

Fitting Pedestrian model (target: ped_total, n_events: 266)...
  Convergence: OK

Fitting Motor vehicle only model (target: vehicle_only_total, n_events: 1295)...
  Convergence: OK

======================================================================
  Bicycle model  (target: bike_total)
======================================================================
                     NegativeBinomial Regression Results
==============================================================================
Dep. Variable:             bike_total   No. Observations:                  346
Model:               NegativeBinomial   Df Residuals:                      335
Method:                           MLE   Df Model:                           10
Date:                Tue, 04 Aug 2026   Pseudo R-squ.:                  0.1183
Time:                        21:44:07   Log-Likelihood:                -275.66
converged:                       True   LL-Null:                       -312.66
Covariance Type:            nonrobust   LLR p-value:                 7.425e-12
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                   -0.5542      3.491     -0.159      0.874      -7.397       6.289
C(legs_cat, Treatment(reference=4))[T.2]    -0.8734      0.513     -1.702      0.089      -1.879       0.132
C(legs_cat, Treatment(reference=4))[T.3]    -1.5643      0.346     -4.521      0.000      -2.242      -0.886
C(legs_cat, Treatment(reference=4))[T.5]     0.1036      0.485      0.214      0.831      -0.846       1.053
C(arterial_class)[T.2]                       0.8197      0.313      2.615      0.009       0.205       1.434
C(arterial_class)[T.3]                       0.0816      0.404      0.202      0.840      -0.711       0.874
C(arterial_class)[T.5]                       1.4434      0.988      1.460      0.144      -0.494       3.381
is_signalized                                1.0092      0.242      4.168      0.000       0.535       1.484
max_speed_limit                             -0.0863      0.138     -0.625      0.532      -0.357       0.184
bike_facility                               -0.6395      0.321     -1.995      0.046      -1.268      -0.011
log_bike_centrality                          0.1412      0.157      0.898      0.369      -0.167       0.449
alpha                                        1.2382      0.329      3.767      0.000       0.594       1.882
============================================================================================================

alpha = 1.2382   (overdispersed (NB correct))
Calibration: sum_pred=166.7 vs sum_actual=169 (-1.4%)
MAE: 0.58 bike crashes per intersection (2018-2023)

======================================================================
  Pedestrian model  (target: ped_total)
======================================================================
                     NegativeBinomial Regression Results
==============================================================================
Dep. Variable:              ped_total   No. Observations:                  346
Model:               NegativeBinomial   Df Residuals:                      335
Method:                           MLE   Df Model:                           10
Date:                Tue, 04 Aug 2026   Pseudo R-squ.:                  0.1707
Time:                        21:44:07   Log-Likelihood:                -346.40
converged:                       True   LL-Null:                       -417.69
Covariance Type:            nonrobust   LLR p-value:                 1.245e-25
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                   -2.6548      3.006     -0.883      0.377      -8.546       3.236
C(legs_cat, Treatment(reference=4))[T.2]    -0.7455      0.323     -2.310      0.021      -1.378      -0.113
C(legs_cat, Treatment(reference=4))[T.3]    -1.7066      0.255     -6.697      0.000      -2.206      -1.207
C(legs_cat, Treatment(reference=4))[T.5]    -0.3658      0.343     -1.066      0.286      -1.038       0.307
C(arterial_class)[T.2]                       0.5575      0.207      2.697      0.007       0.152       0.963
C(arterial_class)[T.3]                       0.0128      0.291      0.044      0.965      -0.558       0.583
C(arterial_class)[T.5]                       0.8543      0.714      1.196      0.232      -0.546       2.255
is_signalized                                1.0961      0.159      6.890      0.000       0.784       1.408
max_speed_limit                             -0.0764      0.106     -0.719      0.472      -0.285       0.132
bike_facility                               -0.6255      0.219     -2.853      0.004      -1.055      -0.196
log_aadt                                     0.2230      0.143      1.559      0.119      -0.057       0.503
alpha                                        0.3049      0.119      2.554      0.011       0.071       0.539
============================================================================================================

alpha = 0.3049   (overdispersed (NB correct))
Calibration: sum_pred=267.8 vs sum_actual=266 (+0.7%)
MAE: 0.70 ped crashes per intersection (2018-2023)

======================================================================
  Motor vehicle only model  (target: vehicle_only_total)
======================================================================
                     NegativeBinomial Regression Results
==============================================================================
Dep. Variable:     vehicle_only_total   No. Observations:                  346
Model:               NegativeBinomial   Df Residuals:                      335
Method:                           MLE   Df Model:                           10
Date:                Tue, 04 Aug 2026   Pseudo R-squ.:                  0.1097
Time:                        21:44:07   Log-Likelihood:                -745.20
converged:                       True   LL-Null:                       -837.01
Covariance Type:            nonrobust   LLR p-value:                 4.174e-34
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
Intercept                                   -0.5476      1.701     -0.322      0.748      -3.882       2.787
C(legs_cat, Treatment(reference=4))[T.2]    -0.5457      0.234     -2.336      0.019      -1.004      -0.088
C(legs_cat, Treatment(reference=4))[T.3]    -1.2627      0.146     -8.674      0.000      -1.548      -0.977
C(legs_cat, Treatment(reference=4))[T.5]     0.2633      0.266      0.990      0.322      -0.258       0.785
C(arterial_class)[T.2]                       0.1959      0.154      1.271      0.204      -0.106       0.498
C(arterial_class)[T.3]                       0.0992      0.212      0.469      0.639      -0.316       0.514
C(arterial_class)[T.5]                       0.8675      0.432      2.008      0.045       0.021       1.714
is_signalized                                0.9850      0.122      8.096      0.000       0.747       1.223
max_speed_limit                             -0.0932      0.054     -1.734      0.083      -0.198       0.012
bike_facility                               -0.3057      0.152     -2.014      0.044      -0.603      -0.008
log_aadt                                     0.2264      0.107      2.123      0.034       0.017       0.435
alpha                                        0.6585      0.083      7.911      0.000       0.495       0.822
============================================================================================================

alpha = 0.6585   (overdispersed (NB correct))
Calibration: sum_pred=1282.8 vs sum_actual=1295 (-0.9%)
MAE: 2.76 vehicle crashes per intersection (2018-2023)

======================================================================
  Cross-mode coefficient comparison  (beta; positive => more crashes)
======================================================================
                                    term   bike    ped  vehicle
                  C(arterial_class)[T.2]  0.820  0.557    0.196
                  C(arterial_class)[T.3]  0.082  0.013    0.099
                  C(arterial_class)[T.5]  1.443  0.854    0.867
C(legs_cat, Treatment(reference=4))[T.2] -0.873 -0.745   -0.546
C(legs_cat, Treatment(reference=4))[T.3] -1.564 -1.707   -1.263
C(legs_cat, Treatment(reference=4))[T.5]  0.104 -0.366    0.263
                               Intercept -0.554 -2.655   -0.548
                                   alpha  1.238  0.305    0.659
                           bike_facility -0.639 -0.625   -0.306
                           is_signalized  1.009  1.096    0.985
                                log_aadt    NaN  0.223    0.226
                     log_bike_centrality  0.141    NaN      NaN
                         max_speed_limit -0.086 -0.076   -0.093

Wrote predictions -> C:\Users\jfbaa\project-cycle-group\data\intermediate\intersection_predictions.parquet
Saved model        -> C:\Users\jfbaa\project-cycle-group\data\model\nb_v3_bike.pkl
Saved model        -> C:\Users\jfbaa\project-cycle-group\data\model\nb_v3_ped.pkl
Saved model        -> C:\Users\jfbaa\project-cycle-group\data\model\nb_v3_vehicle.pkl
```

### Derived metrics (computed by me from the regenerated artifacts, not printed by any repo script)

| mode | α | sum_pred | sum_actual | % | MAE/site | RMSE/site | Spearman ρ | 90% crash coverage | 90% KSI-proxy coverage | citywide KSI share |
|---|---|---|---|---|---|---|---|---|---|---|
| bike | 1.2382 | 166.7 | 169 | −1.37% | 0.577 | 1.046 | +0.425 | 96.5% | 99.1% | 9.47% |
| ped | 0.3049 | 267.8 | 266 | +0.67% | 0.698 | 1.050 | +0.590 | 97.1% | 98.0% | 12.03% |
| vehicle | 0.6585 | 1282.8 | 1295 | −0.94% | 2.764 | 4.294 | +0.609 | 94.8% | 98.8% | 1.93% |

Coverage computed with the same estimator as
[`test_calibration.py`](pipeline/tests/test_calibration.py:33). Spearman is
actual-vs-predicted **crash counts**, which is not the same quantity as the
README's "predicted vs observed bike KSI" ρ.

Sum of the three mode predictions: **1717.2** vs sum of the three mode actuals
**1730** (−0.74%). Note 1730 ≠ 1720: the three targets double-count crashes
flagged as both bike and ped, acknowledged at
[`score_risk.py:14-18`](pipeline/score_risk.py:14).

---

## 2. Model version history from git

Only **three** commits have ever touched `pipeline/fit_risk_model.py`:

```
$ git log --oneline --all -- pipeline/fit_risk_model.py
c72f1ec Initial commit: Vision Zero intersection risk model
2617e1a Add Vision Zero scorecard, counterfactual modeling, analytical explainer
152106e Prompt 4: fit NB risk model (nb_v1_no_aadt) — converged, calibrated +6.4%, alpha sig.
```

| version | commit | date | shape | target(s) | volume term | leg term | offset |
|---|---|---|---|---|---|---|---|
| **v1** `nb_v1_no_aadt` | `152106e` | 2026-05-17 | 1 pooled model, 651 rows | `total_crashes` | **none** | `num_legs` continuous | `log(years_observed)` |
| (no spec change) | `2617e1a` | 2026-05-20 | n/a | n/a | n/a | n/a | n/a |
| **v2** `nb_v2_arterial_aadt` | *no commit of its own* | n/a | 1 pooled model, 346 rows | `total_crashes` | `log_aadt` | `num_legs` continuous | `log(years_observed)` |
| **v3** `nb_v3_{bike,ped,vehicle}` | `c72f1ec` | 2026-06-02 | **3 per-mode models**, 346 rows | `bike_total`, `ped_total`, `vehicle_only_total` | `log_bike_centrality` (bike) / `log_aadt` (ped, vehicle) | `legs_cat` top-coded categorical | `log(years_observed)` |

### What actually changed, and the stated whys

**v1 → v2: add AADT. Reason recorded, and it is a data-availability reason, not a
model-comparison reason.** The v1 docstring (`git show 152106e:pipeline/fit_risk_model.py`)
carries an explicit block:

> `KNOWN LIMITATION. No AADT exposure term:` … "not available in usable form in
> this dataset" … "the available AADT layer produced near-zero spatial coverage
> over Capitol Hill intersections" … "This is a documented MVP compromise."

Corroborated by commit `3890494`, *"Prompt 3: assemble feature matrix (5
features clean; AADT dropped, documented MVP limitation)"*. So v2 exists because
the AADT join was fixed, **not** because anyone measured v1 against v2.

**`2617e1a` is not a spec change at all.** The full diff to `fit_risk_model.py`
is four lines that carry severity sub-counts (`injury_total`, `ksi_total`,
`fatal_total`, `ped_total`, `bike_total`) through to the output frame for the
downstream scorecard. The formula is untouched.

**v2 → v3: no recorded reason whatsoever.** `c72f1ec` is a squashed rewrite of
the entire project. Its message is:

> Initial commit: Vision Zero intersection risk model
>
> Per-mode Negative Binomial SPFs (bike/ped/vehicle) with AASHTO HSM
> Empirical-Bayes shrinkage, FHWA CMF-backed treatment recommendations, a
> FastAPI backend, and a Next.js + Mapbox frontend.

That describes *what* v3 is. It does not say why the pooled model was replaced,
and there is no other commit that does. `git log --all --grep` across
v1/v2/v3/poisson/offset/aadt/spec/compare/ablat returns only the three commits
already listed above.

**v2 has no source commit.** `nb_v2_arterial_aadt.pkl` was committed as a binary
in `c72f1ec` alongside the v3 pickles. The v2 *code* never existed in git. Its
specification survives only as prose in `README.md`
([:139](README.md:139) onward). That is why the README describes a model the
pipeline no longer fits.

### Recorded v1 numbers (real, verified)

`.baseline/` holds v1's actual outputs, which I read directly:

- `model_version = nb_v1_no_aadt`, `fitted_at = 2026-05-21T19:58:51Z`, 651 rows
- sum expected 2017.8 vs sum actual 1896 → **+6.42%**, matching the `152106e`
  commit message "calibrated +6.4%"
- MAE 2.276 crashes/intersection

This is the only version whose headline numbers are independently recorded in
the repo rather than asserted in prose.

---

## 3. Comparison evidence table

| Decision | Evidence found? | Numbers | Source |
|---|---|---|---|
| **log(AADT) vs raw AADT** | **Justification only.** No raw-AADT fit was ever run or recorded. | Fitted β≈0.26 → 2^0.26 ≈ 1.20 per AADT doubling. This is a *property of the chosen model*, not a comparison against the alternative. No AIC/LL/MAE for a raw-AADT spec. | [README.md:125-140](README.md:125). Argument is functional-form + HSM-conformance: raw AADT under a log link gives μ ∝ exp(β·AADT), "not physical". |
| **Negative Binomial vs Poisson** | **Partial evidence, a formal test, not a head-to-head fit.** α is estimated and its significance reported per mode; α>0 rejects Poisson. But no Poisson model is fit, and there is no side-by-side AIC/LL. | bike α=1.2382 (SE 0.329, z=3.767, p<0.001); ped α=0.3049 (SE 0.119, z=2.554, p=0.011); vehicle α=0.6585 (SE 0.083, z=7.911, p<0.001). | Fit output §1. Verdict string at [fit_risk_model.py:363](pipeline/fit_risk_model.py:363), `"overdispersed (NB correct)" if alpha > 0.05 else "near-Poisson"`. |
| **offset=log(years_observed) vs covariate** | **Nothing, and the comparison is not identifiable in this data.** | `years_observed == 6` for **all 651 rows** (verified). A constant covariate is collinear with the intercept, so offset-vs-covariate cannot be distinguished here. | [fit_risk_model.py:142](pipeline/fit_risk_model.py:142); constancy asserted at [:145](pipeline/fit_risk_model.py:145) and confirmed in the run output. Stated explicitly at [hierarchical_nb_sketch.py:32-34](experiments/hierarchical_nb_sketch.py:32): "Exposure … is a constant 6 across all sites, so the offset only shifts the intercept." |
| **log_bike_centrality vs log_aadt in the bike model** | **Nothing.** Not even a stated justification in `fit_risk_model.py`. The swap is made silently in the `MODES` tuple. | Only indirect: in the shipped bike fit `log_bike_centrality` is **not significant** (β=0.141, SE 0.157, p=0.369), while `log_aadt` is significant in the vehicle fit (p=0.034) and not in the ped fit (p=0.119). **No bike model using `log_aadt` was ever fit**, so this is not a comparison. | [fit_risk_model.py:87](pipeline/fit_risk_model.py:87) vs [:88-89](pipeline/fit_risk_model.py:88). Centrality is OSM betweenness centrality, [build_bike_exposure.py:48](pipeline/build_bike_exposure.py:48). |
| **direct KSI-target fits vs total-crash fits** | **Justification only, but with real supporting counts.** Rejected a priori on power grounds; no KSI fit was attempted. | "bike-KSI = 16, ped-KSI = 32, vehicle-only-KSI = 23; all below the ~10 events / parameter threshold for stable NB MLE." I verified: bike 16 ✓, ped 32 ✓, **vehicle-only 25, not 23** ✗. Models have 11 parameters + α. | [fit_risk_model.py:11-15](pipeline/fit_risk_model.py:11). |
| **one pooled all-crash model vs three per-mode models** | **No recorded comparison.** Both fits exist as artifacts (`nb_v2_arterial_aadt.pkl` and the three `nb_v3_*.pkl` are all committed), so a comparison is *reconstructible*, but nobody recorded one, and no commit message, docstring, test, or note compares them. | v2 (from stale README): 1772.7 vs 1720 (+3.1%), MAE 3.47, coverage 95.1%/98.6%, ρ=+0.28. v3 (from my re-fit): per-mode −1.4%/+0.7%/−0.9%, MAE 0.58/0.70/2.76. **These are not comparable**, different targets and different denominators. | README.md:162-167 for v2; §1 above for v3. |

### Everything else the grep turned up

`rg -i 'poisson|pseudo.?r2|aic|loglik|rmse|mae|spearman|baseline|compare|instead of|rejected|we tried|ablat'` over the whole repo (excluding `frontend/`): **31 matches across 6 files**, and essentially none of them are recorded comparison output.

- No results JSON, no results CSV, no metrics file, no notebook. There are **no notebooks in this repo at all.**
- `data/intermediate/` contains only pipeline parquet outputs. No saved comparison runs.
- No commented-out alternative model specs anywhere.
- `pipeline/counterfactual.py:61` `def compare(...)` is a *treatment* counterfactual (what-if on features), unrelated to model selection.
- `pipeline/tests/` contains one calibration test asserting coverage ≥ 85%. It is a threshold guard, not a comparison, and it records no numbers.

**One genuine rejected-alternative-with-numbers exists**, and it is not on your
list: the **continuous per-leg slope**, rejected in favour of a top-coded
categorical, at [`feature_encoding.py:8-16`](pipeline/feature_encoding.py:8):

> "fit on Capitol Hill, where 2-to-4-leg sites are 97% of the data, the slope
> reads a 6-leg intersection as roughly +280% over a 4-leg one, with a credible
> interval spanning +80% to +700% and no six-leg site actually supporting it."

That is the single best-documented modeling tradeoff in the repository. If you
want one honestly-earned "we tested the alternative and rejected it" beat for
the case study, this is the one, though note even here the +280% figure is
asserted in a docstring, with no saved run behind it.

---

## 4. The hierarchical NB experiment

[`experiments/hierarchical_nb_sketch.py`](experiments/hierarchical_nb_sketch.py)

**What it explores:** a partial-pooling (Bayesian hierarchical) NB in PyMC as an
alternative to the production fixed-effects NB. `arterial_class` and `num_legs`
become random effects shrunk toward the global mean in proportion to how little
data supports each level, instead of treatment-coded fixed effects
([:12-21](experiments/hierarchical_nb_sketch.py:12)). It also feeds `num_legs`
in **raw** (levels 2..6) on the argument that adaptive shrinkage makes the manual
"5+" top-coding unnecessary, i.e. it is a direct alternative to the one
tradeoff §3 documents. Bike-only (`TARGET = "bike_total"`).

**Adopted or abandoned:** neither, strictly. It is **abandoned in place**. Line
1 says "NOT wired into the pipeline". Nothing imports it. It is not in
`requirements.txt` (needs `pymc>=5`, `arviz`, optionally `bambi`, none
installed here). It appears only in the squashed `c72f1ec`.

**Results in comments:** **none recorded.** There is no saved trace, no summary
output, no divergence count, no R̂, no comparison to `nb_v3_bike`. The only
forward-looking note is an instruction about what to look for *if you run it*
([:213-214](experiments/hierarchical_nb_sketch.py:213)): "Watch: arterial_class
5 (n=19) and num_legs 6 (n=3) shrink toward 0% with WIDE intervals;
arterial_class 2 (n=185) holds its effect with a tight one." Those are sample
sizes, not results.

It does contain one cross-reference to the production fit: "fitted disp ~= 1.238
for bike" ([:25](experiments/hierarchical_nb_sketch.py:25)), which matches
`nb_v3_bike` α=1.2382 exactly. So the sketch was written against v3, after the
per-mode rewrite. **Do not describe this as a tested alternative.** It is a
runnable specification that was never run, or at least never recorded.

---

## 5. Current architecture, claim by claim

### (a) "a negative binomial (NB2) regression with a log link", one model, all crashes, **OUTDATED**

Correct on family and link; wrong on almost everything else.

- **Three models, not one**, defined at [fit_risk_model.py:86-90](pipeline/fit_risk_model.py:86).
- **Not fit on all crashes.** Targets are `bike_total`, `ped_total`,
  `vehicle_only_total`. `total_crashes` is not a target anywhere in v3.
- **Traffic volume is not a shared predictor.** The bike model uses
  `log_bike_centrality` (OSM betweenness centrality of the bike network);
  ped and vehicle use `log_aadt` ([:87-89](pipeline/fit_risk_model.py:87)).
- **"Number of legs" is no longer a continuous count.** It is a top-coded
  categorical `legs_cat` (5, 6, … collapsed to "5+"), reference level 4 legs
  ([feature_encoding.py:21-32](pipeline/feature_encoding.py:21)).
- Confirmed as stated: NB2 via `smf.negativebinomial`, log link, `offset =
  log(years_observed)` ([:142](pipeline/fit_risk_model.py:142),
  [:214](pipeline/fit_risk_model.py:214)); `is_signalized`, `max_speed_limit`,
  `bike_facility`, `C(arterial_class)` are shared across all three
  ([:72-75](pipeline/fit_risk_model.py:72)).

### (b) "cyclist KSI … Poisson-Gamma EB … all-crash prediction × citywide bike KSI share as prior", **OUTDATED** (one word wrong, but it matters)

The method is right; the input is not. The prior is built from the **bike-crash
NB prediction**, not an all-crash prediction:

- `mu_ksi = predictions[mode.crash_predicted] * city_share` where
  `crash_predicted` is `bike_expected_total`
  ([score_risk.py:207](pipeline/score_risk.py:207), mode table at
  [:84](pipeline/score_risk.py:84)).
- `city_share = bike_ksi_total / bike_actual` summed across sites
  ([:188-193](pipeline/score_risk.py:188)) = **16/169 = 9.47%**.
- Posterior is `Gamma(k + N_KSI, scale = μ_KSI/(k + μ_KSI))` with `k = 1/α`,
  90% credible interval ([:210-214](pipeline/score_risk.py:210)), so
  "Poisson-Gamma direct EB" is CONFIRMED.

Since v3 there is no "all-crash prediction" in the pipeline to scale.

### (c) "Empirical Bayes shrinkage (HSM Part C) pulls extreme predictions back toward observed counts", **CONFIRMED**

[`score_risk.py:148-159`](pipeline/score_risk.py:148): `w = 1/(1 + α·μ)`,
`eb = w·μ + (1−w)·N`. Applied per mode with that mode's own α
([:325](pipeline/score_risk.py:325)).

One refinement worth making in the copy: EB is applied **twice, to two different
quantities**, once to the mode crash count (above), and again as the separate
direct Poisson-Gamma EB on the mode KSI count in (b). They are distinct steps.

### (d) "eight CMFs … approved, bike-involved, intersection studies", **PARTIALLY OUTDATED**

`data/cmf_library.json` contains **11 treatments**, not 8:

| applies_to | count |
|---|---|
| bike | **8** |
| ped | 1 (`lpi_signal_timing`) |
| vehicle | 2 (`road_diet`, `prohibit_right_turn_on_red_vehicle`) |

So "eight" is exactly right *for the bike subset* and wrong for the library. The
filter is also no longer bike-only: the crash-type filter is applied **per mode**
([build_cmf_library.py:415-417](pipeline/build_cmf_library.py:415)), while
`approved == yes`, intersection-related, and non-rural are applied to all
([:211-215](pipeline/build_cmf_library.py:211)). Clearinghouse export date
2025-11-10.

Also note the README's own CMF table ([README.md:188-196](README.md:188)) lists
only the 8 bike rows, another symptom of the same staleness.

### (e) "1,772.7 vs 1,720 (within 3.1%)", "MAE 3.47", "95 to 99% coverage", **OUTDATED**

All three are **v2 numbers**, copied from [README.md:162-166](README.md:162).
The current model does not produce any of them.

| claim | status | current fact |
|---|---|---|
| predicts 1,772.7 total crashes | **OUTDATED** | No single total is produced. Three calibrations: bike 166.7 vs 169, ped 267.8 vs 266, vehicle 1282.8 vs 1295. Summed: **1717.2 vs 1730 (−0.74%)**. |
| against 1,720 observed, within 3.1% | **mixed** | 1,720 is still correct as `total_crashes` over the 346 modelled sites. But it is *not* what v3 predicts against (the three mode targets sum to 1,730 because bike+ped crashes are double-counted, [score_risk.py:14-18](pipeline/score_risk.py:14)). "+3.1%" is dead; current gaps are −1.4% / +0.7% / −0.9%. |
| mean error 3.47 crashes per intersection over six years | **OUTDATED** | Per-mode MAE: **bike 0.58, ped 0.70, vehicle 2.76**. Pooled mean across the three modes is 1.35. Nothing produces 3.47. |
| 95 to 99% predictive coverage | **roughly right, but restate** | Crash-count coverage **94.8 to 97.1%** (vehicle is 94.8%, just under 95); KSI-proxy coverage **98.0 to 99.1%**. Safe phrasing: "94.8 to 99.1%". Nominal is 90%; over-coverage comes from discrete NB quantiles. |

Reminder: these are from my re-fit; the committed pickles would not load.

### (f) "346 arterial intersections", "1,720 reported crashes", "16 cyclists KSI", 2018 to 2023, **CONFIRMED** (all four)

- **346**, fit output "rows used for fit 346"; 651 raw → 286 non-arterial
  dropped, 19 missing AADT ([fit_risk_model.py:167-179](pipeline/fit_risk_model.py:167)).
- **1,720**, `total_crashes` summed over those 346 sites (verified directly).
  Over all 651 intersections it is 1,896.
- **16 bike KSI**, `bike_ksi_total` over the 346 modelled sites. Over all 651
  it is **17**. Both figures appear in the README, which is internally
  inconsistent about it, see §6.
- **2018 to 2023**, `YEAR_MIN = 2018`, `YEAR_MAX = 2023`
  ([snap_crashes.py:34-35](pipeline/snap_crashes.py:34)); `years_observed = 6`
  for every row.

---

### Plain statement of the current architecture

**Three NB2 models are fit today** ([fit_risk_model.py:86-90](pipeline/fit_risk_model.py:86)),
all on the same 346 arterial intersections, all with log link and
`offset = log(years_observed)` (constant 6):

| model | target | volume/exposure predictor | shared predictors | offset |
|---|---|---|---|---|
| `nb_v3_bike` | `bike_total` (169 events) | `log_bike_centrality` | `is_signalized`, `C(legs_cat, ref=4)`, `max_speed_limit`, `bike_facility`, `C(arterial_class)` | `log(years_observed)` |
| `nb_v3_ped` | `ped_total` (266 events) | `log_aadt` | same | same |
| `nb_v3_vehicle` | `vehicle_only_total` (1295 events) | `log_aadt` | same | same |

**How bike KSI risk is ultimately derived** (four steps, all in
[`score_risk.py`](pipeline/score_risk.py)):

1. `nb_v3_bike` predicts expected **bike crashes** per site over 6 years
   (`bike_expected_total`).
2. HSM Part C EB shrinkage on that crash count: `w = 1/(1 + α_bike·μ)`,
   `eb = w·μ + (1−w)·N_observed` ([:148-159](pipeline/score_risk.py:148)).
   *This produces `bike_eb_count`; note it is **not** what feeds step 3.*
3. Bike-KSI prior: `μ_KSI = bike_expected_total × 9.47%`, the citywide bike KSI
   share (16/169) ([:196-207](pipeline/score_risk.py:196)). Step 3 consumes the
   **raw NB prediction**, not the EB-shrunk count from step 2.
4. Direct Poisson-Gamma EB on the observed bike KSI count:
   `Gamma(k + N_KSI, μ_KSI/(k + μ_KSI))`, `k = 1/α_bike`, giving posterior mean
   plus a 90% credible interval, divided by 6 for
   `expected_bike_ksi_per_year` ([:210-221](pipeline/score_risk.py:210)).

Ped and vehicle KSI follow the identical chain with their own α and share
(12.03% and 1.93%). A composite `all_ksi_per_year` is the sum of the three
posteriors ([:229-244](pipeline/score_risk.py:229)), overstated ~2.8% by
bike/ped double-flagging. The frontend headline and `risk_score` / `risk_rank` /
`risk_tier` are still ranked on **bike** KSI percentile, not the composite
([:300](pipeline/score_risk.py:300), noted at [:456-461](pipeline/score_risk.py:456)).

---

## 6. Things you asserted that the code contradicts

1. **"One model fit on all crashes."** Three models, one per mode, none of which
   targets total crashes. This is the biggest correction, and it invalidates the
   framing of claims (a), (b), and (e) together.
2. **"Predictors include traffic volume."** Only for ped and vehicle. The bike
   model, the one your headline metric comes from, uses OSM bike-network
   betweenness centrality instead, and that coefficient is not statistically
   significant (p=0.369).
3. **"Number of legs" as a plain predictor.** It is a top-coded categorical with
   a 4-leg reference, not a count.
4. **"The all-crash prediction scaled by the citywide bike KSI share."** It is
   the bike-crash prediction. There is no all-crash prediction in the pipeline.
5. **1,772.7 / 3.1% / 3.47.** None of these are produced by the current model.
   They describe `nb_v2_arterial_aadt`, which is not fit by any code on `main`.
6. **"Eight CMFs."** Eleven in the library; eight of them bike.
7. **"95 to 99% coverage."** Lower bound is 94.8%. Minor, but you asked for exact.

### Additional discrepancies I found that you did not assert (fix these too)

8. **`README.md` is the root cause.** It documents v2 as current, formula at
   [:113-117](README.md:113) is `total_crashes ~ … + num_legs + … + log_aadt`,
   headline description at [:119](README.md:119) says "all crash NB prediction",
   and the coefficient table at [:139](README.md:139) is explicitly labelled
   "fit `nb_v2_arterial_aadt`". Your portfolio copy is faithful to the README
   and wrong about the code. Fixing the README should probably precede the
   portfolio rewrite.
9. **`pipeline/evaluate_models.py` is broken and has never run** (§1). If the
   case study mentions model diagnostics, VIFs, or pseudo-R², there is no
   working script behind that claim.
10. **The committed model pickles cannot be deserialized** by the environment
    the repo describes. `requirements.txt` pins no versions. Anyone cloning this
    repo, a hiring manager included, cannot load the shipped models without
    re-fitting.
11. **Docstring count error:** [fit_risk_model.py:14](pipeline/fit_risk_model.py:14)
    says "vehicle-only-KSI = 23"; the data says **25**. The argument it supports
    (too few events for a direct KSI fit) is unaffected.
12. **README is internally inconsistent on bike KSI:** 16 at [:20](README.md:20)
    and [:35](README.md:35), 17 at [:112](README.md:112) and [:169](README.md:169).
    Both are real, 16 over the 346 modelled sites, 17 over all 651, but the
    README never distinguishes them. Use 16 when talking about the model.
13. **The EB chain has a seam worth knowing before you describe it:** step 3
    consumes the *raw* NB prediction, while the EB-shrunk crash count computed in
    step 2 is emitted as `{mode}_eb_count` but never feeds the KSI posterior. If
    the copy implies shrinkage flows through into the KSI number, that is not
    what the code does.

### Not recorded, do not write around these

- Any raw-AADT fit, or any metric for one.
- Any Poisson fit, or any NB-vs-Poisson AIC/LL comparison.
- Any offset-vs-covariate comparison (not identifiable: exposure is constant 6).
- Any bike model using `log_aadt`, hence no centrality-vs-AADT comparison.
- Any direct KSI-target fit.
- Any pooled-vs-per-mode comparison.
- Any output, trace, or result from the hierarchical NB sketch.
- Any out-of-sample validation or cross-validation. Explicitly declared out of
  scope at [evaluate_models.py:14-16](pipeline/evaluate_models.py:14), "a Phase
  8 capability". **Every number in this document is in-sample.**
