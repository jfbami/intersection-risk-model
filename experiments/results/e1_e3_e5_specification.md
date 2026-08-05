# E1 / E3 / E5, Specification A/B experiments

Vision Zero intersection crash-risk model. All numbers below were produced by code in
`experiments/ab/` run against the 346-row arterial fit set. **No figure in this document
is estimated, rounded from memory, or invented.** Where a fit failed or a comparison is
inconclusive, that is stated as the result.

- Fit set: 346 arterial rows (`arterial_class >= 1`, `max_aadt > 0`)
- Events: `bike_total` 169, `ped_total` 266, `vehicle_only_total` 1295
- Offset: `log(years_observed)`, constant `log(6) = 1.7917594` for every row
- Family: statsmodels `NegativeBinomial` (NB2), log link
- Environment: Python 3.13.2, pandas 2.2.3, numpy 2.1.3, statsmodels 0.14.6, scipy 1.15.1,
  patsy 1.0.2, scikit-learn 1.6.1 (present, `KFold`/`StratifiedKFold` used directly)

Nothing under `pipeline/` or `data/` was modified. `pipeline.fit_risk_model.main()` was
never called; only `load_and_join`, `prepare`, `MODES`, `SHARED_PREDICTORS` were imported.

---

## Commands run

```bash
cd C:/Users/jfbaa/project-cycle-group

# data probe (row counts, distributions, dtypes)
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e1_probe.py

# harness validation: robust fitter must reproduce the production fit exactly
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e1_sanity.py

# E1 optimizer diagnostic (see "Methodological finding" below)
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e1_raw_aadt_diagnostic.py

# the three experiments
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e1_volume_form.py
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e3_bike_exposure.py
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e5_leg_encoding.py

# merge JSON + print the secondary plain-KFold table
PYTHONPATH=C:/Users/jfbaa/project-cycle-group python experiments/ab/e1_e3_e5_combine.py
```

Outputs: `experiments/results/e1_results.json`, `e3_results.json`, `e5_results.json`,
`e1_raw_aadt_diagnostic.json`, and the merged `e1_e3_e5_results.json`.

---

## Methodology

### Cross-validation

Every spec is evaluated three ways, all with the model **refit from scratch on each
training split** and predicted on the held-out split using that split's offset:

1. **Stratified 5-fold**, `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`,
   strata = held-out count bins `{0, 1, 2 to 3, 4+}` of the mode's target. This is the
   headline metric (crash counts are heavily zero-inflated: 256 of 346 sites have zero
   bike crashes, so unstratified folds vary a lot in event mass).
2. **Plain 5-fold**, `KFold(n_splits=5, shuffle=True, random_state=0)`, exactly as
   specified in the brief. Reported as a secondary table.
3. **Repeated stratified 5-fold over 10 seeds (0 to 9)**, gives a seed-to-seed SD, which is
   a much tighter noise floor than the 5-fold SD and is what the "decisive vs within
   noise" verdicts are judged against.

Reported per spec: fold-mean ± SD of MAE and RMSE, the **pooled** out-of-fold MAE / RMSE /
Spearman ρ (all 346 held-out predictions concatenated), and the repeated-CV mean ± SD.

The patsy design matrix is built **once on the full frame** so the categorical level set is
identical across folds; only coefficients are estimated per fold. Columns constant within a
training split are dropped for that split and flagged. **No fold ever hit this**, see the
E5 sparsity check.

### Methodological finding: the production optimizer ladder silently diverges

This is not incidental. It changed E1's answer, so it is documented rather than buried.

`smf.negativebinomial(...).fit(disp=False)` (newton) **fails to converge on all three
production specs**; `pipeline.fit_risk_model.fit_for_mode` catches this and retries with
BFGS, which does converge. Fine so far. But when a predictor is on a raw `1e3..4e4` scale
(spec E1-b, raw AADT), *both* newton and BFGS land on a diverged optimum:

| `ped_total ~ SHARED + max_aadt`, optimizer | converged | log-likelihood | AIC | β per AADT | SE |
|---|---|---:|---:|---:|---:|
| newton (default) | **False** | −404.2509 | 832.50 | 1.445686e-04 | **nan** |
| bfgs maxiter=200 | **False** | −404.2509 | 832.50 | 1.445686e-04 | **nan** |
| bfgs maxiter=5000 | **False** | −404.2509 | 832.50 | 1.445686e-04 | **nan** |
| lbfgs maxiter=5000 | True | −404.2509 | 832.50 | 1.445686e-04 | **nan** |
| **nm maxiter=10000** | **True** | **−347.2664** | **718.53** | **1.285865e-05** | 1.49044e-05 |
| rescaled `max_aadt/10000`, bfgs | True | −347.2664 | 718.53 | 1.285924e-05 | (=1.49044e-05) |

Rescaling AADT by 10,000 is an exact reparameterisation, identical model, identical
likelihood. And it converges to the same optimum Nelder-Mead finds. So the "failure" is
**numeric conditioning, not model failure**, and the naive answer is wrong by **114 AIC
points**. A first pass of E1 using the production-style ladder reported raw-AADT AIC 832.50
and CV MAE 0.9163 for ped; both were artefacts and are discarded.

`lbfgs` does the same thing to the vehicle raw-AADT spec (llf −1413.2767 vs the true
−745.8095), so this is not a one-off.

**The harness therefore (a) column-scales the design matrix by its training SD, (b) tries
newton / bfgs / lbfgs / Nelder-Mead, and (c) keeps the converged solution with the highest
log-likelihood**, un-scaling coefficients and standard errors afterwards. This is applied
identically to every spec in E1/E3/E5 so comparisons stay fair.

**Validation** (`e1_sanity.py`). The robust fitter reproduces the production fit exactly on
all three production specs:

| mode | production llf / AIC | robust llf / AIC | max abs pred diff | optimizer chosen |
|---|---|---|---:|---|
| bike | −275.656447 / 575.3129 | −275.656447 / 575.3129 | 5.039e-05 | bfgs |
| ped | −346.401704 / 716.8034 | −346.401704 / 716.8034 | 3.986e-04 | bfgs |
| vehicle | −745.202238 / 1514.4045 | −745.202238 / 1514.4045 | 3.253e-04 | newton |

The production bike centrality coefficient is reproduced as β = 0.141156, SE = 0.157275,
p = 0.3694, matching the brief's β=0.141, SE=0.157, p=0.369.

**Every fit reported below converged.** No spec in E1, E3 or E5 failed under the robust
fitter, and the E5 spec-(c) instability that was anticipated did not materialise.

---

# E1, Volume functional form

### Stated reason

`README.md:125-140` argues for `log(AADT)` purely on functional-form theory: raw AADT under
a log link implies μ ∝ exp(β·AADT), which the README calls "not physical" and claims would
give "astronomical predictions" at a 50,000-AADT site. No alternative was ever fit. This
experiment converts that assertion into a measurement.

### Specs compared

Base for all four (`SHARED_PREDICTORS`):
`is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit + bike_facility + C(arterial_class)`

| spec | volume term |
|---|---|
| a_log_aadt (production) | `+ log_aadt` |
| b_raw_aadt | `+ max_aadt` |
| c_sqrt_aadt | `+ sqrt_aadt` (`np.sqrt(max_aadt)`, added column) |
| d_no_volume | *(none)* |

### Results, ped (`ped_total`, 266 events)

| spec | conv | opt | LL | AIC | BIC | CV MAE (strat) | CV RMSE (strat) | pooled OOF MAE | pooled OOF RMSE | ρ | repeated-CV OOF MAE (10 seeds) |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| a_log_aadt | yes | bfgs | −346.40 | **716.80** | 762.96 | 0.7158 ± 0.0817 | 1.0690 ± 0.1677 | 0.7157 | 1.0792 | 0.562 | 0.7232 ± 0.0095 |
| b_raw_aadt | yes | nm | −347.27 | 718.53 | 764.69 | 0.7151 ± 0.0802 | 1.0695 ± 0.1664 | 0.7150 | 1.0796 | 0.560 | 0.7250 ± 0.0116 |
| c_sqrt_aadt | yes | nm | −346.88 | 717.75 | 763.91 | 0.7147 ± 0.0814 | 1.0686 ± 0.1683 | 0.7146 | 1.0789 | 0.561 | 0.7234 ± 0.0107 |
| d_no_volume | yes | nm | −347.64 | 717.27 | **759.58** | **0.7140 ± 0.0817** | **1.0636 ± 0.1661** | **0.7139** | **1.0737** | 0.553 | **0.7222 ± 0.0087** |

Volume coefficients (ped):

| spec | term | β | SE | p | 95% CI |
|---|---|---:|---:|---:|---|
| a | `log_aadt` | 0.222994 | 0.143039 | **0.1190** | [−0.0574, 0.5033] |
| b | `max_aadt` | 1.28589e-05 | 1.49044e-05 | 0.3883 | [−1.635e-05, 4.207e-05] |
| c | `sqrt_aadt` | 0.00379961 | 0.00308265 | 0.2177 | [−0.00224, 0.00984] |

### Results, vehicle (`vehicle_only_total`, 1295 events)

| spec | conv | opt | LL | AIC | BIC | CV MAE (strat) | CV RMSE (strat) | pooled OOF MAE | pooled OOF RMSE | ρ | repeated-CV OOF MAE (10 seeds) |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| a_log_aadt | yes | newton | −745.20 | **1514.40** | 1560.56 | 2.9069 ± 0.1621 | 4.4994 ± 0.6986 | 2.9067 | **4.5424** | 0.580 | **2.8946 ± 0.0198** |
| b_raw_aadt | yes | newton | −745.81 | 1515.62 | 1561.78 | 2.9128 ± 0.1647 | 4.5139 ± 0.7437 | 2.9128 | 4.5624 | 0.578 | 2.8951 ± 0.0241 |
| c_sqrt_aadt | yes | newton | −745.52 | 1515.03 | 1561.19 | 2.9140 ± 0.1622 | 4.5060 ± 0.7112 | 2.9139 | 4.5504 | 0.579 | 2.8964 ± 0.0219 |
| d_no_volume | yes | newton | −747.43 | 1516.86 | **1559.17** | **2.8878 ± 0.1990** | 4.5229 ± 0.8068 | **2.8879** | 4.5798 | 0.579 | 2.8960 ± 0.0276 |

Volume coefficients (vehicle):

| spec | term | β | SE | p | 95% CI |
|---|---|---:|---:|---:|---|
| a | `log_aadt` | 0.226367 | 0.106598 | **0.0337** | [0.01744, 0.43530] |
| b | `max_aadt` | 1.98220e-05 | 1.10979e-05 | 0.0741 | [−1.930e-06, 4.157e-05] |
| c | `sqrt_aadt` | 0.00453221 | 0.00231521 | 0.0503 | [−5.517e-06, 0.00907] |

### Secondary CV, plain `KFold(n_splits=5, shuffle=True, random_state=0)`

| spec | ped OOF MAE | ped OOF RMSE | ped ρ | vehicle OOF MAE | vehicle OOF RMSE | vehicle ρ |
|---|---:|---:|---:|---:|---:|---:|
| a_log_aadt | **0.7271** | 1.1044 | 0.541 | 2.8861 | 4.4717 | 0.579 |
| b_raw_aadt | 0.7298 | 1.1077 | 0.536 | **2.8828** | **4.4484** | 0.574 |
| c_sqrt_aadt | 0.7273 | 1.1054 | 0.540 | 2.8867 | 4.4578 | 0.578 |
| d_no_volume | 0.7300 | **1.1026** | 0.536 | 2.9248 | 4.5811 | 0.572 |

**The KFold ranking disagrees with the stratified ranking.** Under stratified CV `d_no_volume`
has the lowest vehicle MAE (2.8879); under plain KFold it has the *highest* (2.9248) and
`b_raw_aadt` wins. For ped, stratified ranks `d` best and KFold ranks `d` worst on MAE. A
ranking that inverts when you change the fold scheme is not a ranking.

### Direct test of the README's 50,000-AADT claim

Other predictors held at dataset mode / median: `is_signalized=0`, `legs_cat=4`,
`num_legs=4`, `max_speed_limit=25.0`, `bike_facility=0`, `arterial_class=2`,
`offset=log(6)`. Values are expected crash counts over the full 6-year window.

**Observed `max_aadt` range in the fit data: 1,013 to 41,808** (mean 10,021.97, median
8,051.5, SD 5,883.14, q95 20,443, q99 22,541).
Sites above 20,000: 24. Above 25,000: 5. Above 30,000: **1**. Above 40,000: **1**.
**Above 50,000: 0.**

ped:

| AADT | a_log | b_raw | c_sqrt | d_none | raw ÷ log |
|---:|---:|---:|---:|---:|---:|
| 8,052 (median) | 0.8103 | 0.7970 | 0.8012 | 0.8003 | 0.98× |
| 20,443 (q95) | 0.9974 | 0.9346 | 0.9809 | 0.8003 | 0.94× |
| 41,808 (observed max) | 1.1699 | 1.2301 | 1.2391 | 0.8003 | 1.05× |
| **50,000 (README's case)** | **1.2176** | **1.3668** | 1.3325 | 0.8003 | **1.12×** |
| 100,000 (2.4× beyond data) | 1.4211 | 2.5998 | 1.8946 | 0.8003 | 1.83× |

vehicle:

| AADT | a_log | b_raw | c_sqrt | d_none | raw ÷ log |
|---:|---:|---:|---:|---:|---:|
| 8,052 (median) | 3.1478 | 3.0835 | 3.1102 | 3.0971 | 0.98× |
| 20,443 (q95) | 3.8869 | 3.9421 | 3.9592 | 3.0971 | 1.01× |
| 41,808 (observed max) | 4.5703 | 6.0207 | 5.2316 | 3.0971 | 1.32× |
| **50,000 (README's case)** | **4.7592** | **7.0822** | 5.7057 | 3.0971 | **1.49×** |
| 100,000 (2.4× beyond data) | 5.5677 | 19.0808 | 8.6819 | 3.0971 | 3.43× |

### Verdict, E1

**The README's central claim is quantitatively wrong, and the choice it defends does not
matter out of sample.**

- At 50,000 AADT the raw-AADT model predicts **7.08 vehicle crashes over 6 years vs 4.76
  for log(AADT), a factor of 1.49**, and **1.37 vs 1.22 ped crashes, a factor of 1.12**.
  That is not "astronomical". You have to push to 100,000 AADT, 2.4× beyond anything in the
  data, before raw AADT even reaches 3.4× the log model. The README's stated reason for the
  specification does not survive being fit.
- **The 50,000-AADT scenario is entirely hypothetical.** Observed max is 41,808 and exactly
  **zero** sites exceed 50,000; only **one** site exceeds 30,000. Over the range the model is
  actually applied to, the four curves are visually and numerically interchangeable (at the
  median AADT all four predict 0.80 ped / 3.09 to 3.15 vehicle crashes). The README argues from
  a regime the data never enters.
- **Out-of-sample: inconclusive. No winner.** Stratified repeated-CV spread across all four
  ped specs is 0.7222 to 0.7250 (0.0028, 0.4%) against a seed-to-seed SD of ~0.009 and a
  fold-to-fold SD of ~0.08. Vehicle spread is 2.8946 to 2.8964 (0.0018, 0.06%) against a seed SD
  of ~0.02 to 0.04 and fold SD ~0.16 to 0.20. The ranking also **inverts** between stratified and
  plain KFold. Any of the four is defensible on predictive grounds; the differences are noise.
- In-sample, `log_aadt` has the best AIC for both modes (716.80 ped, 1514.40 vehicle) but the
  margin over the worst spec is 1.73 (ped) and 2.46 (vehicle) AIC points, below the
  conventional ~2-point threshold for ped and barely at it for vehicle. **BIC prefers dropping
  volume entirely** for both modes (759.58 ped, 1559.17 vehicle).
- **Volume is not a significant predictor of pedestrian crashes at all**: `log_aadt`
  β = 0.2230, SE = 0.1430, **p = 0.119**, 95% CI [−0.057, 0.503]. The CI includes zero. It is
  marginally significant for vehicle crashes (β = 0.2264, SE = 0.1066, p = 0.034).
- Keep `log_aadt`: it is defensible on HSM convention, in-sample AIC, and interpretability,
  and nothing beats it. But **the README should stop justifying it with a claim about
  astronomical 50,000-AADT predictions, because that claim is measurably false at 1.49×**,
  and it should stop implying the choice is consequential for accuracy. It is not.

---

# E3, Bike volume/exposure predictor

### Stated reason

The bike model silently uses `log_bike_centrality` where ped/vehicle use `log_aadt`. No
commit message, docstring or note justifies the swap, and in the production fit the
centrality coefficient is not significant (β=0.141, SE=0.157, p=0.369). We need to know
whether centrality earns its place.

### Specs compared, target `bike_total` (169 events; **256 of 346 sites have zero**)

| spec | exposure term(s) |
|---|---|
| a_centrality (production) | `+ log_bike_centrality` |
| b_aadt | `+ log_aadt` |
| c_both | `+ log_bike_centrality + log_aadt` |
| d_neither | *(none)* |

### Results

| spec | conv | opt | LL | AIC | BIC | CV MAE (strat) | CV RMSE (strat) | pooled OOF MAE | pooled OOF RMSE | ρ | repeated-CV OOF MAE (10 seeds) |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| a_centrality | yes | bfgs | −275.66 | 575.31 | 621.47 | **0.5980 ± 0.0449** | 1.0754 ± 0.1752 | **0.5981** | 1.0872 | 0.372 | 0.6010 ± 0.0077 |
| b_aadt | yes | bfgs | −275.57 | 575.13 | 621.29 | 0.6004 ± 0.0397 | 1.0830 ± 0.1596 | 0.6004 | 1.0927 | **0.377** | 0.6024 ± 0.0048 |
| c_both | yes | bfgs | −275.29 | 576.58 | 626.59 | 0.5992 ± 0.0399 | 1.0851 ± 0.1575 | 0.5992 | 1.0946 | 0.372 | 0.6032 ± 0.0046 |
| d_neither | yes | bfgs | −276.07 | **574.13** | **616.45** | 0.5992 ± 0.0446 | **1.0709 ± 0.1811** | 0.5993 | **1.0835** | 0.374 | **0.5994 ± 0.0067** |

Coefficients:

| spec | term | β | SE | p | exp(β) | 95% CI (β) |
|---|---|---:|---:|---:|---:|---|
| a | `log_bike_centrality` | 0.141156 | 0.157275 | 0.3694 | 1.1516 | [−0.1671, 0.4494] |
| b | `log_aadt` | 0.210293 | 0.210042 | 0.3167 | 1.2340 | [−0.2014, 0.6220] |
| c | `log_bike_centrality` | 0.116014 | 0.158034 | 0.4629 | 1.1230 | [−0.1937, 0.4258] |
| c | `log_aadt` | 0.182274 | 0.213396 | 0.3930 | 1.1999 | [−0.2360, 0.6005] |

**Not one exposure coefficient is significant in any spec.** Every 95% CI contains zero.

Likelihood-ratio tests against the nested null (spec d, no exposure term):

| spec | LR statistic | df | p |
|---|---:|---:|---:|
| a_centrality | 0.8221 | 1 | 0.3646 |
| b_aadt | 1.0018 | 1 | 0.3169 |
| c_both | 1.5507 | 2 | 0.4605 |

### Are centrality and AADT measuring the same thing?

| pair | statistic | value | p |
|---|---|---:|---:|
| `log_bike_centrality` vs `log_aadt` | Pearson r | **0.2516** (r² = 0.0633) | 2.13e-06 |
| `log_bike_centrality` vs `log_aadt` | Spearman ρ | 0.3304 | 2.93e-10 |
| `bike_centrality` vs `max_aadt` (raw) | Pearson r | 0.4036 | 5.48e-15 |

**No. They measure genuinely different things.** On the log scale they share only 6.3% of
their variance. The correlation is highly significant but weak. Centrality is not a proxy
for AADT.

### VIF, spec (c) design matrix

| VIF | column |
|---:|---|
| 439.167 | `Intercept` *(not interpretable, intercept VIF is an artefact of the constant column)* |
| 2.305 | `C(arterial_class)[T.5]` |
| 2.286 | `C(arterial_class)[T.3]` |
| 2.014 | `max_speed_limit` |
| 1.940 | `C(arterial_class)[T.2]` |
| **1.539** | **`log_aadt`** |
| **1.419** | **`log_bike_centrality`** |
| 1.251 | `is_signalized` |
| 1.189 | `C(legs_cat, …)[T.3]` |
| 1.133 | `C(legs_cat, …)[T.2]` |
| 1.114 | `C(legs_cat, …)[T.5]` |
| 1.058 | `bike_facility` |

Both exposure terms sit at VIF ≈ 1.4 to 1.5, far below any threshold of concern (5 or 10).
**Spec (c) is cleanly identified, including both is statistically unproblematic. It simply
adds nothing.**

### Out-of-sample ranking (repeated stratified 5-fold, 10 seeds, pooled OOF MAE)

| rank | spec | OOF MAE | seed-to-seed SD | fold SD (seed 0) |
|---|---|---:|---:|---:|
| 1 | d_neither | 0.5994 | ± 0.0067 | 0.0446 |
| 2 | a_centrality (production) | 0.6010 | ± 0.0077 | 0.0449 |
| 3 | b_aadt | 0.6024 | ± 0.0048 | 0.0397 |
| 4 | c_both | 0.6032 | ± 0.0046 | 0.0399 |

Full spread best-to-worst: **0.0038 MAE (0.6%)**. Production (a) sits 0.0016 behind the
nominal winner.

Plain KFold agrees on the direction and on the magnitude of the non-difference:
d 0.5919, a 0.5946, b 0.5973, c 0.5988, spread 0.0069, against a fold SD of ~0.107.

### Verdict, E3

**`log_bike_centrality` does not earn its place. But neither does any alternative. The
honest finding is that the bike model has no usable exposure term at all.**

- Nominal OOS winner is **`d_neither` (drop the exposure term entirely)** at 0.5994 ± 0.0067,
  with production `a_centrality` second at 0.6010 ± 0.0077.
- **The margin is 0.0016 MAE, 0.27%, against a seed-to-seed SD of 0.007 and a fold-to-fold
  SD of 0.045. This is not a meaningful difference. It is well inside fold-to-fold noise and
  comparable to seed noise. I am not declaring a winner on out-of-sample error.**
- What *is* decisive is that **all four specs are statistically indistinguishable and no
  exposure term is significant**: every 95% CI covers zero, and no LRT against the no-exposure
  null comes close (p = 0.365, 0.317, 0.461). The total AIC spread across all four specs is
  2.45 points; BIC unambiguously prefers `d_neither` (616.45 vs 621.29 to 626.59), because BIC
  penalises the extra parameter that buys nothing.
- Centrality and AADT are **not** redundant (log-scale r = 0.252, r² = 0.063; VIF 1.42 / 1.54),
  so this is not a collinearity problem. They are two genuinely different measurements, and
  **neither one predicts bike crashes** in this dataset.
- **The swap to `log_bike_centrality` is undocumented and unjustified by the data, but
  swapping back to `log_aadt` would be equally unjustified** (`b` ranks 3rd of 4). At 169
  events across 346 sites with 74% structural zeros, the bike exposure effect is **not
  identifiable**: the data cannot distinguish these hypotheses. If the term is retained it
  should be retained on stated theoretical grounds with an explicit note that it is not
  empirically supported, not left silent in the code.

---

# E5, Leg-count encoding

### Stated reason

`pipeline/feature_encoding.py:8-16` justifies top-coding with a specific uncited claim: that
a continuous per-leg slope "reads a 6-leg intersection as roughly +280% over a 4-leg one,
with a credible interval spanning +80% to +700%", fit where "2-to-4-leg sites are 97% of the
data", and "no six-leg site actually supporting it". No saved run backs this.

### Specs compared

Each mode keeps its production volume term (bike `log_bike_centrality`, ped and vehicle
`log_aadt`); only the leg term varies.

| spec | leg term |
|---|---|
| a_topcoded_cat (production) | `C(legs_cat, Treatment(reference=4))`, 5+ collapsed |
| b_continuous | `num_legs` |
| c_full_cat | `C(num_legs, Treatment(reference=4))`. No top-coding |

### `num_legs` distribution in the 346-row fit set

| num_legs | sites | % of fit set | bike events | ped events | vehicle events |
|---:|---:|---:|---:|---:|---:|
| 2 | 21 | 6.07% | 7 | 14 | 81 |
| 3 | 116 | 33.53% | 12 | 19 | 125 |
| 4 | 196 | 56.65% | 136 | 219 | 965 |
| 5 | 10 | 2.89% | 10 | 9 | 92 |
| 6 | **3** | 0.87% | 4 | 5 | 32 |

2-to-4-leg sites: **333 / 346 = 96.24%**.

### Results

**bike** (`bike_total`, 169 events; volume term `log_bike_centrality`)

| spec | conv | opt | LL | AIC | BIC | CV MAE (strat) | CV RMSE (strat) | pooled OOF MAE | pooled OOF RMSE | ρ | repeated-CV OOF MAE |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| a_topcoded_cat | yes | bfgs | −275.66 | **575.31** | 621.47 | **0.5980 ± 0.0449** | **1.0754 ± 0.1752** | **0.5981** | **1.0872** | **0.372** | **0.6010 ± 0.0077** |
| b_continuous | yes | nm | −281.35 | 582.71 | **621.17** | 0.6560 ± 0.0579 | 1.1983 ± 0.1778 | 0.6560 | 1.2090 | 0.345 | 0.6515 ± 0.0061 |
| c_full_cat | yes | bfgs | −275.54 | 577.08 | 627.09 | 0.6126 ± 0.0506 | 1.0938 ± 0.1899 | 0.6127 | 1.1073 | 0.367 | 0.6087 ± 0.0076 |

**ped** (`ped_total`, 266 events; volume term `log_aadt`)

| spec | conv | opt | LL | AIC | BIC | CV MAE (strat) | CV RMSE (strat) | pooled OOF MAE | pooled OOF RMSE | ρ | repeated-CV OOF MAE |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| a_topcoded_cat | yes | bfgs | −346.40 | **716.80** | **762.96** | **0.7158 ± 0.0817** | **1.0690 ± 0.1677** | **0.7157** | **1.0792** | **0.562** | **0.7232 ± 0.0095** |
| b_continuous | yes | nm | −364.14 | 748.29 | 786.75 | 0.7790 ± 0.1055 | 1.1915 ± 0.2132 | 0.7789 | 1.2064 | 0.510 | 0.7796 ± 0.0076 |
| c_full_cat | yes | nm | −346.33 | 718.66 | 768.66 | 0.7314 ± 0.0941 | 1.0989 ± 0.1848 | 0.7312 | 1.1109 | 0.533 | 0.7359 ± 0.0071 |

**vehicle** (`vehicle_only_total`, 1295 events; volume term `log_aadt`)

| spec | conv | opt | LL | AIC | BIC | CV MAE (strat) | CV RMSE (strat) | pooled OOF MAE | pooled OOF RMSE | ρ | repeated-CV OOF MAE |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| a_topcoded_cat | yes | newton | −745.20 | **1514.40** | **1560.56** | **2.9069 ± 0.1621** | **4.4994 ± 0.6986** | **2.9067** | **4.5424** | 0.580 | **2.8946 ± 0.0198** |
| b_continuous | yes | newton | −762.49 | 1544.97 | 1583.43 | 3.0815 ± 0.0882 | 4.8451 ± 0.5035 | 3.0812 | 4.8655 | 0.529 | 2.9973 ± 0.0387 |
| c_full_cat | yes | newton | −745.17 | 1516.34 | 1566.35 | 2.9832 ± 0.2224 | 4.6912 ± 0.6873 | 2.9828 | 4.7305 | **0.561** | 2.9433 ± 0.0288 |

### Convergence and fold-sparsity honesty check

Spec (c) was flagged in the brief as possibly unstable at sparse levels. **It was not.**
All fits converged under the robust fitter. A dedicated check over **50 training splits
(10 seeds × 5 folds)** for each mode found **zero splits missing any `num_legs` level**,
including the 3-site 6-leg level. So no coefficient was ever dropped or unidentified in
CV, and no fold results were imputed. Spec (c) is estimable here; it just predicts worse.

### Secondary CV, plain `KFold(n_splits=5, shuffle=True, random_state=0)`, pooled OOF MAE

| spec | bike | ped | vehicle |
|---|---:|---:|---:|
| a_topcoded_cat | **0.5946** | **0.7271** | **2.8861** |
| b_continuous | 0.6448 | 0.8014 | 3.0560 |
| c_full_cat | 0.6046 | 0.7459 | 2.9267 |

Same ordering as stratified CV in all three modes. Unlike E1, this ranking is stable.

### Verification of the docstring's numbers

From spec (b), the implied 6-leg vs 4-leg multiplicative effect is `exp(2 · β_num_legs)`.
Intervals are **Wald confidence intervals** from the MLE standard error. Note the docstring
says "credible interval", which implies a Bayesian posterior; this codebase fits by MLE
(no pymc/arviz installed), so a credible interval cannot be reproduced. The comparison
below is confidence-vs-credible and that caveat applies to all three rows.

| mode | β (`num_legs`) | SE | p | exp(β) per leg | 6-leg vs 4-leg | 90% CI | 95% CI |
|---|---:|---:|---:|---:|---|---|---|
| **bike** | 0.671697 | 0.188221 | 3.588e-04 | 1.9576 | **3.8320× = +283.2%** | +106.3% to +611.8% (2.063×-7.118×) | **+83.2% to +701.4%** (1.832×-8.014×) |
| ped | 0.584000 | 0.124392 | 2.668e-06 | 1.7932 | 3.2156× = +221.6% | +113.6% to +384.1% | +97.5% to +423.6% |
| vehicle | 0.547608 | 0.085490 | 1.499e-10 | 1.7291 | 2.9898× = +199.0% | +125.7% to +296.1% | +113.8% to +318.0% |

Docstring claim: **+280%, interval +80% to +700%**.

| claim | measured | verdict |
|---|---|---|
| "+280% for 6-leg vs 4-leg" | **bike: +283.2%** / ped +221.6% / vehicle +199.0% | **CONFIRMED for the bike model** (+283.2% vs +280%, a 1.1% relative discrepancy). APPROXIMATELY RIGHT for ped. For vehicle the point estimate is +199.0%, wrong as stated but not excluded by the 95% CI [+113.8%, +318.0%]. |
| "interval spanning +80% to +700%" | **bike 95% CI: +83.2% to +701.4%** | **CONFIRMED for the bike model**. This is an essentially exact match on both endpoints. |
| "2-to-4-leg sites are 97% of the data" | **96.24%** (333 / 346) | **APPROXIMATELY RIGHT**, off by 0.76 points, consistent with rounding. |
| "no six-leg site actually supporting it" | **3 six-leg sites**, carrying 4 bike, 5 ped and 32 vehicle crashes | **WRONG**, six-leg sites exist in the fit set and contribute events. |

**The docstring's headline numbers reproduce almost exactly against the bike model**
(+283.2% vs +280%; CI +83.2%/+701.4% vs +80%/+700%). Whoever wrote it did run this fit,
on the bike target, and reported it accurately to two significant figures. The only
factually wrong sub-claim is "no six-leg site". There are three, and they carry crashes.

Critically, **the actual 6-leg data contradicts the continuous slope's extrapolation**. From
spec (c), 6-leg vs 4-leg with no functional-form constraint:

| mode | β (6-leg vs 4-leg) | SE | p | exp(β) |
|---|---:|---:|---:|---:|
| bike | −0.2349 | 0.8378 | 0.7792 | **0.7906** |
| ped | −0.1814 | 0.5868 | 0.7572 | **0.8341** |
| vehicle | +0.3751 | 0.5345 | 0.4828 | **1.4551** |

The continuous slope claims +283% for bike; the unconstrained categorical estimate is
**−21%** and nowhere near significant (p = 0.78). Same story for ped (−17%, p = 0.76). The
6-leg effect is **not identifiable at 3 sites**, every CI is enormous, but the point
estimates run in the *opposite* direction to the extrapolation, which is exactly the failure
mode the docstring warned about.

The real leg signal is not a monotone per-leg slope at all. It is that **3-leg sites are
dramatically safer than 4-leg sites**: bike β = −1.562 (p = 6.3e-06, exp(β) = 0.210), ped
β = −1.707 (p = 2.1e-11, exp(β) = 0.182), vehicle β = −1.263 (p = 4.1e-18, exp(β) = 0.283).
Forcing a straight line through that, then extrapolating it past 4 legs, is what produces
the +283%.

### Verdict, E5

**Top-coding wins decisively on out-of-sample error in all three modes, and the docstring's
numbers are real. The top-coding decision is vindicated, one factual sub-claim is wrong.**

- **`a_topcoded_cat` is the out-of-sample winner in every mode, on both CV schemes, on MAE
  and on RMSE.** Repeated-CV OOF MAE margins over the continuous slope:
  - bike: 0.6010 ± 0.0077 vs 0.6515 ± 0.0061, **margin 0.0505, ≈ 6.6 seed-SDs**
  - ped: 0.7232 ± 0.0095 vs 0.7796 ± 0.0076, **margin 0.0564, ≈ 5.9 seed-SDs**
  - vehicle: 2.8946 ± 0.0198 vs 2.9973 ± 0.0387, **margin 0.1027, ≈ 5.2 seed-SDs**

  This is **decisive, not noise**. The only clear-cut result of the three experiments.
  Margins over the full categorical are smaller but consistently in the same direction
  (bike +0.0077 ≈ 1.0 SD, ped +0.0127 ≈ 1.3 SD, vehicle +0.0487 ≈ 2.5 SD), and are corroborated
  by AIC (575.31 vs 577.08; 716.80 vs 718.66; 1514.40 vs 1516.34) and by the same ordering
  under plain KFold. Directionally consistent, individually modest.
- AIC agrees in all three modes. The continuous slope costs **7.4 AIC (bike), 31.5 (ped) and
  30.6 (vehicle)** relative to top-coding. A large in-sample penalty for a spec that also
  predicts worse out of sample. (BIC narrowly prefers `b_continuous` for bike, 621.17 vs
  621.47, a 0.30-point difference that is meaningless; BIC agrees with AIC for ped and vehicle.)
- **Docstring verdict: CONFIRMED on the headline numbers** (bike +283.2% vs claimed +280%;
  95% CI +83.2%-+701.4% vs claimed +80%-+700%), **APPROXIMATELY RIGHT on the 97% share**
  (measured 96.24%), and **WRONG on "no six-leg site actually supporting it"**. There are
  3 six-leg sites carrying 4 bike, 5 ped and 32 vehicle crashes. The docstring should be
  updated to say the six-leg sites are too few to identify the effect (n = 3, p = 0.78),
  which is both true and a *stronger* argument for top-coding than the false claim.
- The `credible interval` wording should be corrected to `confidence interval` unless a
  Bayesian fit is actually being referenced. This codebase fits by MLE.

---

## Cross-cutting summary

| experiment | OOS winner | margin over production | decisive? |
|---|---|---|---|
| E1 ped | none (d nominally, 0.7222 vs a 0.7232) | 0.0010 MAE (0.14%) | **No, within noise; ranking inverts under plain KFold** |
| E1 vehicle | none (a nominally, 2.8946 vs d 2.8960) | 0.0014 MAE (0.05%) | **No, within noise; ranking inverts under plain KFold** |
| E3 bike | none (d nominally, 0.5994 vs a 0.6010) | 0.0016 MAE (0.27%) | **No, within noise; no exposure term is significant in any spec** |
| E5 bike | a_topcoded_cat (production) | 0.0505 MAE over continuous (8.4%) | **Yes, ≈ 6.6 seed-SDs** |
| E5 ped | a_topcoded_cat (production) | 0.0564 MAE over continuous (7.8%) | **Yes, ≈ 5.9 seed-SDs** |
| E5 vehicle | a_topcoded_cat (production) | 0.1027 MAE over continuous (3.5%) | **Yes, ≈ 5.2 seed-SDs** |

Two of the project's three documented specification arguments are not supported by
measurement in the way they are written. The README's log-AADT argument rests on a claim
about 50,000-AADT predictions that measurement contradicts, raw AADT gives 7.08 vehicle
crashes there against log's 4.76, a ratio of 1.49×, which is not "astronomical". And that
concerns a regime containing zero observations (observed max AADT 41,808). The bike model's
undocumented centrality swap is not empirically supported, though neither is any
alternative, so this is a documentation failure rather than a modelling error. The
top-coding argument in `feature_encoding.py` is the one that holds up: its numbers reproduce
almost exactly and its conclusion is confirmed by a decisive out-of-sample margin, despite
one false sub-claim about six-leg sites.

Separately, the production optimizer ladder (newton → BFGS) is fragile on badly-scaled
predictors and can return a diverged optimum that is 114 AIC points wrong while
`mle_retvals["converged"]` is False. `fit_for_mode` would `sys.exit` on that rather than
silently accept it, so production artefacts are not affected. But any future spec search
run through the naive path will produce garbage comparisons unless the design matrix is
scaled or Nelder-Mead is in the ladder.
