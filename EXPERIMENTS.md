# Measured A/B evidence for the modelling choices

**Run date:** 2026-08-04 · **Repo state:** `main` @ `780efca` · **Environment:** Python 3.13.2, pandas 2.2.3, numpy 2.1.3, statsmodels 0.14.6, scipy 1.15.1, patsy 1.0.2, scikit-learn 1.6.1

## Why this document exists

A prior audit ([`MODEL_NOTES.md`](MODEL_NOTES.md)) found that almost every modelling decision in this project was a *stated justification* rather than a *measured comparison*. Nothing in the repo recorded a rejected alternative with numbers. We designed seven experiments to close that gap. Three independent agents ran them, and a fourth reviewed the results without having run any of them.

Everything below is measured. Where an experiment could not resolve a question, it says so. **"Not distinguishable" is a result, and this document reports it as one.**

- Experiment scripts: [`experiments/ab/`](experiments/ab/)
- Raw outputs: [`experiments/results/`](experiments/results/) (markdown + JSON per experiment)
- Independent review: [`experiments/results/eval_review.md`](experiments/results/eval_review.md)

---

## ⚠ Read this before quoting any number below

**The repeated-cross-validation standard deviations understate uncertainty by roughly 2 to 5×.**

Several experiments report stability as "mean ± SD across 10 CV seeds". Reshuffling folds re-partitions the *same fixed 346 intersections*, so that SD measures fold-assignment noise rather than sampling error. It answers "would I get this again on this dataset?", not "is this real?".

Against the correct **paired site-level standard error**, the independent reviewer found:

| claim | as reported | correctly powered |
|---|---|---|
| E4 per-mode vs pooled (bike) | "loses in 9/9 seeds" | paired t = **−0.59**, not distinguishable from zero |
| E4 (ped / vehicle) | n/a | t = −0.13 / +0.72, not distinguishable |
| E5 leg encoding | "6.6 seed-SDs" | ≈ **2.3 SEs**: still real, but much less dramatic |
| E1, E3 all comparisons | various | **|t| ≤ 1.10 throughout** |

**The honest summary of this entire body of work:** this dataset can decisively rule out *badly wrong* specifications such as a linear per-leg slope, Poisson variance or zero-inflation, but it **cannot adjudicate between reasonable ones**. Read every "no winner" below as "underpowered to detect a difference of this size", not "proven equivalent".

---

## Results at a glance

| # | Question | Verdict | Strength |
|---|---|---|---|
| **E5** | Leg count: continuous vs top-coded vs full categorical | **Top-coding confirmed.** Continuous slope decisively rejected | ★★★ strongest result |
| **E2** | Negative Binomial vs Poisson | **NB confirmed, on variance only.** Point accuracy identical | ★★★ decisive |
| **E7** | NB vs zero-inflated NB | **Zero-inflation not warranted** in any mode | ★★★ decisive |
| **E6** | Exposure as offset vs free covariate | **Not identifiable on real data** (demonstrated); offset justified on synthetic | ★★★ decisive |
| **E4** | One pooled model vs three per-mode models | **Per-mode delivers no measurable out-of-sample benefit** | ★★ real but weaker than first claimed |
| **E1** | log(AADT) vs raw vs sqrt vs none | **Cannot distinguish.** README's stated rationale is refuted | ★ underpowered |
| **E3** | Bike exposure: centrality vs AADT vs both vs neither | **Cannot distinguish.** Effect not identifiable at 169 events | ★ underpowered |

---

## Validity anchors

Before trusting any comparison, we checked both reconstructions against known ground truth.

**The current v3 models reproduce exactly.** α = 1.2382 / 0.3049 / 0.6585 and MAE 0.577 / 0.698 / 2.764, identical to the shipped pipeline.

**The historical v2 model reproduces exactly.** Fitting `total_crashes ~ is_signalized + num_legs + max_speed_limit + bike_facility + C(arterial_class) + log_aadt`:

| quantity | reconstruction | old README |
|---|---|---|
| sum predicted | 1772.7 | 1,772.7 |
| calibration gap | +3.06% | +3.1% |
| MAE | 3.474 | 3.47 |
| α | 0.640 | 0.640 |
| `num_legs` β | +0.591 | +0.591 |
| `log_aadt` β | +0.257 | +0.257 |

Every digit matches. This confirms the old README's numbers were real measurements of a model that has since been replaced, and that both experimental arms are faithful.

---

## E5. Leg-count encoding *(strongest result)*

**Why:** [`pipeline/feature_encoding.py`](pipeline/feature_encoding.py) justified top-coding with a specific uncited claim: that a continuous per-leg slope "reads a 6-leg intersection as roughly +280% over a 4-leg one, with a credible interval spanning +80% to +700%". No saved run backed it.

**Specs:** (a) `C(legs_cat, Treatment(reference=4))` top-coded 5+ *(production)* · (b) `num_legs` continuous · (c) `C(num_legs, ...)` full categorical. All three modes.

### The properly-powered evidence: nested likelihood-ratio tests

Specs (a) and (b) are both restrictions of (c), so this is a nested test. No cross-validation needed, and it is far better powered than the CV comparison the experiment originally led with.

| mode | top-coded vs full | continuous vs full |
|---|---|---|
| bike | LR=0.230, df=1, **p=0.632** | LR=11.626, df=3, **p=0.0088** |
| ped | LR=0.146, df=1, **p=0.703** | LR=35.630, df=3, **p=9.0e-08** |
| vehicle | LR=0.060, df=1, **p=0.806** | LR=34.626, df=3, **p=1.5e-07** |

**Top-coding is never rejected. The continuous slope is decisively rejected in all three modes.** Collapsing 5+ into one category costs nothing; forcing a straight line through leg count costs a great deal.

Out-of-sample MAE agrees: 0.6010 vs 0.6515 (bike), 0.7232 vs 0.7796 (ped), 2.8946 vs 2.9973 (vehicle), top-coded over continuous in every mode and every CV scheme.

### The docstring claim: numerically confirmed, but its reasoning is wrong

Refitting the continuous spec on bike: β = 0.671697, SE = 0.188221, so 6-leg vs 4-leg is `exp(2β)` = **+283.2%**, 95% CI **+83.2% to +701.4%**.

Against the claimed "+280%, +80% to +700%", near-exact on all three figures. Whoever wrote that docstring did run this fit. *(Caveat: the docstring says "credible interval"; these are MLE Wald confidence intervals.)*

But two of its supporting statements are wrong:

- **"no six-leg site actually supporting it" is false.** There are **3** six-leg sites, carrying 4 bike, 5 ped and 32 vehicle crashes.
- **"2-to-4-leg sites are 97% of the data"** is actually **96.24%** (333/346). Close enough.

**And the real six-leg data contradicts the extrapolation entirely.** Fit unconstrained, 6-leg vs 4-leg gives exp(β) = **0.79** for bike (p=0.78) and **0.83** for ped (p=0.76), *fewer* crashes, the opposite sign to the extrapolated +283%.

The genuine signal is not a per-leg slope at all. It is that **3-leg intersections are dramatically safer than 4-leg ones**: exp(β) = 0.21 bike (p=6.3e-06), 0.18 ped (p=2.1e-11), 0.28 vehicle (p=4.1e-18). Forcing a straight line through that steep 3→4 drop and extrapolating it outward is what manufactures the +283% figure.

> **Verdict:** top-coding is correct, and for a better reason than the docstring gives. The risk was never "we can't trust the 6-leg estimate". It was "a linear leg term is the wrong functional form."

---

## E2. Negative Binomial vs Poisson

**Why:** the repo asserted NB purely because estimated α > 0.05 ([`fit_risk_model.py:363`](pipeline/fit_risk_model.py:363)). No Poisson model was ever fit.

**Specs:** identical formula and offset under NB2 vs Poisson, all three modes.

| | bike | ped | vehicle |
|---|---|---|---|
| variance / mean | 2.78 | 2.07 | 7.68 |
| ΔAIC (NB better by) | 49.1 | 10.0 | 396.6 |
| LR statistic | 51.075 | 11.977 | 398.566 |
| boundary-corrected p | 4.45e-13 | 2.69e-04 | 5.65e-89 |

Because α = 0 sits on the boundary of the parameter space, the null is a 50:50 mixture of χ²(0) and χ²(1), so the correct p-value is half the naive χ²(1) value. Reported both ways.

**Where it matters, interval calibration.** The "Poisson intervals are too narrow" claim holds decisively, but **only for vehicle**: 90% coverage 76.6% (Poisson) vs 94.8% (NB) in-sample, 74.9% vs 93.6% out-of-fold, with intervals 47% narrower and **81 of 346 sites** falling outside a nominally-90% interval instead of the expected ~35. For bike and ped both families *over*-cover (94.8 to 97.1%) because of count discreteness.

**Where it does not matter, point accuracy.** 5-fold CV MAE is statistically indistinguishable between families: paired per-fold deltas +0.0035±0.0099 (p=0.478), +0.0026±0.0043 (p=0.256), −0.0087±0.0837 (p=0.827). The sign isn't even consistent.

> **Verdict:** NB is the right family, and the reason is *variance*, not accuracy. α governs the second moment, not the mean. Anything consuming only the point prediction is unaffected by this choice; anything consuming the interval, which includes this project's headline credible intervals, depends on it entirely.

---

## E7. Zero-inflation

**Why:** `evaluate_models.py` contains a zero-inflation check that had never run (the script was broken). With 256 of 346 sites recording zero bike crashes, excess zeros were plausible and untested.

**Specs:** NB2 vs `ZeroInflatedNegativeBinomialP` (logit inflation), same formula and offset.

| mode | observed zeros | NB-predicted zeros | fitted inflation π̂ | ΔAIC | ΔBIC |
|---|---|---|---|---|---|
| bike | 256 | 256.45 | 1.34e-13 | +2.000 | +5.846 |
| ped | 204 | 206.02 | 4.96e-14 | +2.000 | +5.846 |
| vehicle | 89 | 90.86 | 1.76e-14 | +2.000 | +5.846 |

The inflation parameter collapses to the NB boundary in every mode; log-likelihood gain ≤ 3e-09; CV MAE differs by ≤ 0.0001. ZINB is strictly worse by exactly the cost of its extra parameter. Every observed-minus-expected zero gap is *negative* and inside a 4000-draw parametric bootstrap band.

The agent implemented a Vuong test, but the result comes out **degenerate** (raw V ≈ 3e-05, meaning "the same model", not "two models that tie"). It deliberately suppressed the corrected variants, which amount to a constant divided by numerical noise, rather than reporting them as if meaningful.

The zero counts here were independently reproduced by `pipeline/evaluate_models.py`, a separate implementation: 256/256.5, 204/206.0, 89/90.9.

> **Verdict:** NB already accounts for the zeros through overdispersion. Despite the sparsity that motivated the question, bike shows no excess-zero problem at all.

---

## E6. Exposure: offset vs free covariate

**Why:** the audit asserted this comparison was untestable because `years_observed` ≡ 6. Assertion replaced with demonstration.

### Part A. Non-identifiability, demonstrated

The design matrix has **rank 11 against 12 columns**: the exposure column is an exact constant multiple of the intercept. A ridge probe restarting along the collinear direction moves the exposure coefficient by **4.000** and the intercept by **24.000** while the log-likelihood moves by **≤1.1e-13**, with `intercept + 6·coefficient` invariant to ≤1.8e-15. The two parameters are not separately estimable; only their combination is.

**The operationally alarming part:** statsmodels raises nothing, drops nothing, reports `converged=True`, and in **4 of 6 rank-deficient specs returns finite, entirely plausible-looking standard errors.** Nothing in a normal workflow would catch this.

### Part B. Tested on a variable-exposure design

Since real exposure is constant, a synthetic variable-exposure dataset was built from the per-year crash grid (each site assigned 2 to 6 years, seeded). **This is a synthetic design and does not change what is knowable from the real 6-year window.** A supplementary arm used real crash-years with only the window synthetic.

Free coefficient on log(exposure), 95% CI. The offset constrains this to exactly 1.0:

| arm | coefficient | 95% CI | contains 1.0? |
|---|---|---|---|
| bike | +1.623 | [0.868, 2.378] | yes (but see below) |
| ped | +1.177 | [0.684, 1.670] | yes |
| vehicle | +1.080 | [0.750, 1.409] | yes |
| real-year total | +1.066 | [0.787, 1.345] | yes |

Across seeds 0 to 9 the CI covers 1.0 in 8/10, 9/10, 10/10 and 8/10 respectively; AIC prefers the constrained offset in 3 of 4 arms. **The bike arm alone is inconclusive**, CI width 1.51, and its between-seed SD (0.496) exceeds its own mean standard error (0.352).

> **Verdict:** on the real data the question is not answerable, and the repo should say so rather than imply a choice was made. On synthetic variable exposure, the offset constraint is justified.

---

## E4. One pooled model vs three per-mode models

**Why:** this is the v2→v3 rewrite, and it is the single largest architectural decision in the project. Its commit message describes *what* v3 is but never says why v3 replaced the pooled model. Nothing in the repo compares them.

**Arms:** (1) one NB on `total_crashes`, mode counts obtained by scaling by a citywide share · (1b) the true historical v2, with `num_legs` continuous · (2) the three per-mode models as shipped. 230 NB fits; shares computed on **training rows only within each fold** (leakage explicitly avoided and verified by the reviewer).

### The headline table: out-of-sample MAE, all 11 cells independently reproduced

| | Arm 1b (true v2) | Arm 1 (pooled + fixed legs) | Arm 2 (v3 per-mode) |
|---|---|---|---|
| bike MAE | 0.6380 | 0.6013 | **0.5946** |
| ped MAE | 0.8010 | 0.7290 | **0.7271** |
| vehicle MAE | 3.0824 | **2.8713** | 2.8861 |
| bike-KSI ρ | +0.1797 | **+0.2043** | +0.1681 |

**The v2→v3 commit changed two things at once:** it split one model into three, *and* it fixed the leg encoding. Separating them shows v3 clearly beats what actually shipped as v2, **the rewrite genuinely improved the model**, but a *pooled* model with only the leg encoding fixed matches or beats the current three-model architecture using **12 parameters instead of 36**.

### The properly-powered evidence: coefficient homogeneity

Arm 1's structure is a nested restriction of Arm 2's, so stacking the modes long allows a direct likelihood-ratio test. The analysis none of the original experiments ran:

**LR = 17.15 on 20 df, p = 0.643. ΔAIC +22.85 and ΔBIC +121.75 against per-mode. Not one of the 11 coefficients differs significantly across modes (minimum p = 0.106).**

The three modes do not have detectably different predictor effects. That is the cleanest statement of the finding.

### A confound the experiment did not control, checked in review

Since the bike model's exposure proxy is itself worthless (E3), per-mode might have been losing because of that rather than because of pooling. Refitting the per-mode bike arm with `log_aadt` (0.60603) and with no exposure term (0.60380). Both still lose to pooled × share (0.60016) across 20 seeds. **The result is about pooling, not the bad proxy.**

### What must NOT be claimed

- **"Per-mode is worse."** Unsupported. Paired site-level t = −0.59 / −0.13 / +0.72. The defensible claim is *"delivered no measurable out-of-sample benefit."*
- **Do not publish the bike-KSI rank-correlation result.** 16 events, the two arms' ρ confidence intervals overlap ~85%, the bootstrap holds predictions fixed, and its p=0.021 is the only marginal p-value among ~62 spec-fits. It is also the strongest-claiming headline in the original report.
- **This tests two extreme points** (12 vs 36 parameters) and recommends one. What it actually found is a **regularisation** result. Intermediate architectures, shared coefficients with mode-specific intercepts, or the repo's own never-run [hierarchical sketch](experiments/hierarchical_nb_sketch.py), were not fit and may well beat both.

### Cross-mode residual correlation

An independent run of `pipeline/evaluate_models.py` gives bike-ped 0.202, bike-vehicle 0.089 (not significant, p=0.097), ped-vehicle 0.192.

These are near-orthogonal to the pooling question: low residual correlation obtains under *both* hypotheses, because Arm 1 constrains the systematic part, not the residual. The small positive residue points at a **site-level random effect that neither arm models**, meaning partial pooling. The coefficient-homogeneity test above is what settles the question.

> **Verdict:** the per-mode split delivered no measurable accuracy benefit, and the measured gain from the v2→v3 rewrite belongs to the leg-encoding change. The defensible case for keeping three models is **interpretability**, per-mode risk reporting and cross-mode coefficient comparison, which the app genuinely uses. That is a legitimate product argument. It is not an accuracy argument, and it was never stated.

---

## E1. Volume functional form

**Why:** the README argued for log(AADT) on pure functional-form theory: that raw AADT under a log link would give "astronomical predictions" at a 50,000-AADT site. Nobody had ever fit an alternative.

**Specs:** (a) `log_aadt` *(production)* · (b) raw `max_aadt` · (c) `sqrt(max_aadt)` · (d) no volume term. Ped and vehicle.

### Result: no winner

Repeated CV (10 seeds), ped: no-volume 0.7222±0.0087, log 0.7232±0.0095, sqrt 0.7234, raw 0.7250, total spread 0.0028 (0.4%) against a fold SD of ~0.08. Vehicle: log 2.8946±0.0198, raw 2.8951, no-volume 2.8960, sqrt 2.8964, spread 0.0018 (**0.06%**).

Under a single `KFold(shuffle, seed=0)` the vehicle ordering **inverts**, raw becomes best (2.8828) and no-volume worst (2.9248). A comparison whose ranking flips with the fold seed has not resolved anything.

### The README's stated rationale is refuted

At 50,000 AADT, raw predicts **7.08 vs log's 4.76** vehicle crashes. A factor of **1.49**, not "astronomical". For ped, 1.37 vs 1.22 (1.12×). Reaching a 3.4× divergence requires 100,000 AADT, roughly 2.4× beyond any observation.

**More importantly, observed `max_aadt` runs 1,013 to 41,808. There are zero sites above 50,000, exactly one above 30,000, and two above 25,000.** The README argued from a traffic-volume regime in which this dataset contains no observations at all.

*(Correction: the E1 report's prose states "24 sites above 20,000, 5 above 25,000". The correct figures, from its own JSON output and confirmed independently, are **27 and 2**.)*

### A finding worth its own line

`log_aadt` is **not statistically significant for pedestrian crashes**: β=0.2230, SE=0.1430, **p=0.119**, CI covering zero. Vehicle reaches p=0.034. BIC prefers dropping volume entirely in both modes.

This must not be read as "traffic volume doesn't affect pedestrian crashes." The ped CI spans [0.96×, 1.42×] per AADT doubling, and the design can only detect effects ≥1.32×. **Non-significance here means underpowered, not absent.**

> **Verdict:** the theoretical argument for log(AADT) remains sound and matches HSM convention, keep it. But it is a *convention-and-theory* choice, not one this dataset supports empirically, but the specific "astronomical predictions" justification is false, so remove it.

---

## E3. Bike exposure: centrality vs AADT

**Why:** the bike model silently uses `log_bike_centrality` where ped and vehicle use `log_aadt`. No commit, docstring or comment justifies the swap, and the coefficient is not significant (p=0.369).

**Specs:** (a) `log_bike_centrality` *(production)* · (b) `log_aadt` · (c) both · (d) neither. Target `bike_total`.

### Result: no exposure term earns its place, and none can be ruled in either

Nominal winner is **(d) neither**, 0.5994±0.0067 vs production's 0.6010±0.0077. A margin of 0.0016 MAE (**0.27%**) against a fold SD of 0.045. Firmly within noise; no winner declared.

The decisive part is the significance testing: **no exposure coefficient is significant in any spec**, all 95% CIs cover zero, and likelihood-ratio tests against the no-exposure null give p = 0.365 (centrality), 0.317 (AADT), 0.461 (both). Total AIC spread across all four specs is 2.45; BIC clearly prefers dropping exposure (616.45).

**The two variables are not redundant**, log-scale Pearson r = 0.2516 (r² = 0.063), Spearman 0.3304, VIF 1.42 and 1.54. They are genuinely different measurements, and neither predicts bike crashes at this sample size. Confirmed by `evaluate_models.py`'s VIF table, where the maximum VIF anywhere is 2.27.

With 169 events and 256 of 346 sites at zero, the effect is simply **not identifiable**. As with E1, non-significance is a statement about power, not about cycling.

> **Verdict:** a documentation failure rather than a modelling error. Swapping back to `log_aadt` would be equally unjustified. It ranks third of four. The honest position is that the bike model currently has **no working exposure term**, and that real cyclist counts (Strava Metro, permanent counters) are the fix, not a different proxy.

---

## What changed in the repo as a result

| change | driver |
|---|---|
| `README.md` rewritten to describe the three-model v3 architecture | audit ([`MODEL_NOTES.md`](MODEL_NOTES.md)) |
| README's "astronomical predictions" argument removed | E1 |
| Centrality documented as an unvalidated proxy | E3 |
| Leg top-coding justified by the nested LR test | E5 |
| `feature_encoding.py` docstring corrected (three 6-leg sites exist) | E5 |
| `evaluate_models.py` import bug fixed. The script had never run | E7 investigation |

## Known limitations of this work

1. **Everything is one dataset.** 346 intersections, one neighbourhood, six years. No experiment here generalises beyond Capitol Hill.
2. **Repeated-CV SDs understate uncertainty 2 to 5×** (see the warning at the top). The decisive results (E5, E2-variance, E7, E6) do not depend on them; the "no winner" results (E1, E3, E4) are underpowered.
3. **Multiple comparisons.** The seven experiments compared roughly 62 spec-fits between them. The decisive findings survive that easily (p-values from 1e-07 to 1e-89), but treat any marginal p-value in this document as unreliable, which is precisely why this write-up excludes E4's bike-KSI result.
4. **E4 and E5 are not independent evidence** on leg encoding. E4's Arm 1b is E5's finding aggregated over modes on the same 346 sites. A consistency check, not confirmation.
5. **We tested only two architectural extremes** (fully pooled, fully separate). Nobody fit partial pooling, which is what the residual correlations actually point toward.
6. **The experiments hit three separate instances of statsmodels reporting `converged=True` at a bad or unidentified optimum** (E1: a 114-point AIC error surviving both Newton and BFGS; E6: a local optimum that would have produced a spurious rejection of the offset; E6 Part A: finite standard errors on a rank-deficient design). [`fit_risk_model.py`](pipeline/fit_risk_model.py)'s `_is_converged` trusts exactly this self-report. I verified that the three production specs sit at true optima across four optimizers, but this remains a live trap for anyone editing the model spec.
