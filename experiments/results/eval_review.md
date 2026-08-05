# eval_review.md, independent statistical review of E1-E7

Reviewer: independent agent. Nothing under `pipeline/` or `data/` was read-modified;
`fit_risk_model.main()` / `score_risk` were never invoked. Only `load_and_join`,
`prepare`, `MODES`, `SHARED_PREDICTORS` were imported. All fits below are my own,
in memory, Python 3.13.2 / pandas 2.2.3 / statsmodels 0.14.6 / sklearn 1.6.1,
on the 346-row arterial frame.

My fitting harness differs from all three teams': for every fit I run
newton → bfgs(200) → Nelder-Mead(20000) and **keep the converged rung with the
highest log-likelihood**, so no seed is ever dropped for non-convergence and no
result depends on which optimiser happened to be tried first.

---

## Verdicts at a glance

| exp | verdict | one-line reason |
|---|---|---|
| **E1** volume form | **SOUND WITH CAVEATS** | Numbers reproduce; two site-count figures in the prose contradict its own JSON; "not significant" is written as "no effect"; the null is underpowered by ~5 to 10×. |
| **E2** NB vs Poisson | **SOUND** | Every figure reproduced to the digit. Best-argued experiment of the seven. |
| **E3** bike exposure | **SOUND WITH CAVEATS** | All numbers reproduce; conclusion is right; the power shortfall is acknowledged qualitatively but never quantified, and the verdict prose overshoots it. |
| **E4** pooled vs per-mode | **SOUND METHOD / OVERCLAIMED VERDICT** | No leakage, folds shared, I reproduce the headline table exactly and it survives 20 seeds. But the effects are indistinguishable from zero against site-sampling noise, and the bike-KSI ρ result should not be published. |
| **E5** leg encoding | **SOUND** | Conclusion correct and reproduced exactly; the *stated* evidence (seed-SDs) is inflated ~3×, but a nested LR test they never ran makes it stronger than they claimed. |
| **E6** exposure offset | **SOUND** | Rank deficiency, condition numbers and the likelihood-ridge probe reproduce; Part B is correctly fenced as synthetic. |
| **E7** zero inflation | **SOUND** | Reproduced to 2 dp including bootstrap SDs. Correctly refuses the LRT and suppresses the degenerate Vuong variants. |

---

## 1. What I independently reproduced

### Data facts (all confirmed)
346 rows · `total_crashes` 1720 · bike 169 / ped 266 / vehicle-only 1295 ·
bike-KSI 16, ped-KSI 32, vehicle-only-KSI 25, `ksi_total` 71 ·
zeros 256 / 204 / 89 · var/mean 2.780 / 2.072 / 7.683 ·
`years_observed` unique `[6]`, `offset` unique `[1.79175947]` ·
AADT min 1,013, max 41,808, mean 10,021.97, median 8,051.5 ·
`num_legs` = 21 / 116 / 196 / 10 / 3 for 2/3/4/5/6 legs (333/346 = 96.24 % are 2 to 4 legs) ·
mode targets sum to 1730 = 1.005814 × 1720.

### E1
| quantity | reported | mine |
|---|---|---|
| ped `log_aadt` β / SE / p | 0.222994 / 0.143039 / 0.1190 | 0.222979 / 0.1430 / **0.1190** |
| vehicle `log_aadt` β / p | 0.226367 / 0.0337 | 0.226358 / **0.0337** |
| raw-AADT β (ped) | 1.28589e-05 | 1.28588e-05 (fit as AADT/10⁴, β=0.128588) |
| AIC a/b/c/d (ped) | 716.80 / 718.53 / 717.75 / 717.27 | 716.803 / 718.533 / 717.750 / 717.271 |
| AIC a/b/c/d (vehicle) | 1514.40 / 1515.62 / 1515.03 / 1516.86 | 1514.404 / 1515.619 / 1515.033 / 1516.856 |
| vehicle @ 50,000 AADT raw ÷ log | 7.0822 ÷ 4.7592 = **1.49×** | 7.0822 ÷ 4.7591 = **1.488×** |
| ped @ 50,000 raw ÷ log | 1.3668 ÷ 1.2176 = 1.12× | 1.3668 ÷ 1.2176 = 1.123× |
| sites above 50,000 | 0 | **0** |

I also independently confirm their optimiser finding: fitting raw AADT rescaled to
`max_aadt/10⁴` reaches llf −347.2664 on plain newton, i.e. the same optimum their
Nelder-Mead found and 57 log-likelihood points above the diverged −404.25 that the
production ladder returns on the unscaled column. The conditioning diagnosis is right.

### E2, reproduced to the digit
LR = 51.075 / 11.977 / 398.566; boundary-corrected p = 4.446e-13 / 2.693e-04 /
**5.651e-89**; AIC NB 575.31 / 716.80 / 1514.40 vs Poisson 624.39 / 726.78 / 1910.97;
90 % coverage vehicle **NB 94.8 % (width 10.62) vs Poisson 76.6 % (width 5.65)**,
18 vs **81** sites outside; paired per-fold MAE t-test p = **0.478 / 0.256 / 0.827**.
All identical.

### E3, reproduced to the digit
Pearson r(log_bike_centrality, log_aadt) = **0.2516**, p = 2.13e-06; Spearman 0.3304;
raw-scale Pearson 0.4036. LRT vs no-exposure null: **0.8221 / 1.0018 / 1.5507** →
p = **0.3646 / 0.3169 / 0.4605**. AIC 575.313 / 575.133 / 576.584 / 574.135.
All four exposure coefficients reproduce to 5 dp.

### E4, headline table reproduced **exactly**, then extended
My own CV (independent code, same `KFold(5, shuffle=True, random_state=0)`),
all 11 cells of their "most important table":

| cell | theirs | mine |
|---|---|---|
| bike MAE A1b / A1 / A2 | 0.6380 / 0.6013 / 0.5946 | 0.6380 / 0.6013 / 0.5946 |
| bike RMSE | 1.1379 / 1.0749 / 1.0866 | 1.1379 / 1.0749 / 1.0866 |
| bike ρ | +0.3820 / +0.4014 / +0.3917 | +0.3820 / +0.4014 / +0.3917 |
| ped MAE | 0.8010 / 0.7290 / 0.7271 | 0.8010 / 0.7290 / 0.7271 |
| ped RMSE | 1.2680 / 1.0893 / 1.1044 | 1.2680 / 1.0893 / 1.1044 |
| ped ρ | +0.5224 / +0.5536 / +0.5415 | +0.5224 / +0.5536 / +0.5415 |
| vehicle MAE | 3.0824 / 2.8713 / 2.8861 | 3.0824 / 2.8713 / 2.8861 |
| vehicle RMSE | 5.1422 / 4.4620 / 4.4717 | 5.1422 / 4.4620 / 4.4717 |
| vehicle ρ | +0.5393 / +0.5895 / +0.5795 | +0.5393 / +0.5895 / +0.5795 |
| bike-KSI MAE | 0.08668 / 0.08492 / 0.08409 | 0.08668 / 0.08492 / 0.08409 |
| bike-KSI ρ | +0.1797 / +0.2043 / +0.1681 | +0.1797 / +0.2043 / +0.1681 |

Arm 1b's provenance check also reproduces: `num_legs`-continuous pooled fit gives
sum_pred **1772.7**, MAE **3.4739**, +3.06 %, matching `README.md:162`'s recorded
v2 numbers (1772.7 / 3.47 / +3.1 %). That reconstruction is genuinely well done.

**Extension to 20 seeds** (they ran 10; I ran 0 to 19 and dropped no seed):

| mode | metric | Δ (A2−A1) mean ± SD | range | A2 wins |
|---|---|---|---|---|
| bike | MAE | **+0.00685 ± 0.00760** | [−0.00669, +0.01824] | 4/20 |
| bike | RMSE | +0.02418 ± 0.01170 | [−0.00116, +0.04133] | 1/20 |
| bike | ρ | **−0.04090 ± 0.01773** | [−0.08166, −0.00976] | **0/20** |
| ped | MAE | −0.00139 ± 0.00652 | [−0.01450, +0.00881] | 10/20 |
| ped | RMSE | +0.01538 ± 0.00977 | [−0.00376, +0.03461] | 1/20 |
| ped | ρ | −0.00307 ± 0.00881 | [−0.01917, +0.01388] | 8/20 |
| vehicle | MAE | **+0.01331 ± 0.00929** | [+0.00093, +0.03651] | **0/20** |
| vehicle | ρ | −0.01193 ± 0.00344 | [−0.01781, −0.00624] | **0/20** |
| bike-KSI | ρ | **−0.05182 ± 0.01792** | n/a | **0/20** |

Their per-seed bike-KSI ρ values match mine element-for-element for seeds 0 to 9
(0.2043/0.1681, 0.2025/0.1799, 0.2004/0.1423, 0.2014/0.1371, …). **The "9/9" is not
cherry-picking**: the excluded seed 3 also favours Arm 1 (0.2014 vs 0.1371), so
including it makes 0/10, and my clean 20-seed run gives 0/20.

### E5, reproduced to 6 dp
bike `num_legs` β = **0.671697**, SE = **0.188221**, p = 3.588e-04 →
exp(2β) = **3.8320× = +283.2 %**, 95 % CI **[+83.2 %, +701.4 %]**, 90 % CI [+106.3 %, +611.8 %].
Unconstrained full-categorical 6-vs-4: bike **−0.2350 (SE 0.8378, p = 0.779, exp = 0.7906)**,
ped −0.1814 (p = 0.757, 0.8341), vehicle +0.3750 (p = 0.483, 1.4550).
3-vs-4 legs: bike exp(β) = 0.2092 (p = 6.2e-06), ped 0.1815 (p = 2.1e-11), vehicle 0.2829 (p = 4.2e-18).
Six-leg sites: **3**, carrying **4 bike / 5 ped / 32 vehicle** crashes. All confirmed.

### E6, reproduced
bike degenerate designs: 12 columns, **rank 11**, condition numbers 1.54e+16
(`years_observed`) and 7.19e+16 (`log(years_observed)`), matching to 3 significant
figures. Control offset spec: 11 columns, rank 11, full rank. `years_observed` is
literally `[6]`.

### E7, reproduced including bootstrap SDs
NB-implied zeros **256.45 / 206.02 / 90.86** vs observed 256 / 204 / 89.
4000-draw parametric bootstrap: **256.5 ± 7.6 [244, 269]**, **205.9 ± 8.0 [193, 219]**,
**91.0 ± 7.4 [79, 104]**. Identical to their table.

### Cross-mode residual correlations, reproduced
bike-ped **0.2020** (p = 1.55e-04), bike-vehicle **0.0894** (p = 0.097, *not* significant),
ped-vehicle **0.1923** (p = 3.2e-04). Raw target correlations for context: 0.396 / 0.303 / 0.435.

---

## 2. Is the CV methodology sound?

**Leakage: none found. The share is computed on training rows only, and I verified
it in the code, not just in the prose.**
`e4_pooled_vs_permode.py:488` inside the primary CV loop and `:887` inside the
repeated-CV loop both compute `trd[TARGET].sum() / trd["total_crashes"].sum()` from
the training split; `:490 to 491` and `:888 to 889` do the same for the two KSI shares.
Nothing held-out enters a prediction. My independent implementation computes shares
the same way and lands on identical numbers, which it would not if theirs leaked.

**Fold consistency: correct.** All arms in a given seed consume the same `folds`
list built once per `random_state` (`:447`, `:875`); every arm is refit inside every
fold; both per-fold means ± SD and pooled out-of-fold metrics are reported.
E1/E3/E5 use a different scheme (stratified 5-fold plus a plain-KFold secondary) but
apply it identically to every spec in a comparison, which is what matters.

One design choice in E1/E3/E5 is worth naming: the patsy design matrix is built
**once on the full frame** and row-subset per fold (`e1_e3_e5_specification.md` §Methodology;
E2/E6/E7 do the same). Only the *column set* crosses the fold boundary, never a
response value, with `legs_cat == 5` at n = 13 and `arterial_class == 5` at n = 19,
a fold-local patsy build would silently change the design and make folds
incomparable. This is the right call and is a mild, correctly-disclosed
transductive step, not leakage.

**Spearman ρ on held-out folds with tied zeros: computed sensibly, but interpret with care.**
Predictions are continuous so ties occur only in `y`; `scipy.stats.spearmanr` assigns
mid-ranks, which is standard. Two consequences the reports handle unevenly:
- ρ is attenuated by the tie mass (256/346 bike zeros), so absolute values are not
  comparable to a continuous-outcome ρ. Fine for arm-vs-arm comparison, which is all
  they use it for.
- For **bike KSI**, `y` is effectively binary (16 events across 15 sites, 331 zeros),
  so ρ is a rank-biserial statistic ≈ AUC. E4 reports 4 of 5 folds yield a defined ρ
  (fold 4 has no held-out KSI events), honest, and it tells you how thin this is.
  E4 correctly evaluates ρ on the *pooled* OOF vector rather than averaging per-fold
  ρ's, which would have been much worse.

**One real methodological weakness, shared by E1, E3, E4 and E5.** Repeated CV over
seeds reshuffles fold assignments **on the same 346 sites**. Seed-to-seed SD therefore
measures fold-assignment noise, not sampling error. Three of the four experiments use
it as their noise floor ("6.6 seed-SDs", "9/9 seeds", "seed SD ~0.009"). It is the
wrong yardstick for any claim about which spec is better *in general*, see §3 and §4.

---

## 3. Are the "no difference" conclusions justified, or underpowered?

Both. And the reports do not distinguish them. I computed, for each comparison, the
paired site-level standard error of the difference in out-of-fold absolute errors
(`d_i = |y_i − μ_A,i| − |y_i − μ_B,i|`, SE = sd(d)/√346), which is the correct
yardstick for a paired A/B on the same sites, and the corresponding minimum
detectable effect at 80 % power, α = 0.05 two-sided (MDE = 2.802 × SE).

**E1 (KFold seed 0, vs production `a_log_aadt`):**

| comparison | ΔMAE | SE | t | MDE | MDE as % of MAE |
|---|---|---|---|---|---|
| ped: raw − log | +0.00271 | 0.00361 | +0.75 | 0.01011 | 1.4 % |
| ped: sqrt − log | +0.00022 | 0.00174 | +0.12 | 0.00487 | 0.7 % |
| ped: none − log | +0.00293 | 0.00744 | +0.39 | 0.02086 | 2.9 % |
| vehicle: raw − log | −0.00334 | 0.01555 | −0.21 | 0.04357 | 1.5 % |
| vehicle: sqrt − log | +0.00060 | 0.00780 | +0.08 | 0.02186 | 0.8 % |
| vehicle: none − log | +0.03870 | 0.03521 | +1.10 | 0.09866 | 3.4 % |

**E3 (vs production `a_centrality`):**

| comparison | ΔMAE | SE | t | MDE | MDE as % of MAE |
|---|---|---|---|---|---|
| aadt − centrality | +0.00278 | 0.00904 | +0.31 | 0.02534 | 4.3 % |
| both − centrality | +0.00422 | 0.00732 | +0.58 | 0.02050 | 3.4 % |
| neither − centrality | −0.00265 | 0.00456 | −0.58 | 0.01277 | 2.1 % |

**Reading.** E1 and E3 could have detected differences of roughly **1 to 3 % of MAE**
(E1) and **2 to 4 % of MAE** (E3). The differences they actually measured are
**0.06 to 0.6 %**. An order of magnitude below the detection floor. So "we measured no
difference" is correct; "the specs are equivalent" is *not* established. E1's own
framing (spread vs fold SD) reaches the right conclusion by a rough route; the
paired SE is the number it should have quoted.

**On coefficients, the power story is sharper and both reports overshoot it.**

| model | term | β | 95 % CI | per doubling of x | smallest detectable per-doubling effect |
|---|---|---|---|---|---|
| ped | `log_aadt` | +0.2230 | [−0.0574, +0.5033] | 1.167× [0.961×, **1.417×**] | **1.320×** |
| vehicle | `log_aadt` | +0.2264 | [+0.0174, +0.4353] | 1.170× [1.012×, 1.352×] | 1.230× |
| bike | `log_bike_centrality` | +0.1412 | [−0.1671, +0.4494] | 1.103× [0.891×, **1.365×**] | **1.357×** |
| bike | `log_aadt` | +0.2103 | [−0.2014, +0.6220] | 1.157× [0.870×, 1.539×] | 1.504× |

E1 writes "**Volume is not a significant predictor of pedestrian crashes at all**".
The ped data are consistent with anything from a 4 % *decrease* to a **42 % increase**
in ped crashes per doubling of AADT. Which comfortably contains an HSM-typical
volume effect. E3 writes "**neither one predicts bike crashes**"; the bike model
could only have detected an exposure effect of ≥1.36× per doubling of centrality.
Both should say *"we cannot detect an effect of the size this dataset can resolve"*,
not *"there is no effect."* This matters because the sentence as written invites a
reader to drop the exposure term, which the data do not license either.

---

## 4. Does E4's headline survive?

**The accuracy half survives and is stronger than they made it. The "pooled wins"
half does not.**

### What I confirm
1. I reproduce the seed-0 table cell-for-cell with independent code (table in §1).
2. I extend to 20 seeds with a fitter that never drops one: bike ΔMAE +0.00685
   (A2 wins 4/20), bike Δρ −0.0409 (0/20), vehicle ΔMAE +0.01331 (0/20),
   vehicle Δρ −0.0119 (0/20), bike-KSI Δρ −0.0518 (0/20). Sign stability is real.
3. **A confound E4 never tested, and it comes out in their favour.** The bike
   comparison pits a pooled model using `log_aadt` against a per-mode model using
   `log_bike_centrality`, which E3 shows predicts nothing. Is the bike result about
   pooling or about a bad exposure proxy? I fit two extra per-mode bike arms across
   all 20 seeds. Mean OOF MAE: production centrality **0.60701**, `log_aadt`
   **0.60603**, no exposure term **0.60380**, pooled × share **0.60016**. Mean OOF ρ:
   +0.3649 / +0.3717 / +0.3678 vs pooled **+0.4058**. Every per-mode variant loses to
   pooled. The finding is about pooling, not the exposure variable.
4. **The analysis all seven experiments missed, and it also favours E4.** Arm 1's
   structure *is* a nested restriction of Arm 2's: "mode shifts only the intercept"
   vs "every coefficient is mode-specific". Stacking the three modes long
   (1038 rows, 1730 events) and fitting NB with a common `log_aadt` volume term:

   | model | llf | k | AIC | BIC |
   |---|---|---|---|---|
   | H0, mode intercepts only (= pooled × share) | −1380.811 | 14 | **2789.62** | **2858.85** |
   | H1, full mode × predictor interactions (= per-mode) | −1372.236 | 34 | 2812.47 | 2980.60 |

   **LR = 17.15 on 20 df, p = 0.643.** ΔAIC **+22.85** and ΔBIC **+121.75**, both
   against the per-mode structure. Coefficient-by-coefficient heterogeneity
   (inverse-variance Q on 2 df): **not one of the 11 coefficients differs across
   modes**, smallest p is 0.106 (`C(arterial_class)[T.2]`, β = +0.887 / +0.557 /
   +0.196), then 0.273 (3-leg), 0.332, 0.385 (`bike_facility`), …, 0.998 (`log_aadt`).
   A cluster-robust Wald on the 20 interaction terms is marginal (χ² = 31.68,
   df = 20, **p = 0.047**). But a Wald test on 20 restrictions with 346 clusters
   over-rejects, and it disagrees with a well-behaved LR at p = 0.64.
   *Caveats: a single α is forced across modes (production fits 1.238/0.305/0.659),
   and the three rows per site are correlated. Both are disclosed above.*
   **This is genuinely independent, likelihood-based support for E4's conclusion
   that the split has nothing to model.**

### What does not survive
**a) The effects are not distinguishable from zero.** Paired site-level test on the
seed-0 OOF absolute errors:

| mode | mean d (A2 − A1) | SE | t | 95 % CI |
|---|---|---|---|---|
| bike | −0.00669 | 0.01141 | **−0.59** | [−0.02905, +0.01567] |
| ped | −0.00184 | 0.01407 | **−0.13** | [−0.02942, +0.02573] |
| vehicle | +0.01481 | 0.02053 | **+0.72** | [−0.02542, +0.05505] |

Even the 20-seed mean differences (+0.0069 bike, +0.0133 vehicle) sit at ~0.6 site-level
SEs. **"9/9 seeds" is not 9 replications**. Every seed reuses the same 346 sites, so it
measures fold-assignment stability, not sampling uncertainty. Conditional on this
sample, Arm 1 is reliably a hair better; on a fresh sample of intersections, nothing
here says it would be. The report's §A.3/§A.4 phrasing, "the dedicated model is
*worse* out of sample", "Arm 1 is **strictly preferable**", is not supported.
Its other phrasing, "delivered no measurable out-of-sample benefit", is exactly right.
The report uses both; only the second should ship.

**b) "8 of 11 cells" is arithmetically fair but evidentially oversold.** I count 8/11
too. But: (i) it comes from a *single* fold assignment (`random_state=0`), which the
same report says was "overturned" for bike by the multi-seed run, yet this table is
labelled "the most important table in the report"; (ii) the 11 cells are nowhere near
11 independent tests. They are 3 metrics on 4 overlapping targets, and Arm 2's
bike-KSI prediction is `μ_bike × a per-fold constant`, so the bike-KSI cells are
near-deterministic rescales of the bike cells; effective independent comparisons ≈ 3 to 4;
(iii) the margins are 0.25 to 1.1 % of MAE. To their credit it is not cherry-picked
across my 20 seeds the seed-mean favours Arm 1 in 10 of the same 11 cells (ped MAE the
exception at 10/20). But "8 of 11" reads as a tally of independent wins and is not one.

**c) The bike-KSI ρ comparison is not meaningful and should not be published.**
16 events across 15 sites; `y` is effectively binary so ρ ≈ AUC. Their own arm CIs are
[+0.120, +0.284] (Arm 1) and [+0.069, +0.257] (Arm 2), roughly 85 % overlap, as they
themselves note. The paired Δρ CI is [−0.0764, **−0.0011**], barely excluding zero, at
p = 0.021, from a bootstrap that **holds predictions fixed** (no refit inside the
resample. They disclose this, and it makes the CI too narrow). Against the *true* v2
(Arm 1b) it is a clean tie: Δρ = −0.0112, CI [−0.0469, +0.0240]. And the
decision-relevant metric is flat: out-of-fold top-k event capture differs by 0 to 2 events
of 16 at every k. One marginal p = 0.021 among ~62 spec-fits across seven experiments
is not a finding.

**d) "Cost robustness" rests on n = 1.** One fit of 200 (`permode_bike_seed3_fold0`)
failed the production ladder; Nelder-Mead rescued it, which makes it an optimiser-ladder
property, not a model property. The asymmetry (bike has ~14 events/parameter, pooled
~143) is a sound *a priori* argument; the single observation adds almost nothing to it.

**e) E4 tests two extreme points and recommends one.** 12 params/one model vs
36 params/three models. Mechanically, what E4 found is that **shrinking the bike model
toward the total-crash signal helps at 169 events**. A regularisation result.
Intermediate architectures were never fit: per-mode with the pooled prediction as an
offset, per-mode with a reduced parameter set, or partial pooling (the
`hierarchical_nb_sketch.py` that has never been run). Any of these could dominate both
arms. "Arm 1 is strictly preferable" is a stronger conclusion than a two-point A/B can
license.

**Bottom line on E4's recommendation against the shipped architecture:** it holds as
an *accuracy* claim in its weak form and I found two new pieces of evidence for it
(the exposure-variable control, and the coefficient-homogeneity test). It does not hold
in its strong form. The publishable sentence is: *"the three-model split buys no
measurable out-of-sample accuracy over one pooled model with the same leg encoding, and
no cross-mode coefficient heterogeneity is statistically detectable (LR p = 0.64; no
individual coefficient differs, min p = 0.106). So the split has to be justified on
interpretability grounds, which is legitimate but is not an accuracy argument."*
That last clause is E4's own, and it is the correct landing point.

---

## 5. Multiple comparisons

Roughly **62 distinct spec-fits** were compared across E1-E7 (E1 8, E2 6, E3 4, E4 11,
E5 9, E6 18, E7 6). Two observations:

**Multiplicity does not threaten a null.** E1, E3, E2-on-MAE, E7 and E4-on-MAE all
conclude "no difference". Testing more specs makes a null *more* credible, not less.
No one is reading significance into noise on those.

**The decisive claims survive correction.**
- **E5** continuous-vs-categorical: nested LR p = 8.8e-03 (bike) / 9.0e-08 (ped) /
  1.5e-07 (vehicle), AIC penalties 7.4 / 31.5 / 30.6, consistent direction in three
  modes. Ped and vehicle clear a Bonferroni threshold of 8e-04 over 62 comparisons by
  four orders of magnitude; bike alone would not, but is corroborated by the other two.
- **E2** α > 0: p = 4.4e-13 / 2.7e-04 / 5.7e-89 plus a 396-point AIC gap on vehicle
  and an 18-point coverage miss. Survives anything.
- **E7** is a boundary collapse (π̂ ≈ 1e-13, ΔLL ≈ 1e-9), not a p-value; multiplicity
  is not in play.
- **E6 Part A** is linear algebra (rank 11 of 12), not inference.

**One claim multiplicity kills, and it is the one E4 leans on hardest:** the bike-KSI
Δρ at **p = 0.021**. It is the only marginal p-value in the entire body of work, it
does not survive correction even for the 11 cells of E4's own table, and it carries
E4's headline about the project's headline metric. Drop it.

---

## 6. Do any conclusions conflict?

**a) E4 and E5 on leg encoding are NOT independent convergent evidence.**
E5 compares `num_legs` vs `legs_cat` inside each per-mode model. E4's Arm 1b-vs-Arm 1
compares the *same two encodings* inside a pooled model on `total_crashes`. Which is
`bike + ped + vehicle` minus 10 double-counted crashes, on the *same 346 sites*, with
the same predictor, driven by the same phenomenon (3-leg sites are ~4 to 5× safer than
4-leg, so a straight line through 2→6 legs misfits and then extrapolates). E4's Arm 1b
result is substantially E5's result aggregated over modes. It is a **consistency check,
not a second test**, and the case study must not present it as two confirmations. The
effect is large enough that the conclusion is unaffected. But the claimed weight of
evidence would be roughly doubled by the mistake.

**b) E4's Part C reason #3 is overstated by its own project's data.** It says the three
mode targets "are correlated at a site by construction." Measured cross-mode residual
correlations are **0.202 / 0.089 / 0.192**, and bike-vehicle is not even significant
(p = 0.097). The double-counting is 10 crashes of 1730. "By construction" should be
"weakly correlated". Reasons #1 (different random variables) and #2 (different event
totals) are correct and fully sufficient, so the conclusion, don't sum AIC across the
three fits, stands.

**c) On the cross-mode residual correlations and E4's pooling conclusion, reasoned through.**
The naive reading ("modes barely co-vary, so they need separate models") is backwards,
and so is the opposite naive reading. Residual correlation is about the *unexplained*
part; Arm 1's constraint (μ_mode,i = share_m × μ_total,i) is a restriction on the
*systematic* part. It forbids mode × predictor interactions. Those are different
quantities, and low residual correlation would be observed under both hypotheses. So:

- **On their own, 0.202 / 0.089 / 0.192 neither support nor undercut E4**. They are
  close to orthogonal to the pooling question.
- What they *do* say is that once the shared predictors are removed, the three modes
  at a site are close to conditionally independent, with a small positive residue.
  That small positive residue (2 of 3 significant) is the signature of a **site-level
  random effect that neither arm models**, mild evidence pointing at a *third*
  architecture (partial pooling / hierarchical NB), which is exactly what
  `experiments/hierarchical_nb_sketch.py` specifies and which has never been run.
- The thing that actually settles the pooling question is the coefficient-homogeneity
  test in §4.4: LR = 17.15 on 20 df, p = 0.643, no single coefficient differing.
  **That supports E4.** The residual correlations should be reported as context, not
  as evidence either way.

**d) A presentational conflict to fix.** E1 says `log_aadt` is not significant for ped
(p = 0.119); E4's winning arm is a pooled model whose only volume term is `log_aadt`.
There is no real contradiction. The pooled target carries 1720 events, so the
coefficient is far better determined there. But a case study that says "volume doesn't
predict crashes" on one page and "the pooled log_aadt model beats three models" on the
next needs to explain the event-count difference or it will read as incoherent.

---

## 7. Things they got outright wrong

1. **E1, factual error in the prose.** `e1_e3_e5_specification.md:199` states
   "Sites above 20,000: 24. Above 25,000: 5." The correct values are **27** and **2**
   and *its own* `e1_results.json` records `n_above_20000: 27, n_above_25000: 2`.
   Two numbers were mistyped into a document whose opening line is "**No figure in this
   document is estimated, rounded from memory, or invented.**" Everything else in E1
   checked out; this is a transcription slip, not a fabrication, but it must be fixed
   before publication because that sentence is a strong claim.
2. **E4: "worse out of sample" / "strictly preferable".** Not supported
   t = −0.59 / −0.13 / +0.72 against site-sampling noise. Only "no measurable benefit"
   is defensible.
3. **E4 Part C #3:** modes "correlated at a site by construction", measured 0.089 to 0.202,
   one not significant.
4. **E4/E5: seed-to-seed SD used as the noise floor.** Seeds reshuffle folds on the same
   346 sites; they are not replications. E5's "≈ 6.6 seed-SDs" is **≈ 2.3 site-level SEs**
   (my paired test: bike t = +2.27, ped t = +2.13, vehicle t = **+1.60**, the last not
   significant). Evidence strength inflated ~3×. E5's *conclusion* is unharmed, see #9.
5. **E1: "Volume is not a significant predictor of pedestrian crashes at all."**
   Conflates non-significance with absence. CI per AADT doubling: [0.961×, 1.417×].
6. **E3: "neither one predicts bike crashes."** Same conflation; MDE is 1.36× per
   doubling of centrality. The correct statement is that the effect is not identifiable
   at 169 events. Which E3 does say elsewhere, and should say only that.
7. **E4's internal inconsistency:** the seed-0 "8 of 11" table is called "the most
   important table in the report" three sections after the report explains that seed-0
   was overturned by the multi-seed analysis.
8. **E4's Part D top-10 framing.** "A city funding its worst 10 intersections would send
   crews to a materially different set" (4/10 overlap) is technically true but, on their
   own numbers, 9 of the 10 largest rank shifts are at sites with 0 observed bike KSI
   and μ differences of hundredths of an event. The report says this two paragraphs
   later; the headline sentence should not outrun it.
9. **What E5 got *right* but for the wrong reason. And the fix makes it stronger.**
   Both leg encodings are nested inside the full leg categorical, so there is a clean
   LR test they never ran. I ran it:

   | mode | top-coding vs full cat | continuous vs full cat |
   |---|---|---|
   | bike | LR = 0.230, 1 df, **p = 0.632**, restriction not rejected | LR = 11.63, 3 df, **p = 0.0088**, rejected |
   | ped | LR = 0.146, 1 df, **p = 0.703** | LR = 35.63, 3 df, **p = 9.0e-08** |
   | vehicle | LR = 0.060, 1 df, **p = 0.806** | LR = 34.63, 3 df, **p = 1.5e-07** |

   Top-coding is not rejected in any mode; the linear slope is decisively rejected in
   all three. This is a properly-powered, non-CV test that says exactly what E5
   concluded. **E5 should re-justify its verdict on these numbers and retire the
   seed-SD framing.**
10. Confirming `MODEL_NOTES.md`: `fit_risk_model.py:14` says "vehicle-only-KSI = 23";
    the data give **25**. The argument it supports is unaffected.

---

## 8. Ranked publication list

### Safe to publish as written (after the E1 typo fix)
1. **E7, zero inflation is not warranted.** π̂ collapses to 1e-13/5e-14/1.8e-14,
   ZINB worse by ΔAIC +2.000 / ΔBIC +5.846, and plain NB reproduces the observed zeros
   (256 vs 256.45, 204 vs 206.02, 89 vs 90.86, all near the centre of a 4000-draw band).
   The refusal to run an LRT (π = 0 ⇔ logit intercept = −∞) and the suppression of the
   degenerate Vuong variants are the right calls and are worth mentioning as such.
2. **E2, NB over Poisson, on variance not on point accuracy.** LR p = 4.4e-13 / 2.7e-04
   / 5.7e-89; vehicle Poisson 90 % intervals cover 76.6 % with 81 of 346 sites outside;
   CV point accuracy indistinguishable (p = 0.478 / 0.256 / 0.827). The explanation
   α moves the variance, not the mean, so MAE is nearly blind to family, is correct and
   is the most quotable insight in the whole set.
3. **E6 Part A, offset-vs-covariate is not identifiable here.** Design rank 11 of 12,
   exposure column an exact constant multiple of the intercept, likelihood flat to 1e-13
   along the collinear direction. Plus the operational finding that statsmodels returns
   `converged=True` with finite standard errors in 4 of 6 cases. Publish the offset as a
   *convention* (HSM), explicitly not as a validated choice.
4. **E5, top-coded categorical over the continuous per-leg slope.** Publish, but
   **re-justify with the nested LR test** (p = 0.0088 / 9.0e-08 / 1.5e-07) rather than
   "6.6 seed-SDs". Publish the docstring correction as well: the "+280 %, CI +80 to 700 %"
   numbers reproduce exactly (+283.2 %, [+83.2 %, +701.4 %]) but the stated *reason* is
   false. There are **3** six-leg sites carrying 4/5/32 crashes, and unconstrained they
   estimate exp(β) = **0.79**, the opposite sign. That correction is a better story than
   the original claim.
5. **E1's 50,000-AADT rebuttal.** 1.49× (vehicle) and 1.12× (ped), not "astronomical";
   observed max 41,808; zero sites above 50,000; only one above 30,000. Fix
   "24 / 5" → **"27 / 2"** first.
6. **E6's optimiser finding (from E1).** The production newton→BFGS ladder returns a
   diverged optimum 114 AIC points wrong on a badly-scaled predictor. Real, useful, and
   I confirmed it independently by rescaling.

### Publish only with hedging
7. **E1 / E3 "no winner".** Frame as *underpowered*, with the MDEs: ~1 to 3 % of MAE (E1),
   ~2 to 4 % of MAE (E3), 1.32× per AADT doubling for ped, 1.36× per centrality doubling for
   bike. Never as "volume does not predict crashes."
8. **E4's headline.** Publish the weak form only: *no measurable out-of-sample accuracy
   benefit from the split, and no detectable cross-mode coefficient heterogeneity
   (LR p = 0.64)*. Cite the coefficient-homogeneity test and the exposure-variable
   control from this review, not the seed counts. Never "pooled is better."
9. **E4's Arm 1b v2 reconstruction.** Excellent provenance work (sum_pred 1772.7, MAE
   3.4739 vs README's 1772.7 / 3.47), publish it, but as a *consistency check* on E5's
   leg-encoding finding, not as independent evidence.
10. **E6 Part B.** Always carrying the SYNTHETIC label. It validates the estimator on a
    design where the true coefficient is 1 by construction; it cannot speak to the real
    6-year window, and the bike arm is uninformative on its own (between-seed SD 0.496 >
    mean nominal SE 0.352).

### Do not publish
11. **E4's bike-KSI Δρ = −0.036 / −0.054 "in 9/9 seeds".** 16 events in 15 sites,
    arm CIs overlapping ~85 %, a fixed-prediction bootstrap the authors themselves flag
    as too narrow, p = 0.021 among ~62 comparisons, a genuine tie against the true v2,
    and identical top-k event capture. This is the weakest link in the strongest-claiming
    report.
12. **"The per-mode split cost robustness."** One non-converged fit of 200, rescued by
    Nelder-Mead.
13. **Any AIC comparison between the pooled and the summed per-mode fits.** E4 already
    says this and is right; keep it in the "do not use" column, not in a results table.

---

## 9. The one caveat I would attach before any of this goes public

**Every result here is conditioned on a single sample: 346 Capitol Hill arterial
intersections over one 6-year window. Repeated cross-validation reshuffles folds
*within that fixed sample*, so seed-to-seed standard deviations. The yardstick E4 and
E5 use for their strongest claims ("9/9 seeds", "6.6 seed-SDs"), measure
fold-assignment noise, not sampling error, and are 2 to 5× too small.** Measured against
the correct paired site-level standard error, none of E1's, E3's or E4's differences is
distinguishable from zero (|t| ≤ 1.10 throughout), and E5's is ~2.3 SEs rather than 6.6.
The honest headline for the whole body of work is: *at n = 346 with 169 bike events, this
dataset can rule out badly-wrong specifications (a linear per-leg slope, Poisson variance,
zero inflation) but cannot adjudicate between reasonable ones (volume functional form,
bike exposure proxy, pooled vs per-mode).* That is a genuinely valuable and publishable
finding. It is just a different finding from "we determined which spec is better."

---

## 10. What I could not verify

- The committed `data/model/*.pkl` files (pandas-version pickle incompatibility, as
  MODEL_NOTES records). Every number here comes from my own in-memory refits.
- E4's per-fit convergence ledger beyond what its JSON records (I confirmed
  `n_fits = 200`, `n_failed = 1`, tag `permode_bike_seed3_fold0`, and that the seed-3
  exclusion does not change the conclusion). My harness uses a 3-rung ladder by design,
  so it cannot reproduce production's failure counts.
- E6 Part B's synthetic construction end-to-end (the exposure draw, allocation and
  thinning). I verified Part A fully and read Part B's construction; I take its
  seed-0 and 10-seed tables on the code's word. Since Part B is explicitly fenced off
  from any real-data conclusion, this does not affect any verdict.
- E4's 2000-resample paired bootstrap CIs. I verified the inputs (per-seed ρ values,
  which match mine exactly) and the recorded outputs in `e4_results.json`, and I
  re-derived the point estimates; I did not re-run the bootstrap itself.
- E1's stratified-CV tables. I re-ran the plain-KFold arm (which matches: ped
  0.7271/0.7298/0.7273/0.7300, vehicle 2.8861/2.8828/2.8867/2.9248) and confirmed the
  cross-scheme ranking inversion they report, but did not re-run the stratified scheme.
