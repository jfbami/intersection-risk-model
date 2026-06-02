# project-cycle-group

Capitol Hill (Seattle) intersection crash-risk model and interactive map.

A Negative Binomial Safety Performance Function scores the **346 arterial intersections** in Capitol Hill on their expected bike-KSI rate, following AASHTO Highway Safety Manual Chapter 12 methodology. Results are served by a FastAPI backend and rendered by a Next.js + Mapbox frontend, with a drill-in panel that surfaces the model's contributing factors per intersection.

## What the model does

- **Family:** statsmodels `NegativeBinomial` (NB2), log link, `offset = log(years_observed)` for the 6-year exposure window (2018–2023).
- **Scope:** arterial intersections only — `arterial_class >= 1` with positive AADT. Local-access streets are excluded per HSM Chapter 12 facility-type stratification. In Capitol Hill the local tail is 63% zero-crash and contributes 1 of 17 bike-KSI events; mixing two physically different facility classes would force one coefficient set to describe both.
- **Formula:**
  ```
  total_crashes ~ is_signalized + num_legs + max_speed_limit
                + bike_facility + C(arterial_class) + log_aadt
  ```
- **EB adjustment:** AASHTO HSM Part C Empirical-Bayes shrinkage (`w = 1 / (1 + α·μ); eb = w·μ + (1−w)·N`) pulls extreme model predictions toward observed counts at sites with enough data.
- **Headline metric served to the UI:** `expected_bike_ksi_per_year` with a 90% credible interval (`ci_low` / `ci_high`), derived by applying a Poisson-Gamma direct EB to bike-KSI counts using the all-crash NB prediction × citywide bike-KSI share as the prior. Plus a `top_contributors` list — the 3 features pushing this site farthest from the modelled reference, each labeled with `+X%` or `−X%`.
- **Secondary fields:** `risk_score` (0–100 percentile rank) and `risk_tier` (`very_high` ≥ 90, `high` 70–89, `moderate` 40–69, `low` 20–39, `very_low` < 20), kept for sorting only.

## Why `log(AADT)` and not raw `AADT`

This is a functional-form choice, not a scaling choice — important enough to call out because it's easy to misread.

With a log-link GLM, `log(μ) = β₀ + … + β_aadt · <something with AADT>`:

- **Raw AADT** ⇒ `μ ∝ exp(β · AADT)` — crashes grow *exponentially* in volume. A 50,000-AADT site would have astronomical predicted crashes compared to a 5,000-AADT site, and any extrapolation breaks. Not physical.
- **log(AADT)** ⇒ `μ ∝ AADT^β` — crashes follow a *power law* in volume. With β < 1 we get the well-documented **sub-linear "safety-in-numbers" effect**: doubling volume multiplies crashes by 2^β, not 2.

Concrete: at our fitted `β = 0.26`, doubling AADT multiplies expected crashes by **2^0.26 ≈ 1.20** (a 20% increase, not 100%). Drivers slow down in denser traffic, cyclists adjust routes, pedestrian behavior shifts — the per-vehicle risk drops as volume rises.

This power-law form is what the AASHTO HSM SPFs are written in: `μ = exp(β₀) · AADT_major^β₁ · AADT_minor^β₂ · CMFs · years`. Taking log of both sides yields exactly our linear predictor with `log(AADT)` as the term. The log specification *is* the HSM specification — every state-DOT SPF in active use is the same shape.

> **Note on scaling:** for unpenalized MLE on a GLM, predictor scale doesn't affect the fit — the coefficient absorbs it. So this isn't a normalization trick. (Scaling *would* matter for penalized regression, distance-based methods, or gradient-based optimizers — different contexts.)

## Vision Zero framing

The pipeline emits five severity counts per intersection from SDOT's `MAXSEVERITYCODE`:
- `injury_total` — any injury collision (code ≥ 2)
- `ksi_total` — Killed or Seriously Injured (code ≥ 3, the Vision Zero target metric)
- `fatal_total` — fatal only (code = 4)
- `ped_total` / `bike_total` — count of crashes with `PEDCOUNT > 0` / `PEDCYLCOUNT > 0`

These appear in the Vision Zero scorecard at the top of the map and per intersection in the drill-in panel. Over the modelled arterial set, 2018–23 saw **1,720 crashes, 169 bike crashes, 16 bike-KSI events**. Restricting to arterials drops only 5.9% of bike-KSI events while removing 286 mostly-zero local intersections from the fit. **Caveat:** SDOT's per-crash `PEDCOUNT` / `PEDCYLCOUNT` fields are sparsely populated for records post-2018, so the displayed ped/bike count is conservative — `snap_crashes.py` falls back to keyword matching on `SDOT_COLDESC` to recover the gap.

## Coefficient interpretation (current fit, `nb_v2_arterial_aadt`)

Each `β` is on the log-rate scale; `exp(β)` is the multiplicative effect on the expected crash count, all else equal.

| Term | β | exp(β) | Reading |
|---|---|---|---|
| `log_aadt` | +0.257 | 1.29 per e-fold | **+20% per AADT doubling** (sub-linear "safety in numbers") |
| `is_signalized` | +1.159 | 3.19× | Signals are placed at the busiest, highest-conflict junctions. Selection bias after AADT, not causation. |
| `num_legs` | +0.591 | 1.81 per leg | Each extra approach adds conflict points (geometry) |
| `C(arterial_class)[T.2]` Minor | +0.421 | 1.52× | +52% over principal arterial (baseline) |
| `C(arterial_class)[T.3]` Collector | +0.246 | 1.28× | +28% over principal |
| `C(arterial_class)[T.5]` Other | +1.022 | 2.78× | Heterogeneous catch-all; high but noisy |
| `max_speed_limit` | −0.108 | 0.90 per mph | Suspicious sign — posted speed has narrow range within arterials and is a poor proxy for operating speed |
| `bike_facility` | −0.306 | 0.74× | **Protective**: bike facility nearby → −26%. Blended across protected / painted / sharrow. |
| α (NB dispersion) | 0.640 | — | Meaningful overdispersion; NB is the correct family |

**Read with care:**
- `is_signalized` is *not* a causal estimate of "signalizing causes crashes" — signals correlate with high-conflict places. A genuine causal estimate needs a before-after study with comparison group (CMF clearinghouse / Phase 5).
- `max_speed_limit`'s small negative coefficient is residual confounding after AADT, plus the fact that posted ≠ operating speed. Don't pitch it as "lower speed limits cause more crashes."
- `bike_facility`'s −26% is the literature-direction protective effect, but blends protected lanes (~−60% in CMFs) with sharrows (~−5%). Useful for ranking; not yet "what would happen if we add a protected lane here."

## Calibration and verification

- **Calibration:** sum predicted = 1772.7 vs. sum actual = 1720 (+3.1% gap; HSM threshold is ±15%).
- **MAE per intersection (6-year window):** 3.47 crashes.
- **90% predictive coverage:**
  - all-crash: 95.1% (nominal 90%; slight over-coverage from discrete NB step quantiles, normal)
  - bike-KSI: 98.6% (conservative — Phase 1 borrows the all-crash α; Phase 2's bike-specific model will tighten this)
- **Spearman rank correlation (predicted vs. observed bike-KSI):**
  - top-20 sites: ρ = +0.30
  - all 346 sites: ρ = +0.28

The rank correlation is modest because 17 bike-KSI events across 346 sites in 6 years is a fundamentally noisy sample, not because the model is mis-specified. Improving it requires either (a) cyclist exposure data (Strava Metro / bike counts), (b) a bike-specific target with more events (all bike crashes, not just KSI), or (c) expanding scope to all of central Seattle. The path is more data, not a fancier model class.

## Recommended treatments (Phase 5 — SPF × CMF prescriptive analysis)

The model alone is **descriptive**: it tells you *where* crashes happen and *what correlates* with them, but its coefficients are observational and not safe to read as causal treatment effects. To answer "what should we *do*?", we apply the canonical HSM Part C two-stage method:

```
expected_bike_KSI_prevented_per_year(site, treatment) =
    expected_bike_ksi_per_year(site) × (1 - CMF(treatment))
```

The SPF (our NB model) supplies the **site-level baseline**; published **Crash Modification Factors** (CMFs) supply the **causal treatment-effect multiplier**. CMFs come from before-after studies with comparison groups, peer-reviewed by FHWA and rated 1–5 stars in the CMF Clearinghouse.

[`data/cmf_library.json`](data/cmf_library.json) is generated by [`pipeline/build_cmf_library.py`](pipeline/build_cmf_library.py) from a direct export of the [FHWA CMF Clearinghouse](https://www.cmfclearinghouse.org/) (source date stamped in the JSON file). The ingester filters to **bike-involved, intersection-related, approved, non-rural studies**, then for each treatment curated for our context aggregates across all qualifying studies. Every entry carries the Clearinghouse `cmid`, full study citations (often multiple), per-study AMFs, the SE method used, and an explicit `direction` label.

**Aggregation rule:** variance-weighted average only when *every* study in the group reports a standard error; otherwise a simple mean with across-study sample variance for the CI. This avoids the systematic bias of over-weighting whichever paper happened to report SEs (often a single paper reporting sub-analyses).

**Current library — sourced from CMF Clearinghouse export 2025-11-10:**

| Treatment | Studies | CMF (90% CI) | Direction | Applies when |
|---|---|---|---|---|
| Install cycle track / protected bike lane (cmid 1080) | 4 | 0.43 (0.14–0.71) | helpful | `bike_facility = 0` |
| Raised bicycle crossing (cmid 1042) | 1 | 0.49 (0.30–0.68) | helpful | any |
| Offset cycle track w/ cyclist priority (cmid 1037) | 1 | 0.55 (0.28–0.82) | helpful | `bike_facility = 0` |
| Install painted bike lane (cmid 543) | 4 | 0.56 (0.07–1.04) | helpful | `bike_facility = 0` |
| Prohibit right-turn-on-red (cmid 114, inverted) | 8 | 0.58 (0.54–0.63) | anti-treatment ⇒ prevention | `is_signalized = 1` |
| Bike lane at signalized intersection (cmid 872) | 8 | 1.08 (0.78–1.37) | anti-indication | `bike_facility = 0`, `is_signalized = 1` |
| Convert yield→signalized (cmid 1421) | 10 | 1.04 (0.64–1.45) | anti-indication | `is_signalized = 0` |
| Convert to single-lane roundabout (cmid 1283) | 23 | 1.40 (1.25–1.56) | anti-indication | any |

Notice what the actual evidence says:
- **Cycle tracks and raised crossings are the clear winners** — strong, consistent reductions across multiple studies.
- **Prohibiting RTOR has 8 studies of consistent evidence** that permitting it raises bike crashes ~77%, so prohibition is recommended with high confidence.
- **Roundabouts INCREASE bike crashes** by ~40% across 23 studies — this is the well-documented bike-roundabout paradox (roundabouts are safer for cars and pedestrians but more dangerous for cyclists due to entry/exit conflict geometry). They appear in the library as an explicit anti-indication so SDOT planners considering a roundabout retrofit see the bike-specific evidence.
- **Bike lanes at signalized intersections show no net effect** (8 studies, CMF ~1.08). The likely interpretation is that the facility raises cyclist exposure roughly as much as it reduces per-cyclist risk — which is exactly why bike-volume data (Strava Metro) is the critical missing covariate for the next phase.

**How the recommendations rank** ([`pipeline/treatments.py`](pipeline/treatments.py)): for each intersection, filter to applicable treatments, compute `prevented = prediction × (1 − cmf)`, sort descending, take top 3. Anti-indications (CMF > 1) naturally sort to the bottom because their `prevented` value is negative — they're informational, not endorsement.

**Why this is more honest than reading model coefficients as treatment effects:** the SPF's `is_signalized` coefficient (+219%) is *selection bias* — signals correlate with the busiest, highest-conflict junctions. Its `bike_facility` coefficient (−26%) is the right direction but blends protected/painted/sharrow into one category. The CMF library, by contrast, gives **per-treatment causal estimates** from controlled before-after studies, with explicit per-study sourcing.

**Honest limitations:**
- **No cyclist-volume denominator.** The CMFs are measured on raw crash counts. When a facility brings more cyclists in, raw counts can rise even though per-cyclist risk falls. The "bike lane at signalized intersection" finding is almost certainly an instance of this. Bike-volume integration (Strava Metro) is the highest-impact data-acquisition step we have not yet done.
- **Preconditions are coarse.** No checks on right-of-way availability, lane count, AADT range of the original study, etc. Treat the recommended-treatment list as a *first-pass screen*, not an engineering scope of work.
- **No cost dimension yet.** A protected bike lane prevents 5× the bike-KSI of a raised crossing but costs 10× as much. **Cost-per-bike-KSI-prevented** is the natural next addition.
- **Some entries are single-study.** Raised bicycle crossing and offset cycle track each rest on one European urban paper. The point estimates are credible but the CIs reflect within-study sampling error only, not between-study heterogeneity.
- **Re-pull annually.** [`pipeline/build_cmf_library.py`](pipeline/build_cmf_library.py) is re-runnable when the Clearinghouse refreshes its database (typically annually).

## Counterfactual / "what-if" intervention modeling (developer tool)

[`pipeline/counterfactual.py`](pipeline/counterfactual.py) is the *secondary* prescriptive tool, useful for inspecting the model's own view: given an intersection's current feature row plus hypothetical overrides (toggle signal, add bike facility, downgrade arterial, change speed limit, change number of legs, change AADT), it returns the model's predicted Δ in expected crashes/year.

With AADT in the model, design-effect coefficients are roughly 80% smaller than the v1 (no-AADT) fit, so counterfactual Δ now reads as approximately the *design* effect rather than the *design + missing-volume* effect. But these Δ values are still **observational coefficients**, not causal. **The Phase-5 CMF-based recommendations above are the right tool for stakeholder-facing "what should we do" questions.** This module remains useful for developers understanding what the model is doing under the hood.

## Pipeline

Run in order. The first script downloads ~5 MB from Seattle's GeoData ArcGIS portal (cached as GeoJSON afterward). On Windows, prefix each command with `python -X utf8` to avoid `cp1252` encoding errors on Unicode print statements.

```sh
python seattle_arcgis.py                # download raw layers -> data/raw/
python -m pipeline.build_intersections  # 651 intersection points -> data/intermediate/intersections.parquet
python -m pipeline.snap_crashes         # crashes per intersection per year
python -m pipeline.assemble_features    # feature matrix (signal, legs, speed, bike, arterial)
python -m pipeline.fit_risk_model       # NB regression -> data/model/nb_v2_arterial_aadt.pkl
python -m pipeline.score_risk           # EB + percentile ranks -> data/intermediate/intersection_scores.parquet
```

## Running the app

The stack is FastAPI (port 8000) + Next.js (port 3000).

```sh
pip install -r requirements.txt
uvicorn api_server:app --port 8000 --reload
```

In a second terminal:

```sh
cd frontend
npm install
cp .env.local.example .env.local   # add your Mapbox public token
npm run dev
```

Open http://localhost:3000.

- **Vision Zero scorecard** at the top: total crashes, injuries, KSI, fatalities, ped/bike-involved — recomputed over the current tier filter.
- **Map**: all 651 intersections, colored by `risk_tier`, sized by `risk_score`. Hover for tooltip, click to populate the drill-in panel.
- **Layer toggles**: tier filter, bike-facilities overlay.
- **Drill-in panel**: score badge, expected-vs-actual crash counts, severity breakdown, feature table.

## File layout

```
seattle_arcgis.py             ArcGIS REST fetcher (one function per Seattle dataset)
api_server.py                 FastAPI: /api/intersections, /api/bike-facilities
data/
  cmf_library.json                                Curated CMFs (generated by build_cmf_library.py)
  raw/cmf_clearinghouse_2025-11-10.csv            FHWA CMF Clearinghouse export (9,777 CMFs)
  raw/cmf_clearinghouse_dictionary_2025-11-10.pdf Clearinghouse data dictionary
pipeline/
  __init__.py                 Makes pipeline a proper Python package
  columns.py                  Shared severity-column constants
  build_intersections.py      Street endpoints -> clustered intersection points
  snap_crashes.py             Crashes within 25 m -> per-intersection counts + severity sub-counts
  assemble_features.py        Per-intersection features (signal, legs, speed, bike, arterial, AADT)
  fit_risk_model.py           NB2 SPF fit + predictions (arterial-only, log(AADT) included)
  score_risk.py               EB shrinkage + bike-KSI proxy + 90% credible intervals + treatments
  contributors.py             Per-intersection top-contributors derivation
  treatments.py               Load CMF library, filter applicability, rank per-site treatments
  build_cmf_library.py        Re-runnable ingester: Clearinghouse CSV -> data/cmf_library.json
  counterfactual.py           Load pkl + predict at hypothetical feature configurations
  tests/                      pytest suite: EB math, mode inference, contributors, treatments, calibration
frontend/
  app/                        Next.js app router pages
  components/                 Map, DrillInPanel, MetricPrimitives, VisionZeroScorecard, ...
  lib/                        Typed API client + shared types
```

## Known limitations

- **Phase-1 bike-KSI proxy.** Headline metric scales the all-crash NB by the citywide bike-KSI share and borrows the all-crash dispersion α. This is honest as a Phase-1 placeholder (intervals over-cover, on the conservative side) but Phase 2 will fit a model directly on bike-KSI counts.
- **Bike volume / cyclist exposure is missing entirely.** AADT is now in the model, but bike volume isn't. Per-bike risk is what Vision Zero actually cares about; without a cyclist denominator we're modelling "where do bike-KSI events land" rather than "where is a cyclist most likely to be killed." Strava Metro is the obvious unlock.
- **All-crash target, not bike-specific.** The NB target is `total_crashes`. Bike-KSI enters only through the scoring step's share multiplier. Phase 2: refit separately on `bike_total` and `ped_total`.
- **Capitol Hill scope only.** 346 modelled arterial intersections in a single neighborhood. Expanding to all-Seattle is a refit + re-snap, not an app change, and would more than 10× the bike-KSI sample.
- **AADT is AWDT, not true AADT.** Seattle publishes Annual Weekday Daily Traffic. Typically ~5–10% lower than true AADT; standard local-practice substitute.
- **AADT coverage on Capitol Hill class-5 ("other arterial") is 50%.** 19 of 38 class-5 intersections are dropped from the fit because they have no usable AADT.
- **6-year observation window (2018–2023).** Re-fit when more recent SDOT collision data lands.
- **SDOT ped/bike severity fields are sparse post-2018.** `PEDCOUNT` / `PEDCYLCOUNT` are essentially zero post-2018; `snap_crashes.py` falls back to keyword matching on `SDOT_COLDESC` to recover them.
- **Coefficients are associative, not causal.** Especially `is_signalized` (selection bias toward busy junctions) and `max_speed_limit` (posted ≠ operating speed). Causal treatment effects require before-after studies with comparison groups (Phase 5 / CMF Clearinghouse).

## Archive

`project_plan updated.pdf` is the current project plan; `detailed plan.pdf` (referenced in older notes) and the VLM and Supabase-export sections in earlier drafts are no longer the implementation path. The README above is the source of truth.
