# Intersection Risk Model — Capitol Hill, Seattle

An interactive map and statistical model that scores Capitol Hill's intersections by **bike-crash risk** and tells you, per intersection, which engineering fixes the evidence supports.

Built on Seattle's own crash records (SDOT, 2018–2023) and the AASHTO Highway Safety Manual methodology that state DOTs use to prioritize safety projects.

---

## See it in action

> **Screenshots and a short walkthrough of the live app go here.** Drop images into `docs/screenshots/` and a recording link (Loom / YouTube / .mp4) just below.

<!--
Suggested layout — replace the placeholders with real assets once recorded.

### 1. Map overview
![Map overview — 651 intersections colored by risk tier](docs/screenshots/01-map-overview.png)

### 2. Vision Zero scorecard
![Scorecard — totals for crashes, injuries, KSI, ped/bike](docs/screenshots/02-scorecard.png)

### 3. Drill-in panel
![Drill-in panel — predicted vs actual, top contributors, recommended treatments](docs/screenshots/03-drill-in.png)

### Walkthrough video
[![Watch the 2-minute walkthrough](docs/screenshots/video-thumbnail.png)](https://www.loom.com/share/REPLACE_WITH_LINK)
-->

---

## What this project is, in plain English

Seattle has a **Vision Zero** commitment: eliminate traffic deaths and serious injuries. Of the city's 1,720 reported crashes in Capitol Hill from 2018–2023, **16 left a cyclist killed or seriously injured ("KSI")**. With limited budget for safety upgrades, the question becomes: *which intersections deserve attention first, and what should we actually build there?*

This project answers both:

1. **Risk score per intersection.** A statistical model (the same family the U.S. Highway Safety Manual recommends) reads each intersection's features — traffic volume, signalization, number of approaches, presence of a bike facility, arterial class, posted speed — and predicts an **expected bike-KSI rate per year**, with an uncertainty range.
2. **Recommended fixes per intersection.** For each site, the app filters the FHWA's **Crash Modification Factor (CMF) Clearinghouse** — a peer-reviewed catalog of before-and-after safety studies — to find the treatments that apply, ranks them by predicted bike-KSI prevented, and shows the top three.

A FastAPI backend serves the scored intersections; a Next.js + Mapbox frontend lets you explore them visually.

---

## At a glance

| | |
|---|---|
| **Scope** | 346 arterial intersections in Capitol Hill, Seattle |
| **Crash window** | 6 years, 2018–2023 |
| **Crashes observed** | 1,720 total · 169 bike · 16 bike-KSI |
| **Model family** | Negative Binomial regression (HSM Chapter 12 Safety Performance Function) |
| **Calibration vs. observed** | Predicted total 1,772.7 vs. actual 1,720 — within 3.1% (HSM threshold ±15%) |
| **Treatment library** | 8 curated CMFs from FHWA Clearinghouse, 2025-11-10 export |
| **Stack** | Python / statsmodels · FastAPI · Next.js · Mapbox |

---

## Quick start

The stack is two services: a FastAPI backend (port 8000) and a Next.js frontend (port 3000). Pipeline outputs are already committed to `data/`, so you can run the app on a fresh clone without re-running the model.

### 1. Backend (Python ≥ 3.10)

```sh
pip install -r requirements.txt
uvicorn api_server:app --port 8000 --reload
```

### 2. Frontend (Node 18+)

```sh
cd frontend
npm install
cp .env.local.example .env.local      # then paste your Mapbox public token
npm run dev
```

Open <http://localhost:3000>.

### 3. (Optional) Rebuild the model from scratch

```sh
python seattle_arcgis.py                # download raw SDOT layers -> data/raw/
python -m pipeline.build_intersections  # intersection points
python -m pipeline.snap_crashes         # crashes per intersection per year
python -m pipeline.assemble_features    # feature matrix
python -m pipeline.fit_risk_model       # NB regression -> data/model/nb_v2_arterial_aadt.pkl
python -m pipeline.score_risk           # EB scoring + treatments -> data/intermediate/intersection_scores.parquet
```

> Windows note: prefix each script with `python -X utf8` to avoid `cp1252` encoding errors on Unicode print statements.

---

## What you see in the app

- **Vision Zero scorecard** at the top of the page — total crashes, injuries, KSI, fatalities, and ped/bike involvement, recomputed live whenever you filter the map.
- **Interactive map** of all 651 intersections, colored by `risk_tier` and sized by `risk_score`. Hover for a tooltip; click for the full drill-in.
- **Layer toggles** for tier filtering and a bike-facilities overlay.
- **Drill-in panel** for any clicked intersection:
  - Score badge (`very_high` → `very_low`)
  - **Expected vs. actual** crash counts over the 6-year window
  - Severity breakdown (injury / KSI / fatal / ped / bike)
  - **Top contributors** — the three features pushing this intersection farthest from the modelled baseline, each labeled `+X%` or `−X%`
  - **Recommended treatments** — top-3 CMF-supported fixes, with predicted bike-KSI prevented per year

---

## How the model works

<details>
<summary><b>The dataset</b></summary>

Every numeric input comes from official Seattle GIS layers, downloaded by `seattle_arcgis.py` from the city's ArcGIS portal:

- **Intersections** — derived by clustering street-segment endpoints in `pipeline/build_intersections.py`
- **Crashes** — SDOT's collision dataset, snapped within 25 m of each intersection (`pipeline/snap_crashes.py`)
- **Severity** — five counts per intersection derived from SDOT's `MAXSEVERITYCODE`:
  - `injury_total` — any injury (code ≥ 2)
  - `ksi_total` — Killed or Seriously Injured (code ≥ 3, the Vision Zero target)
  - `fatal_total` — fatal only (code = 4)
  - `ped_total` / `bike_total` — crashes with `PEDCOUNT > 0` / `PEDCYLCOUNT > 0`
- **Features** — signalization, leg count, posted speed, bike-facility presence, arterial class, AADT (annual weekday daily traffic)

Caveat: SDOT's per-crash `PEDCOUNT` / `PEDCYLCOUNT` fields are sparse for records after 2018, so `snap_crashes.py` falls back to keyword matching on `SDOT_COLDESC` to recover the gap.

</details>

<details>
<summary><b>The statistical model</b></summary>

- **Family:** statsmodels `NegativeBinomial` (NB2), log link, `offset = log(years_observed)` over the 6-year window.
- **Scope:** **arterial intersections only** (`arterial_class >= 1` with positive AADT). Local-access streets are excluded per HSM Chapter 12 facility-type stratification — they're 63% zero-crash, contribute only 1 of 17 bike-KSI events, and would force one coefficient set to describe two physically different facility classes.
- **Formula:**
  ```
  total_crashes ~ is_signalized + num_legs + max_speed_limit
                + bike_facility + C(arterial_class) + log_aadt
  ```
- **Empirical Bayes adjustment** (HSM Part C): `w = 1 / (1 + α·μ); eb = w·μ + (1−w)·N` — pulls extreme model predictions toward observed counts at sites with enough data.
- **Headline metric served to the UI:** `expected_bike_ksi_per_year` with a 90% credible interval, derived by applying a Poisson-Gamma direct EB to bike-KSI counts using the all-crash NB prediction × citywide bike-KSI share as the prior.
- **Secondary fields:** `risk_score` (0–100 percentile rank) and `risk_tier` (`very_high` ≥ 90, `high` 70–89, `moderate` 40–69, `low` 20–39, `very_low` < 20), used for sorting and map color.

</details>

<details>
<summary><b>Why <code>log(AADT)</code> instead of raw <code>AADT</code></b></summary>

This is a functional-form choice, not a scaling trick. With a log-link GLM:

- **Raw AADT** ⇒ `μ ∝ exp(β · AADT)` — crashes grow *exponentially* with volume. A 50,000-AADT site would have astronomical predictions. Not physical.
- **log(AADT)** ⇒ `μ ∝ AADT^β` — crashes follow a *power law*. With β < 1 you recover the well-documented **sub-linear "safety in numbers" effect**: doubling volume multiplies crashes by `2^β`, not 2.

At our fitted `β = 0.26`, doubling AADT multiplies expected crashes by **2^0.26 ≈ 1.20** (a 20% increase, not 100%). Drivers slow down in denser traffic, cyclists adjust routes, pedestrian behavior shifts — per-vehicle risk drops as volume rises.

This is exactly how AASHTO HSM SPFs are written: `μ = exp(β₀) · AADT_major^β₁ · AADT_minor^β₂ · CMFs · years`. Taking log of both sides yields our linear predictor with `log(AADT)` as the term. Every state-DOT SPF in active use is the same shape.

</details>

<details>
<summary><b>Coefficient interpretation (fit <code>nb_v2_arterial_aadt</code>)</b></summary>

Each `β` is on the log-rate scale; `exp(β)` is the multiplicative effect on the expected crash count, all else equal.

| Term | β | exp(β) | Reading |
|---|---|---|---|
| `log_aadt` | +0.257 | 1.29 per e-fold | **+20% per AADT doubling** (sub-linear safety in numbers) |
| `is_signalized` | +1.159 | 3.19× | Signals sit at the busiest, highest-conflict junctions. Selection bias after AADT, not causation. |
| `num_legs` | +0.591 | 1.81 per leg | Each extra approach adds conflict points |
| `arterial_class = Minor` | +0.421 | 1.52× | +52% over principal arterial baseline |
| `arterial_class = Collector` | +0.246 | 1.28× | +28% over principal |
| `arterial_class = Other` | +1.022 | 2.78× | Heterogeneous catch-all; high but noisy |
| `max_speed_limit` | −0.108 | 0.90 per mph | Suspicious sign — posted speed has narrow range and is a poor proxy for operating speed |
| `bike_facility` | −0.306 | 0.74× | **Protective:** facility nearby → −26%. Blended across protected / painted / sharrow. |
| α (NB dispersion) | 0.640 | — | Meaningful overdispersion; NB is the correct family |

**Read with care:** these coefficients are *associative*, not causal. `is_signalized` is not "signalizing causes crashes" — signals correlate with high-conflict places. `bike_facility`'s −26% blends protected lanes (~−60% in CMFs) with sharrows (~−5%). For genuinely causal treatment effects, see the CMF section below.

</details>

<details>
<summary><b>Calibration and verification</b></summary>

- **Calibration:** sum predicted 1,772.7 vs. sum actual 1,720 (+3.1%; HSM threshold ±15%).
- **MAE per intersection (6-year window):** 3.47 crashes.
- **90% predictive coverage:**
  - all-crash: 95.1% (nominal 90%; slight over-coverage from discrete NB step quantiles, normal)
  - bike-KSI: 98.6% (conservative — Phase 1 borrows the all-crash α; Phase 2's bike-specific model will tighten this)
- **Spearman rank correlation (predicted vs. observed bike-KSI):** ρ = +0.30 at top-20 sites, ρ = +0.28 over all 346.

The rank correlation is modest because **17 bike-KSI events across 346 sites in 6 years is a fundamentally noisy sample**, not because the model is mis-specified. Lifting it requires more data, not a fancier model class — cyclist exposure (Strava Metro / bike counts), a denser bike-specific target, or expanded geographic scope.

</details>

---

## From "where is risk?" to "what should we build?" — the CMF layer

The model is **descriptive**: it tells you *where* crashes happen and *what correlates* with them, but its coefficients are observational, not causal. To answer *"what should we do?"*, the app uses the canonical HSM Part C two-stage method:

```
bike_KSI_prevented_per_year(site, treatment) =
    expected_bike_ksi_per_year(site) × (1 - CMF(treatment))
```

The model supplies the **site baseline**; published **Crash Modification Factors** supply the **causal treatment-effect multiplier**. CMFs come from before-and-after studies with comparison groups, peer-reviewed by FHWA and rated 1–5 stars in the [CMF Clearinghouse](https://www.cmfclearinghouse.org/).

[`pipeline/build_cmf_library.py`](pipeline/build_cmf_library.py) ingests a direct Clearinghouse CSV export, filters to **bike-involved, intersection-related, approved, non-rural studies**, and aggregates per treatment. It uses variance-weighted averaging only when every study reports a standard error; otherwise a simple mean with across-study variance. This avoids over-weighting whichever paper happened to report SEs.

### Current library (Clearinghouse export 2025-11-10)

| Treatment | Studies | CMF (90% CI) | Direction | Applies when |
|---|---|---|---|---|
| Install cycle track / protected bike lane | 4 | 0.43 (0.14–0.71) | helpful | `bike_facility = 0` |
| Raised bicycle crossing | 1 | 0.49 (0.30–0.68) | helpful | any |
| Offset cycle track w/ cyclist priority | 1 | 0.55 (0.28–0.82) | helpful | `bike_facility = 0` |
| Install painted bike lane | 4 | 0.56 (0.07–1.04) | helpful | `bike_facility = 0` |
| Prohibit right-turn-on-red | 8 | 0.58 (0.54–0.63) | helpful (prevention) | `is_signalized = 1` |
| Bike lane at signalized intersection | 8 | 1.08 (0.78–1.37) | anti-indication | `bike_facility = 0`, `is_signalized = 1` |
| Convert yield → signalized | 10 | 1.04 (0.64–1.45) | anti-indication | `is_signalized = 0` |
| Convert to single-lane roundabout | 23 | 1.40 (1.25–1.56) | anti-indication | any |

What the evidence actually says:

- **Cycle tracks and raised crossings are the clear winners** — strong, consistent reductions across multiple studies.
- **Prohibiting right-turn-on-red has 8 consistent studies** showing permitting it raises bike crashes ~77%. Prohibition is recommended with high confidence.
- **Roundabouts *increase* bike crashes by ~40%** across 23 studies — the well-documented bike-roundabout paradox. Safer for cars and pedestrians, more dangerous for cyclists due to entry/exit conflict geometry. They appear in the library as an explicit anti-indication so planners considering them see the bike-specific evidence.
- **Bike lanes at signalized intersections show no net effect** (CMF ~1.08). The likely explanation is that the facility raises cyclist exposure roughly as much as it reduces per-cyclist risk — exactly why bike-volume data is the next big data acquisition.

### How recommendations rank in the app

[`pipeline/treatments.py`](pipeline/treatments.py): for each intersection, filter to applicable treatments → compute `prevented = prediction × (1 − cmf)` → sort descending → keep top 3. Anti-indications naturally sort to the bottom (negative `prevented`); they're shown for informational purposes, not endorsement.

---

## What this model does *not* do (honest limitations)

- **Bike-KSI is a Phase-1 proxy.** The headline metric scales the all-crash NB by the citywide bike-KSI share and borrows the all-crash dispersion α. Honest as a placeholder; Phase 2 will fit a model directly on bike-KSI counts.
- **No cyclist exposure / bike-volume data.** AADT is in the model, but bike volume isn't. Per-cyclist risk is what Vision Zero actually cares about; without a cyclist denominator we're modelling *"where do bike-KSI events land"* rather than *"where is a cyclist most likely to be killed."* Strava Metro is the obvious unlock.
- **All-crash target, not bike-specific.** The NB target is `total_crashes`. Bike-KSI enters only through the scoring step's share multiplier. Phase 2: separate fits for `bike_total` and `ped_total`.
- **Capitol Hill scope only.** 346 modelled arterial intersections in one neighborhood. Expanding to all of Seattle is a refit + re-snap, not an app change, and would more than 10× the bike-KSI sample.
- **AADT is AWDT, not true AADT.** Seattle publishes Annual Weekday Daily Traffic — typically ~5–10% lower than true AADT. Standard local-practice substitute.
- **AADT coverage on class-5 arterials is 50%.** 19 of 38 class-5 ("other arterial") intersections in Capitol Hill are dropped from the fit for lack of usable AADT.
- **6-year observation window (2018–2023).** Re-fit when more recent SDOT collision data lands.
- **Coefficients are associative, not causal** — especially `is_signalized` (selection bias) and `max_speed_limit` (posted ≠ operating speed). For causal treatment effects, use the CMF section, not the coefficients.

A developer tool — [`pipeline/counterfactual.py`](pipeline/counterfactual.py) — lets you ask the model itself "what if I toggled this feature at this site?" Useful for understanding the model's internals, but **the CMF-based recommendations above are the right tool for stakeholder-facing 'what should we do' questions.**

---

## Repo layout

```
seattle_arcgis.py        ArcGIS REST fetcher for Seattle GIS datasets
api_server.py            FastAPI: /api/intersections, /api/bike-facilities
data/
  cmf_library.json                              Curated CMFs (generated)
  raw/cmf_clearinghouse_2025-11-10.csv          FHWA Clearinghouse export (9,777 CMFs)
  intermediate/                                 Pipeline outputs (intersections, features, scores)
  model/                                        Fitted .pkl models
pipeline/
  build_intersections.py    Street endpoints -> clustered intersection points
  snap_crashes.py           Crashes within 25 m -> per-intersection counts + severity
  assemble_features.py      Per-intersection feature matrix
  fit_risk_model.py         NB2 SPF fit + predictions (arterial-only, log(AADT))
  score_risk.py             EB shrinkage + bike-KSI proxy + 90% CIs + treatments
  contributors.py           Per-intersection top-contributors derivation
  treatments.py             Filter applicability and rank per-site treatments
  build_cmf_library.py      Re-runnable Clearinghouse CSV -> data/cmf_library.json
  counterfactual.py         Predict at hypothetical feature configurations
  evaluate_models.py        Calibration, coverage, rank-correlation diagnostics
  tests/                    pytest: EB math, contributors, treatments, calibration
frontend/
  app/                      Next.js app-router pages
  components/               Map, IntersectionReport, LeftPanel, MetricPrimitives
  lib/                      Typed API client + shared types
experiments/                One-off model sketches (e.g. hierarchical NB)
```

---

## Project background

This is a portfolio / civic-tech project. The aim is to demonstrate the **HSM-standard methodology** — Safety Performance Function plus CMF — on real Seattle data, end-to-end, with an honest accounting of what the model can and can't say.

Pull requests, issue reports, and suggestions are welcome.
