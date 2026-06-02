"""Hierarchical Negative-Binomial SPF — exploratory sketch (NOT wired into the pipeline).

Purpose
-------
A concrete, runnable spec of the partial-pooling alternative to the production
NB model in `pipeline/fit_risk_model.py`. It demonstrates *adaptive* shrinkage:
thin feature categories are pulled hard toward the population mean, data-rich
ones barely move — the behaviour a single global ridge penalty cannot deliver.

What differs from production
----------------------------
1. `arterial_class` and `num_legs` are PARTIALLY POOLED (random effects) instead
   of treatment-coded fixed effects. Each level becomes a deviation from the
   global mean, shrunk toward 0 in proportion to how little data supports it.
2. `num_legs` is fed in RAW (levels 2..6, including the n=3 six-leg level).
   Partial pooling shrinks that sparse level automatically, so the manual
   "5+" top-coding from feature_encoding.py becomes unnecessary here — the
   model decides how much to trust it.
3. Well-supported predictors (signalized, bike_facility, speed, centrality)
   stay as fixed effects with weakly-informative priors (a Bayesian "ridge").

Modelling notes worth knowing before reading the code
-----------------------------------------------------
- Dispersion parameterization differs from statsmodels. statsmodels NB2 uses
  Var = mu + disp * mu**2 (fitted disp ~= 1.238 for bike). PyMC uses
  Var = mu + mu**2 / alpha, so PyMC's `alpha` is the INVERSE of statsmodels'
  dispersion. We report `dispersion = 1/alpha` so the two are comparable.
- With only 4-5 levels per factor, the group-level SD (sigma_*) is itself
  estimated from few numbers, so its HalfNormal prior is deliberately tight
  and does real work. This is the main caveat: full hierarchical payoff comes
  with a MANY-level grouping factor (geography, corridor) — see the README note.
- Exposure (years_observed) is a constant 6 across all sites, so the offset
  only shifts the intercept. It is included anyway so the spec stays correct
  if exposure ever varies (e.g. ped/vehicle modes or a longer window).

Run
---
    pip install "pymc>=5" arviz       # (and optionally bambi for the short form)
    python -m experiments.hierarchical_nb_sketch
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "intermediate" / "intersection_features.parquet"
CRASHES_PATH = ROOT / "data" / "intermediate" / "crashes_by_intersection.parquet"

TARGET = "bike_total"
GROUPING_FACTORS = ("arterial_class", "num_legs")
CONTINUOUS_PREDICTORS = ("max_speed_limit", "log_bike_centrality")
BINARY_PREDICTORS = ("is_signalized", "bike_facility")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_modelling_frame() -> pd.DataFrame:
    """Arterial intersections with usable exposure — mirrors fit_risk_model scope."""
    features = pd.read_parquet(FEATURES_PATH)
    crashes = pd.read_parquet(CRASHES_PATH)
    df = features.merge(crashes, on="intersection_id", how="inner")

    modellable = (
        (df["arterial_class"] >= 1)
        & df["max_aadt"].notna() & (df["max_aadt"] > 0)
        & df["bike_centrality"].notna()
    )
    df = df.loc[modellable].copy()
    df["log_bike_centrality"] = np.log(df["bike_centrality"])
    return df


def standardize(values: pd.Series) -> np.ndarray:
    """Zero-mean, unit-SD — keeps fixed-effect priors on a common scale."""
    return ((values - values.mean()) / values.std()).to_numpy()


def encode_levels(values: pd.Series) -> tuple[np.ndarray, list]:
    """Map a categorical column to contiguous 0-based indices plus its level list."""
    levels = sorted(values.unique().tolist())
    lookup = {level: i for i, level in enumerate(levels)}
    return values.map(lookup).to_numpy(), levels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(df: pd.DataFrame):
    """The hierarchical NB. Import pymc lazily so the module loads without it."""
    import pymc as pm

    arterial_idx, arterial_levels = encode_levels(df["arterial_class"])
    legs_idx, legs_levels = encode_levels(df["num_legs"])

    speed_z = standardize(df["max_speed_limit"])
    centrality_z = standardize(df["log_bike_centrality"])
    signalized = df["is_signalized"].to_numpy()
    bike_facility = df["bike_facility"].to_numpy()
    log_exposure = np.log(df["years_observed"].to_numpy())
    crashes = df[TARGET].to_numpy()

    coords = {"arterial_class": arterial_levels, "num_legs": legs_levels}

    with pm.Model(coords=coords) as model:
        # --- fixed effects: weakly-informative priors == a gentle "ridge" ---
        intercept = pm.Normal("intercept", mu=0.0, sigma=5.0)
        b_signalized = pm.Normal("b_signalized", mu=0.0, sigma=1.0)
        b_bike_facility = pm.Normal("b_bike_facility", mu=0.0, sigma=1.0)
        b_speed = pm.Normal("b_speed", mu=0.0, sigma=1.0)
        b_centrality = pm.Normal("b_centrality", mu=0.0, sigma=1.0)

        # --- partially-pooled categorical effects (the heart of the model) ---
        # sigma_* is LEARNED: it is how wide the spread of class/leg effects is.
        # Small per-level samples -> the level cannot pull sigma up on its own,
        # so its effect is dragged toward 0. That is the adaptive shrinkage.
        sigma_arterial = pm.HalfNormal("sigma_arterial", sigma=1.0)
        sigma_legs = pm.HalfNormal("sigma_legs", sigma=1.0)

        # Non-centered form: sample standard normals, then scale. Stabilises the
        # sampler when a group SD is near zero (the classic "funnel" pathology).
        z_arterial = pm.Normal("z_arterial", mu=0.0, sigma=1.0, dims="arterial_class")
        z_legs = pm.Normal("z_legs", mu=0.0, sigma=1.0, dims="num_legs")
        u_arterial = pm.Deterministic("u_arterial", z_arterial * sigma_arterial, dims="arterial_class")
        u_legs = pm.Deterministic("u_legs", z_legs * sigma_legs, dims="num_legs")

        # --- linear predictor (log link) ---
        log_mu = (
            intercept
            + b_signalized * signalized
            + b_bike_facility * bike_facility
            + b_speed * speed_z
            + b_centrality * centrality_z
            + u_arterial[arterial_idx]
            + u_legs[legs_idx]
            + log_exposure
        )

        # --- NB2 likelihood. PyMC alpha = 1 / (statsmodels dispersion). ---
        nb_alpha = pm.Exponential("nb_alpha", lam=1.0)
        pm.Deterministic("dispersion", 1.0 / nb_alpha)  # comparable to nb_v3 alpha
        pm.NegativeBinomial("crashes", mu=pm.math.exp(log_mu), alpha=nb_alpha, observed=crashes)

    return model


# ---------------------------------------------------------------------------
# Concise equivalent (bambi) — same model, formula form, for reference
# ---------------------------------------------------------------------------

def build_model_bambi(df: pd.DataFrame):
    """The (1|group) syntax IS partial pooling. Offset omitted: exposure is constant."""
    import bambi as bmb

    formula = (
        f"{TARGET} ~ is_signalized + bike_facility"
        " + scale(max_speed_limit) + scale(log_bike_centrality)"
        " + (1|arterial_class) + (1|num_legs)"
    )
    return bmb.Model(formula, data=df, family="negativebinomial")


# ---------------------------------------------------------------------------
# Run + read
# ---------------------------------------------------------------------------

def _print_multiplier_effects(idata, df: pd.DataFrame, varname: str, factor: str) -> None:
    """Group effects as risk multipliers exp(u)-1, annotated with each level's n."""
    counts = df[factor].value_counts().to_dict()
    posterior = np.exp(idata.posterior[varname]) - 1.0
    stacked = posterior.stack(sample=("chain", "draw")).transpose(factor, "sample")
    print(f"\n{factor}  (effect vs. the average intersection):")
    for level, draws in zip(stacked[factor].values, stacked.values):
        pct = draws * 100
        low, high = np.percentile(pct, [5, 95])
        print(f"  {factor}={level!s:<3} n={counts.get(level, 0):>3}   "
              f"{pct.mean():+5.0f}%   [90% {low:+.0f}%, {high:+.0f}%]")


def main() -> None:
    import arviz as az
    import pymc as pm

    df = load_modelling_frame()
    print(f"Modelling {len(df)} intersections, {int(df[TARGET].sum())} bike crashes.\n")

    model = build_model(df)
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, target_accept=0.95,
            chains=4, random_seed=0,
            progressbar=False,  # avoids pymc's rich/matplotlib progress dependency
        )

    divergences = int(idata.sample_stats["diverging"].sum())
    rhat = az.rhat(idata)
    max_rhat = float(max(float(rhat[name].max()) for name in rhat.data_vars))
    print(f"\nDivergences: {divergences}   (0 is good)")
    print(f"Max R-hat:   {max_rhat:.3f}   (~1.00 is good)\n")
    print(az.summary(
        idata,
        var_names=["sigma_arterial", "sigma_legs", "dispersion",
                   "b_signalized", "b_bike_facility"],
        ci_prob=0.9,  # arviz 1.x renamed hdi_prob -> ci_prob
    ))

    # Watch: arterial_class 5 (n=19) and num_legs 6 (n=3) shrink toward 0% with
    # WIDE intervals; arterial_class 2 (n=185) holds its effect with a tight one.
    _print_multiplier_effects(idata, df, "u_arterial", "arterial_class")
    _print_multiplier_effects(idata, df, "u_legs", "num_legs")


if __name__ == "__main__":
    main()
