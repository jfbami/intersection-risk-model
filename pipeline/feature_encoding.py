"""Encoding of raw intersection features into model-ready predictors.

Centralises transforms that must stay byte-identical between model fitting
(`fit_risk_model`) and coefficient interpretation (`contributors`), so the
formula a coefficient was estimated under can never drift from the formula
used to read it back.

Leg count enters the model as a *top-coded categorical*, not a continuous
slope. A log-linear per-leg term forces a constant multiplicative effect and
extrapolates it without bound: fit on Capitol Hill — where 2-to-4-leg sites
are 97% of the data — the slope reads a 6-leg intersection as roughly +280%
over a 4-leg one, with a credible interval spanning +80% to +700% and no
six-leg site actually supporting it. Collapsing 5-or-more legs into a single
category lets the observed data, rather than an extrapolated line, set the
effect for rare high-leg geometries.
"""

from __future__ import annotations

REFERENCE_NUM_LEGS = 4
MAX_DISTINCT_LEGS = 5  # 5, 6, ... collapse into one "5+" category

LEG_CATEGORY_COLUMN = "legs_cat"
LEG_CATEGORY_TERM = (
    f"C({LEG_CATEGORY_COLUMN}, Treatment(reference={REFERENCE_NUM_LEGS}))"
)


def leg_category(num_legs: int) -> int:
    """Top-coded leg count used as the model's categorical leg predictor."""
    return min(int(num_legs), MAX_DISTINCT_LEGS)


def leg_category_param(num_legs: int) -> str:
    """Name of the fitted dummy coefficient for this site's leg category."""
    return f"{LEG_CATEGORY_TERM}[T.{leg_category(num_legs)}]"


def leg_label(num_legs: int) -> str:
    """Human-readable leg-count label that honours the top-coding."""
    legs = int(num_legs)
    if legs >= MAX_DISTINCT_LEGS:
        return f"{MAX_DISTINCT_LEGS}+ legs"
    return f"{legs}-leg intersection"
