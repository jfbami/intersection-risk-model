"""Encoding of raw intersection features into model-ready predictors.

Centralises transforms that must stay byte-identical between model fitting
(`fit_risk_model`) and coefficient interpretation (`contributors`), so the
formula a coefficient was estimated under can never drift from the formula
used to read it back.

Leg count enters the model as a *top-coded categorical*, not a continuous
slope. A log-linear per-leg term forces a constant multiplicative effect and
extrapolates it without bound. Fit on Capitol Hill, where 2-to-4-leg sites
make up 96% of the data (333/346), the slope reads a 6-leg intersection as
+283% over a 4-leg one, with a 95% confidence interval spanning +83% to
+701%. Collapsing 5-or-more legs into a single category lets the observed
data, rather than an extrapolated line, set the effect for rare high-leg
geometries.

Measured support (experiments/results/, experiment E5). Nested LR tests
against a full per-level categorical reject the continuous slope in every
mode (p = 0.0088 bike, 9.0e-08 ped, 1.5e-07 vehicle) and never reject this
top-coding (p = 0.63, 0.70, 0.81).

Note the extrapolation is not merely uncertain, it points the wrong way.
Three six-leg sites do exist (4 bike, 5 ped, 32 vehicle crashes), and fit
unconstrained they show exp(beta) ~ 0.79 for bike and 0.83 for ped, i.e.
*fewer* crashes than a 4-leg site, the opposite sign to the extrapolated
+283%. The
real signal is that 3-leg sites are much safer than 4-leg ones (exp(beta)
0.21 / 0.18 / 0.28, all p < 1e-05); forcing a straight line through that
drop is what manufactures the +283% figure.
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
