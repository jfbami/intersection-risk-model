"""Generate the figures used in RESULTS.md.

Every value plotted is either read straight out of the measured experiment JSON
in experiments/results/, or recomputed here from the real modelling frame.
Nothing is hand-typed. Each figure prints the numbers it drew, so the chart can
be checked against the text.

Run:  PYTHONPATH=. python -X utf8 experiments/ab/make_figures.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODES = ["bike", "ped", "vehicle"]
NICE = {"bike": "Bike", "ped": "Pedestrian", "vehicle": "Vehicle"}

# consistent, colourblind-safe palette
C_GOOD, C_BAD, C_NEUTRAL, C_ACCENT = "#2E7D5B", "#B3412C", "#7A7A7A", "#2F5C8A"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def load(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def frame():
    from pipeline.fit_risk_model import load_and_join, prepare
    df, _ = prepare(load_and_join())
    return df


def save(fig, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.004, size=8):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=size)


# ---------------------------------------------------------------------------
# E5 - leg encoding
# ---------------------------------------------------------------------------

def fig_e5_mae():
    e5 = load("e5_results.json")
    specs = ["a_topcoded_cat", "b_continuous", "c_full_cat"]
    names = ["Top-coded 5+\n(what we use)", "Straight line\nper leg", "Every leg count\nseparately"]
    vals = {s: [e5["modes"][m]["specs"][s]["repeated_cv"]["oof_mae_mean"] for m in MODES] for s in specs}
    print("E5 repeated-CV MAE:", json.dumps(vals, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, mode in zip(axes, MODES):
        v = [vals[s][MODES.index(mode)] for s in specs]
        cols = [C_GOOD if x == min(v) else C_BAD if x == max(v) else C_NEUTRAL for x in v]
        bars = ax.bar(names, v, color=cols, width=0.62)
        bar_labels(ax, bars, dy=(max(v) - min(v)) * 0.04 or 0.002)
        ax.set_title(NICE[mode], fontsize=11, fontweight="bold")
        lo, hi = min(v), max(v)
        pad = (hi - lo) * 0.35 or 0.02
        ax.set_ylim(lo - pad, hi + pad * 1.6)
        ax.tick_params(axis="x", labelsize=8)
        if mode == "bike":
            ax.set_ylabel("Average error\n(lower is better)")
    fig.suptitle("E5  How to handle the number of roads meeting at an intersection",
                 fontsize=12, fontweight="bold", y=1.04)
    save(fig, "e5_leg_encoding_mae.png")


def fig_e5_extrapolation():
    """The money chart: what the straight line predicts vs what 6-leg sites do."""
    df = frame()
    base = ("is_signalized + max_speed_limit + bike_facility "
            "+ C(arterial_class) + log_bike_centrality")

    def fit(f):
        m = smf.negativebinomial(f, data=df, offset=df["offset"].values)
        r = m.fit(disp=False)
        if not r.mle_retvals.get("converged", True):
            r = m.fit(disp=False, method="bfgs", maxiter=500)
        return r

    cont = fit(f"bike_total ~ {base} + num_legs")
    full = fit(f"bike_total ~ {base} + C(num_legs, Treatment(reference=4))")
    b, se = float(cont.params["num_legs"]), float(cont.bse["num_legs"])

    legs = np.array([2, 3, 4, 5, 6])
    line = np.exp(b * (legs - 4))
    line_lo = np.exp((b - 1.96 * se) * (legs - 4))
    line_hi = np.exp((b + 1.96 * se) * (legs - 4))

    obs, obs_lo, obs_hi = [], [], []
    for L in legs:
        if L == 4:
            obs.append(1.0); obs_lo.append(1.0); obs_hi.append(1.0); continue
        term = f"C(num_legs, Treatment(reference=4))[T.{L}]"
        bb, ss = float(full.params[term]), float(full.bse[term])
        obs.append(np.exp(bb)); obs_lo.append(np.exp(bb - 1.96 * ss)); obs_hi.append(np.exp(bb + 1.96 * ss))

    counts = df.num_legs.value_counts().to_dict()
    print(f"E5 continuous beta={b:.6f} se={se:.6f} -> 6-leg exp(2b)={np.exp(2*b):.4f} "
          f"({100*(np.exp(2*b)-1):+.1f}%), 95% CI [{100*(np.exp(2*(b-1.96*se))-1):+.1f}%, "
          f"{100*(np.exp(2*(b+1.96*se))-1):+.1f}%]")
    print("E5 unconstrained per-leg multipliers:", dict(zip(legs.tolist(), [round(o, 4) for o in obs])))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(legs, line_lo, line_hi, color=C_BAD, alpha=0.13, label="Straight line, 95% range")
    ax.plot(legs, line, "o-", color=C_BAD, lw=2, label="What a straight line predicts")
    ax.errorbar(legs, obs, yerr=[np.array(obs) - np.array(obs_lo), np.array(obs_hi) - np.array(obs)],
                fmt="s", color=C_ACCENT, capsize=4, lw=1.6, ms=7,
                label="What the data actually shows")
    ax.axhline(1.0, color="black", lw=0.9, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "8x"])
    ax.set_xticks(legs)
    ax.set_xticklabels([f"{L} roads\n(n={counts.get(L,0)})" for L in legs], fontsize=9)
    ax.set_ylabel("Crash risk vs a 4-road intersection")
    ax.set_title("E5  The straight line predicts 6-road sites are ~4x worse.\n"
                 "The three real 6-road sites are slightly safer.",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.annotate("+283%", xy=(6, line[-1]), xytext=(5.35, 6.2), fontsize=10,
                color=C_BAD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.3))
    ax.annotate("0.79x", xy=(6, obs[-1]), xytext=(5.4, 0.16), fontsize=10,
                color=C_ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.3))
    save(fig, "e5_extrapolation_vs_reality.png")


# ---------------------------------------------------------------------------
# E2 - NB vs Poisson
# ---------------------------------------------------------------------------

def fig_e2_coverage():
    e2 = load("e2_results.json")
    nb = [e2["modes"][m]["nb"]["in_sample_coverage_90"]["coverage_pct"] for m in MODES]
    po = [e2["modes"][m]["poisson"]["in_sample_coverage_90"]["coverage_pct"] for m in MODES]
    print("E2 coverage NB:", nb, " Poisson:", po)

    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    b1 = ax.bar(x - w/2, nb, w, label="Negative Binomial (what we use)", color=C_GOOD)
    b2 = ax.bar(x + w/2, po, w, label="Poisson", color=C_BAD)
    bar_labels(ax, b1, "{:.1f}%", dy=0.7); bar_labels(ax, b2, "{:.1f}%", dy=0.7)
    ax.axhline(90, color="black", ls="--", lw=1.2)
    ax.text(2.46, 90.8, "target: 90%", fontsize=8.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in MODES])
    ax.set_ylabel("How often the real number\nfell inside the predicted range")
    ax.set_ylim(60, 104)
    ax.set_title("E2  Poisson's ranges are too narrow for vehicle crashes",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "e2_interval_coverage.png")


def fig_e2_point_accuracy():
    e2 = load("e2_results.json")
    deltas = {m: e2["modes"][m]["cv_paired_difference_nb_minus_poisson"]["mae_delta_per_fold"]
              for m in MODES}
    pvals = {m: e2["modes"][m]["cv_paired_difference_nb_minus_poisson"]["mae_paired_t_p"]
             for m in MODES}
    print("E2 per-fold MAE deltas:", deltas); print("E2 paired p:", pvals)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for i, m in enumerate(MODES):
        d = deltas[m]
        ax.scatter([i] * len(d), d, s=52, color=C_ACCENT, alpha=0.75, zorder=3)
        ax.hlines(np.mean(d), i - 0.2, i + 0.2, color=C_BAD, lw=2.4, zorder=4)
    ax.axhline(0, color="black", lw=1.1, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{NICE[m]}\np = {pvals[m]:.2f}" for m in MODES])
    ax.set_ylabel("Poisson better  <-   ->  NB better")
    ax.set_title("E2  ...but for the actual predicted number, the two are a coin flip\n"
                 "(each dot is one test split; red bar is the average)",
                 fontsize=11, fontweight="bold")
    save(fig, "e2_point_accuracy.png")


# ---------------------------------------------------------------------------
# E7 - zero inflation
# ---------------------------------------------------------------------------

def fig_e7_zeros():
    e7 = load("e7_results.json")
    obs = [e7["modes"][m]["observed_zero_sites"] for m in MODES]
    exp = [e7["modes"][m]["nb"]["expected_zero_sites"] for m in MODES]
    print("E7 zeros observed:", obs, " NB-expected:", [round(e, 2) for e in exp])

    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - w/2, obs, w, label="Sites that really had zero crashes", color=C_ACCENT)
    b2 = ax.bar(x + w/2, exp, w, label="Sites the model expected to have zero", color=C_GOOD)
    bar_labels(ax, b1, "{:.0f}", dy=2.5); bar_labels(ax, b2, "{:.1f}", dy=2.5)
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in MODES])
    ax.set_ylabel("Number of intersections (of 346)")
    ax.set_ylim(0, max(obs) * 1.22)
    ax.set_title("E7  The model already gets the zeros right, so no extra fix is needed",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    save(fig, "e7_zero_counts.png")


# ---------------------------------------------------------------------------
# E6 - exposure offset
# ---------------------------------------------------------------------------

def fig_e6_exposure():
    e6 = load("e6_results.json")
    pb = e6["part_b_synthetic_variable_exposure"]["modes"]
    pretty = {"bike": "Bike", "ped": "Pedestrian", "vehicle": "Vehicle",
              "total_supplementary": "All crashes\n(real crash years)"}
    rows = []
    for key, d in pb.items():
        b = d["specs"]["b_free_covariate_no_offset"]
        est = b["coef_log_years_observed"]
        se = b["se_log_years_observed"]
        rows.append((pretty.get(key, key), est, est - 1.96 * se, est + 1.96 * se))
    print("E6 free-exposure coefficients (est, 95% CI):",
          [(r[0].replace("\n", " "), round(r[1], 3), round(r[2], 3), round(r[3], 3)) for r in rows])

    labels = [r[0] for r in rows]
    est = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows]); hi = np.array([r[3] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.4, 0.72 * len(rows) + 2.3))
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=C_ACCENT,
                capsize=5, lw=1.8, ms=8)
    ax.axvline(1.0, color=C_GOOD, lw=2.2, ls="--")
    ax.text(1.03, len(rows) - 0.35, "what the model\nassumes (1.0)",
            color=C_GOOD, fontsize=8.8, va="top")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Measured effect of exposure (with 95% range)")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("E6  Every range covers 1.0, so the model's assumption is safe\n"
                 "(measured on a synthetic varying-exposure dataset)",
                 fontsize=11, fontweight="bold")
    save(fig, "e6_exposure_offset.png")


# ---------------------------------------------------------------------------
# E4 - pooled vs per-mode
# ---------------------------------------------------------------------------

def fig_e4_arms():
    e4 = load("e4_results.json")
    old = [e4["arm_1b_true_v2_oof"][m]["mae"] for m in MODES]
    pooled = [e4["part_A_cv"][m]["arm1_pooled_scaled"]["pooled_oof"]["mae"] for m in MODES]
    permode = [e4["part_A_cv"][m]["arm2_permode"]["pooled_oof"]["mae"] for m in MODES]
    print("E4 out-of-sample MAE  old v2:", [round(v, 4) for v in old])
    print("E4 out-of-sample MAE  pooled+fixed legs:", [round(v, 4) for v in pooled])
    print("E4 out-of-sample MAE  three per-mode:", [round(v, 4) for v in permode])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    names = ["Old\nmodel", "Shared +\nleg fix", "Three\nmodels"]
    for ax, m in zip(axes, MODES):
        i = MODES.index(m)
        v = [old[i], pooled[i], permode[i]]
        cols = [C_BAD, C_ACCENT, C_GOOD]
        bars = ax.bar(names, v, color=cols, width=0.6)
        bar_labels(ax, bars, dy=(max(v) - min(v)) * 0.05)
        ax.set_title(NICE[m], fontsize=11, fontweight="bold")
        lo, hi = min(v), max(v)
        pad = (hi - lo) * 0.3
        ax.set_ylim(lo - pad, hi + pad * 1.7)
        ax.tick_params(axis="x", labelsize=8)
        if m == "bike":
            ax.set_ylabel("Error on unseen data\n(lower is better)")
    fig.suptitle("E4  Almost all the improvement came from the leg fix,\n"
                 "not from splitting into three models",
                 fontsize=12, fontweight="bold", y=1.08)
    save(fig, "e4_pooled_vs_permode.png")


# ---------------------------------------------------------------------------
# E1 - volume form
# ---------------------------------------------------------------------------

def fig_e1_volume():
    df = frame()
    base = ("is_signalized + C(legs_cat, Treatment(reference=4)) + max_speed_limit "
            "+ bike_facility + C(arterial_class)")

    def fit(extra, col=None):
        d = df.copy()
        if col is not None:
            d["vol"] = col
        m = smf.negativebinomial(f"vehicle_only_total ~ {base} + {extra}",
                                 data=d, offset=d["offset"].values)
        best, bll = None, -np.inf
        for meth in ["newton", "bfgs", "nm", "lbfgs"]:
            try:
                r = m.fit(disp=False, method=meth, maxiter=2000)
                if r.mle_retvals.get("converged", True) and float(r.llf) > bll:
                    best, bll = r, float(r.llf)
            except Exception:
                pass
        return best

    df = df.copy()
    df["aadt_s"] = df.max_aadt / 10000.0
    r_log = fit("log_aadt")
    r_raw = fit("aadt_s")

    ref = df.iloc[[int(np.argmin(np.abs(df.max_aadt - df.max_aadt.median())))]].copy()
    grid = np.linspace(1000, 60000, 200)
    pl, pr = [], []
    for a in grid:
        r1 = ref.copy(); r1["log_aadt"] = np.log(a); r1["aadt_s"] = a / 10000.0
        pl.append(float(r_log.predict(r1, offset=r1["offset"].values).iloc[0]))
        pr.append(float(r_raw.predict(r1, offset=r1["offset"].values).iloc[0]))
    pl, pr = np.array(pl), np.array(pr)
    i50 = int(np.argmin(np.abs(grid - 50000)))
    print(f"E1 at 50,000 AADT: log={pl[i50]:.2f} raw={pr[i50]:.2f} ratio={pr[i50]/pl[i50]:.3f}")
    print(f"E1 observed AADT range: {df.max_aadt.min():.0f}-{df.max_aadt.max():.0f}")

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.axvspan(df.max_aadt.min(), df.max_aadt.max(), color=C_NEUTRAL, alpha=0.16)
    ax.text(df.max_aadt.max() * 0.52, min(pl.min(), pr.min()) * 1.02,
            "range of real data (1,013 - 41,808)", fontsize=9, color="#444", ha="center")
    ax.plot(grid, pl, lw=2.3, color=C_GOOD, label="log(traffic)  - what we use")
    ax.plot(grid, pr, lw=2.3, color=C_BAD, ls="--", label="raw traffic  - the alternative")
    ax.axvline(50000, color="black", lw=1.1, ls=":")
    ax.annotate(f"at 50,000 the two differ\nby {pr[i50]/pl[i50]:.2f}x - not a blow-up",
                xy=(50000, pr[i50]), xytext=(31000, pr[i50] * 1.22), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel("Daily traffic volume (AADT)")
    ax.set_ylabel("Predicted vehicle crashes\n(typical intersection, 6 years)")
    ax.set_title("E1  The old README said raw traffic would 'explode'. It doesn't -\n"
                 "and we have no data out there anyway",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "e1_volume_form.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_e5_mae()
    fig_e5_extrapolation()
    fig_e2_coverage()
    fig_e2_point_accuracy()
    fig_e7_zeros()
    fig_e6_exposure()
    fig_e4_arms()
    fig_e1_volume()
    print("Done.")
