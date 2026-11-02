"""Step 8: README figures. Reads the CSVs from steps 2-6 and recomputes nothing."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm.context import FIGURES, load
from alm.projection import Assumptions

PRICING_SPREAD = Assumptions().pricing_spread

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 9, "savefig.dpi": 160,
    "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def _line(ax, x, y, color, label, **kw):
    ax.plot(x, y, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", ms=6, mec=SURFACE, mew=1.5, label=label, **kw)


def hero() -> None:
    prof = load("02_value_vs_shift.csv").dropna(subset=["dynamic_duration"])
    summ = load("02_duration_convexity.csv").set_index("variant")
    static = summ.loc["dynamic", "static_duration"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _clean(ax)
    x = prof["shift_bp"]
    ax.axhline(static, color=MUTED, lw=1.2)
    ax.text(x.min(), static + 0.12, f"static cash flow duration, {static:.1f} years "
            "(cash flows held fixed)", color=INK2, fontsize=9, va="bottom")
    _line(ax, x, prof["no_dynamic_duration"], ORANGE, "credited rate resets, no dynamic surrender")
    _line(ax, x, prof["dynamic_duration"], BLUE, "credited rate resets + dynamic surrender")
    p = prof.set_index("shift_bp")
    for s in (-200, 0, 200):
        hi = "dynamic_duration" if p.loc[s, "dynamic_duration"] >= p.loc[s, "no_dynamic_duration"] \
            else "no_dynamic_duration"
        for col in ("dynamic_duration", "no_dynamic_duration"):
            ax.annotate(f"{p.loc[s, col]:.1f}", (s, p.loc[s, col]),
                        xytext=(0, 9 if col == hi else -16), textcoords="offset points",
                        ha="center", color=INK2, fontsize=9)
    ax.set_xlabel("instantaneous parallel shift of today's Treasury curve (bp)")
    ax.set_ylabel("liability effective duration (years)")
    ax.set_ylim(0, static + 1.0)
    ax.set_title(f"The block's duration falls as rates rise: {p.loc[-200, 'dynamic_duration']:.1f} "
                 f"years at -200 bp, {p.loc[200, 'dynamic_duration']:.1f} at +200 bp")
    ax.legend(loc="lower left")
    fig.savefig(FIGURES / "00_hero_duration_vs_rates.png")
    plt.close(fig)


def surplus() -> None:
    prof = load("04_surplus_vs_shift.csv")
    dist = load("04_surplus_by_path.csv")
    series = (("static_match", BLUE, "static duration match"),
              ("static_match_plus_payers", ORANGE, "static match + payer swaptions"),
              ("effective_match", AQUA, "effective duration match"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.22})
    for ax in (a1, a2):
        _clean(ax)
    for key, color, label in series:
        _line(a1, prof["shift_bp"], prof[f"{key}_surplus"] / 1e6, color, label)
    a1.axhline(0, color=AXIS, lw=0.8)
    a1.set_xlabel("instantaneous parallel shift (bp)")
    a1.set_ylabel("economic surplus, assets less liability ($M)")
    a1.set_title("Surplus under an instantaneous shock")
    a1.legend(loc="lower left")

    x = dist["avg_factor_0_10y"] * 1e4
    for key, color, label in series:
        a2.scatter(x, dist[f"{key}__dynamic"] / 1e6, s=10, color=color, alpha=0.55,
                   linewidths=0, label=label)
    a2.axvline(0, color=AXIS, lw=0.8)
    a2.set_xlabel("average rate move over the first 10 years on the path (bp)")
    a2.set_ylabel("run-off surplus on the path, present value ($M)")
    a2.set_title("Surplus on each of 1,000 paths")
    leg = a2.legend(loc="upper right", markerscale=2.2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.savefig(FIGURES / "01_surplus_shock_and_paths.png")
    plt.close(fig)


def krd() -> None:
    k = load("03_key_rate_durations.csv")
    k = k[k["key_rate"] != "sum"].copy()
    k["key_rate"] = k["key_rate"].astype(float)
    labels = [f"{v:g}y" for v in k["key_rate"]]
    xs = np.arange(len(k))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    _clean(ax)
    for i, (col, color, label) in enumerate((
            ("liability_dv01", BLUE, "liability"),
            ("static_match_dv01", ORANGE, "static match assets"),
            ("effective_match_dv01", AQUA, "effective match assets"))):
        ax.bar(xs + (i - 1) * w, k[col] / 1e3, width=w, color=color, label=label,
               edgecolor=SURFACE, linewidth=1.5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("key rate")
    ax.set_ylabel("dollar duration ($ thousand per bp)")
    ax.set_title("Key rate exposure: the liability sits at 3-7 years, the barbell at 2 and 20")
    ax.legend(loc="upper left")
    fig.savefig(FIGURES / "02_key_rate_durations.png")
    plt.close(fig)


def replay() -> None:
    d = load("06_replay_path.csv")
    st = load("06_stress.csv").set_index("scenario")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.22})
    for ax in (a1, a2):
        _clean(ax)
    horizon = 10
    spread = PRICING_SPREAD
    for name, color, label in (("base", MUTED, "base path (forward curve)"),
                               ("2022 replay", BLUE, "2022-2023 replay")):
        s = d[(d["scenario"] == name) & (d["t"] < horizon)]
        gap = (s["new_money"] - spread - s["credited"]) * 1e4
        a1.plot(s["t"], gap, color=color, lw=2, label=label, drawstyle="steps-post")
        full = d[d["scenario"] == name]
        yearly = [full.loc[(full["t"] > y - 1 + 1e-9) & (full["t"] <= y + 1e-9), "surrenders"].sum()
                  for y in range(1, horizon + 1)]
        _line(a2, np.arange(1, horizon + 1), np.array(yearly) / 1e6, color, label)
    a1.axhline(25, color=AXIS, lw=0.8)
    a1.text(horizon, 27, "25 bp buffer", ha="right", va="bottom", color=INK2, fontsize=9)
    a1.set_xlabel("years from valuation")
    a1.set_ylabel("new contract rate less the block's credited rate (bp)")
    a1.set_title("The gap policyholders respond to")
    a1.legend(loc="upper right")
    a2.set_xlabel("projection year")
    a2.set_ylabel("surrenders in the year ($M)")
    imp = st.loc["2022 replay"]
    a2.set_title(f"Surrenders: +{imp['surrenders_yr1_3_vs_base']:.0%} over the first three years")
    a2.legend(loc="upper right")
    fig.savefig(FIGURES / "03_replay_2022.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    hero()
    surplus()
    krd()
    replay()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
