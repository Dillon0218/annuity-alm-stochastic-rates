"""Step 2: the block, its value, and how its rate sensitivity moves with rates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import risk
from alm.context import SHIFT, build, save

SHIFTS_BP = np.arange(-250, 251, 50)


def block_summary(ctx) -> pd.DataFrame:
    inf = ctx.inforce
    by = (inf.groupby("issue_year")
          .agg(policies=("policy_id", "size"), account_value=("av", "sum"),
               mean_age=("age", "mean"), gmir=("gmir", "first"))
          .reset_index())
    by["policy_year_now"] = 2026 - by["issue_year"] + 1
    by["surrender_charge_now"] = [
        (0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)[py - 1] if py <= 7 else 0.0
        for py in by["policy_year_now"]]
    total = pd.DataFrame([{
        "issue_year": "all", "policies": len(inf), "account_value": inf["av"].sum(),
        "mean_age": inf["age"].mean(), "gmir": np.average(inf["gmir"], weights=inf["av"]),
        "policy_year_now": np.nan, "surrender_charge_now": np.nan}])
    out = pd.concat([by, total], ignore_index=True)
    out["female_share"] = np.nan
    out.loc[out.index[-1], "female_share"] = inf["female"].mean()
    out.loc[out.index[-1], "model_points"] = ctx.mp.n_cells
    return out


def main() -> None:
    ctx = build()
    av0 = ctx.av0
    save(block_summary(ctx), "02_block_summary.csv")

    variants = {"dynamic": ctx.asm, "no_dynamic": ctx.asm.but(dynamic="none")}

    # value against an instantaneous parallel shift, same draws throughout
    prof = {"shift_bp": SHIFTS_BP}
    for name, asm in variants.items():
        prof[name] = [ctx.liability(ctx.curve.bumped(s / 1e4), asm)[2].value for s in SHIFTS_BP]
    prof = pd.DataFrame(prof)
    for name in variants:            # local effective duration at each shift
        v = prof[name].to_numpy()
        d = np.full(len(v), np.nan)
        d[1:-1] = (v[:-2] - v[2:]) / (2 * v[1:-1] * SHIFT)
        prof[f"{name}_duration"] = d
    save(prof, "02_value_vs_shift.csv")

    rows = []
    for name, asm in variants.items():
        v = prof.set_index("shift_bp")[name]
        dur, cvx = risk.duration_convexity(v[-50], v[0], v[50], SHIFT)
        det = {s: ctx.deterministic_liability(ctx.curve.bumped(s / 1e4), asm).value
               for s in (-50, 0, 50)}
        det_dur, det_cvx = risk.duration_convexity(det[-50], det[0], det[50], SHIFT)
        r = ctx.deterministic_liability(asm=asm)
        pv = r.cf[0] * ctx.curve.df(ctx.times)
        rows.append({
            "variant": name, "stochastic_value": v[0], "stochastic_pct_av": v[0] / av0,
            "deterministic_value": det[0], "deterministic_pct_av": det[0] / av0,
            "static_duration": (ctx.times * pv).sum() / pv.sum(),
            "deterministic_effective_duration": det_dur,
            "effective_duration": dur, "effective_convexity": cvx,
            "duration_at_minus100": prof.set_index("shift_bp").loc[-100, f"{name}_duration"],
            "duration_at_plus100": prof.set_index("shift_bp").loc[100, f"{name}_duration"],
        })
    summ = pd.DataFrame(rows)
    save(summ, "02_duration_convexity.csv")
    print(summ.round(4).to_string(index=False))

    # expected cash flow profile, base assumptions
    _, _, r = ctx.liability(keep_detail=True)
    det = ctx.deterministic_liability()
    years = np.arange(1, 31)
    q = 4
    def by_year(a):
        return np.array([a[:, q * (y - 1) + 1: q * y + 1].sum(axis=1).mean() for y in years])
    d = r.detail
    out = pd.DataFrame({
        "year": years,
        "surrenders_av": by_year(d["surrenders"]),
        "surrender_charges": by_year(d["surrender_charges"]),
        "deaths": by_year(d["deaths"]),
        "total_benefits": by_year(r.cf),
        "credited_rate": [d["credited"][:, q * (y - 1)].mean() for y in years],
        "rate_gap": [d["rate_gap"][:, q * (y - 1)].mean() for y in years],
        "deterministic_benefits": [det.cf[0, q * (y - 1) + 1: q * y + 1].sum() for y in years],
    })
    save(out, "02_cash_flows.csv")
    dyn, nod = summ.iloc[0], summ.iloc[1]
    print(f"\ndynamic surrender adds {dyn.stochastic_value - nod.stochastic_value:,.0f} "
          f"({(dyn.stochastic_value - nod.stochastic_value) / av0:.2%} of AV)")


if __name__ == "__main__":
    main()
