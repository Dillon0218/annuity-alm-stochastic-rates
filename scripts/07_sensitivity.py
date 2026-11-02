"""Step 7: how the answer moves with the three assumptions that drive it.

Dynamic surrender strength, crediting strategy and Hull-White volatility. The assets
are held fixed at the base static match, so each row is the same balance sheet under a
different view of the liability or of rates.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import assets as A
from alm import risk
from alm.context import SHIFT, build, save
from alm.hull_white import b_factor

_spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).with_name("03_assets_and_krd.py"))
step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step3)


def refit_sigma(vol_table: pd.DataFrame, a: float) -> float:
    """Least squares sigma for a fixed mean reversion, same tenors as the base fit."""
    use = vol_table[vol_table["in_fit"]]
    g = b_factor(a, use["tenor"].to_numpy()) / use["tenor"].to_numpy()
    return float((g * use["realized_vol"]).sum() / (g * g).sum())


def main() -> None:
    ctx = build()
    static = step3.portfolios(ctx)["static_match"][0]
    base, s0, a0 = ctx.asm, ctx.hw.sigma, ctx.hw.a
    a_alt = 0.05
    variants = [
        ("dynamic surrender", "none", base.but(dynamic="none"), {}),
        ("dynamic surrender", "Academy x0.5", base.but(strength=0.5), {}),
        ("dynamic surrender", "Academy (base)", base, {}),
        ("dynamic surrender", "Academy x2", base.but(strength=2.0), {}),
        ("dynamic surrender", "PHBDG power form", base.but(dynamic="phbdg"), {}),
        ("crediting", "book yield, 1/7 turnover (base)", base, {}),
        ("crediting", "book yield, 1/3 turnover", base.but(turnover=1 / 3), {}),
        ("crediting", "new money rate", base.but(turnover=np.inf), {}),
        ("crediting", "spread 125 bp", base.but(pricing_spread=0.0125), {}),
        ("crediting", "spread 225 bp", base.but(pricing_spread=0.0225), {}),
        ("Hull-White", "sigma x0.5", base, {"sigma": 0.5 * s0}),
        ("Hull-White", "sigma x0.75", base, {"sigma": 0.75 * s0}),
        ("Hull-White", f"base (a={a0:.4f}, sigma={s0:.4%})", base, {}),
        ("Hull-White", "sigma x1.25", base, {"sigma": 1.25 * s0}),
        ("Hull-White", "sigma x1.5", base, {"sigma": 1.5 * s0}),
        ("Hull-White", f"a = {a_alt}, sigma refit",
         base, {"a": a_alt, "sigma": refit_sigma(ctx.vol_table, a_alt)}),
    ]
    rows = []
    for group, label, asm, hwkw in variants:
        vals = {}
        for s in (-1, 0, 1):
            hw, paths, r = ctx.liability(ctx.curve.bumped(s * SHIFT), asm, **hwkw)
            vals[s] = r.value
            if s == 0:
                pvs = A.pathwise_pv(static, paths) - r.pv
        dur, cvx = risk.duration_convexity(vals[-1], vals[0], vals[1], SHIFT)
        var, es = risk.var_es(pvs)
        rows.append({"group": group, "variant": label, "liability_value": vals[0],
                     "liability_pct_av": vals[0] / ctx.av0, "effective_duration": dur,
                     "effective_convexity": cvx, "static_match_surplus": pvs.mean(),
                     "static_match_var_995": var, "static_match_es_995": es,
                     "sigma": hwkw.get("sigma", s0), "a": hwkw.get("a", a0)})
        print(f"{group:18s} {label:34s} L {vals[0] / 1e6:8.1f}M  D {dur:5.2f}  C {cvx:6.1f}  "
              f"VaR {var / 1e6:6.1f}M  ES {es / 1e6:6.1f}M")
    save(pd.DataFrame(rows), "07_sensitivity.csv")


if __name__ == "__main__":
    main()
