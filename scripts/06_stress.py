"""Step 6: deterministic stress paths, including a replay of the 2022-2023 rise.

Each scenario is one rate path. The surplus on it is the same run-off present value as
in step 4: assets held to maturity, benefits paid, everything deflated along the path.
The impact is measured against the unstressed path (today's forward curve).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import assets as A
from alm.context import build, load, save
from alm.hedge import Swaption
from alm.projection import project
from alm.scenarios import deterministic_paths, replay_2022_x

_spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).with_name("03_assets_and_krd.py"))
step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step3)


def main() -> None:
    ctx = build()
    ports = step3.portfolios(ctx)
    pos = load("05_hedge_position.csv").iloc[0]
    sw = Swaption(pos.expiry, pos.tenor, pos.strike, pos.notional)
    scale = (ctx.av0 - pos.premium) / ctx.av0
    hedged_bonds = [A.Bond(b.maturity, b.coupon, b.face * scale, b.spread)
                    for b in ports["static_match"][0]]

    hw0 = ctx.model(sigma=0.0)
    x_replay, moves = replay_2022_x(hw0, ctx.par, ctx.times)
    scenarios = {"base": (0, None)}
    for bp in (-200, -100, 100, 200):
        scenarios[f"parallel {bp:+d} bp"] = (bp, None)
    scenarios["2022 replay"] = (0, x_replay)

    rows, replay_detail = [], None
    q = 4
    for name, (bp, x) in scenarios.items():
        h = ctx.model(ctx.curve.bumped(bp / 1e4), sigma=0.0)
        p = deterministic_paths(h, ctx.times, x)
        r = project(ctx.mp, h, p, ctx.asm, keep_detail=True)
        d = r.detail
        surr = d["surrenders"][0]
        row = {"scenario": name,
               "liability_pv": r.pv[0],
               "static_match": A.pathwise_pv(ports["static_match"][0], p)[0] - r.pv[0],
               "static_match_plus_payers": (A.pathwise_pv(hedged_bonds, p)[0]
                                            + sw.pathwise_pv(h, p)[0] - r.pv[0]),
               "effective_match": A.pathwise_pv(ports["effective_match"][0], p)[0] - r.pv[0],
               "surrenders_yr1_3": surr[1: 3 * q + 1].sum(),
               # new contract rate less the block's declared (AV-weighted) credited rate
               "max_rate_gap_bp": 1e4 * np.max((d["nmr"][0] - ctx.asm.pricing_spread
                                                - d["credited"][0])[:-1]),
               "payer_payoff_at_expiry": sw.pathwise_pv(h, p)[0] / p.D[0, int(pos.expiry * q)]}
        rows.append(row)
        if name == "2022 replay" or name == "base":
            frame = pd.DataFrame({"scenario": name, "t": ctx.times, "x": p.x[0],
                                  "new_money": d["nmr"][0], "book_yield": d["book"][0],
                                  "credited": d["credited"][0], "rate_gap": d["rate_gap"][0],
                                  "surrenders": surr, "seven_year_move": moves if name != "base" else 0.0})
            replay_detail = frame if replay_detail is None else pd.concat([replay_detail, frame])
    out = pd.DataFrame(rows)
    base = out.iloc[0]
    for col in ("static_match", "static_match_plus_payers", "effective_match"):
        out[f"{col}_impact"] = out[col] - base[col]
    out["surrenders_yr1_3_vs_base"] = out["surrenders_yr1_3"] / base["surrenders_yr1_3"] - 1
    save(out, "06_stress.csv")
    save(replay_detail, "06_replay_path.csv")
    show = out.set_index("scenario")[[c for c in out.columns if c.endswith("_impact")]
                                     + ["surrenders_yr1_3", "max_rate_gap_bp"]]
    print((show.assign(**{c: show[c] / 1e6 for c in show.columns if c != "max_rate_gap_bp"}))
          .round(1).to_string())


if __name__ == "__main__":
    main()
