"""Step 5: payer swaptions against the surrender option, before and after.

Hedge: 2-year into 10-year payer swaptions struck at the forward swap rate, bought out
of the static match portfolio (bonds are sold pro rata to pay the premium, so total
assets still equal the account value). Notional is set by one rule, chosen before
looking at the tail: the hedge must offset the economic surplus loss in an
instantaneous +200 bp shock. Everything else (the 1,000-path tail, the stress paths) is
then a test of that hedge, not an input to it.

The 2-year expiry was the cheapest per dollar of tail reduction among 2, 3, 4, 5 and
7-year expiries into 10-year swaps (all sized by the same rule); the comparison is
written to 05_hedge_alternatives.csv.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import assets as A
from alm import black, risk
from alm.context import build, load, save
from alm.hedge import Swaption

_spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).with_name("03_assets_and_krd.py"))
step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step3)

EXPIRY, TENOR = 2.0, 10.0
SIZING_SHOCK_BP = 200
ALTERNATIVE_EXPIRIES = (2.0, 3.0, 4.0, 5.0, 7.0)


def sized_hedge(ctx, hw, bonds, liab_by_shift, expiry, tenor):
    """Swaption notional that offsets the surplus loss in the sizing shock."""
    up = ctx.curve.bumped(SIZING_SHOCK_BP / 1e4)
    loss = ((A.value(bonds, ctx.curve) - liab_by_shift[0])
            - (A.value(bonds, up) - liab_by_shift[SIZING_SHOCK_BP]))
    fwd, _ = hw.forward_swap(expiry, tenor)
    unit = Swaption(expiry, tenor, fwd)
    gain = unit.value_under(hw, up) - unit.price(hw)
    # the premium is paid by selling bonds, which also gives up some of their gain in
    # the shock; solve for the notional with that included
    a0 = A.value(bonds, ctx.curve)
    bond_loss_per_dollar = (a0 - A.value(bonds, up)) / a0
    per_notional = gain + unit.price(hw) * bond_loss_per_dollar
    return unit.scaled(loss / per_notional)


def fund(bonds, premium, av0):
    scale = (av0 - premium) / av0
    return [A.Bond(b.maturity, b.coupon, b.face * scale, b.spread) for b in bonds]


def main() -> None:
    ctx = build()
    static = step3.portfolios(ctx)["static_match"][0]
    hw, paths, r = ctx.liability()
    liab = load("02_value_vs_shift.csv").set_index("shift_bp")["dynamic"]

    base_pvs = A.pathwise_pv(static, paths) - r.pv
    alt = []
    for expiry in ALTERNATIVE_EXPIRIES:
        sw = sized_hedge(ctx, hw, static, liab, expiry, TENOR)
        prem = sw.price(hw)
        pvs = A.pathwise_pv(fund(static, prem, ctx.av0), paths) + sw.pathwise_pv(hw, paths) - r.pv
        var, es = risk.var_es(pvs)
        alt.append({"swaption": f"{expiry:g}y x {TENOR:g}y", "notional": sw.notional,
                    "premium": prem, "var_995": var, "es_995": es,
                    "es_reduction_per_premium": (risk.var_es(base_pvs)[1] - es) / prem})
    alt = pd.DataFrame(alt)
    save(alt, "05_hedge_alternatives.csv")
    print(alt.round(3).to_string(index=False))

    sw = sized_hedge(ctx, hw, static, liab, EXPIRY, TENOR)
    prem = sw.price(hw)
    hedged_bonds = fund(static, prem, ctx.av0)
    hedged_pvs = A.pathwise_pv(hedged_bonds, paths) + sw.pathwise_pv(hw, paths) - r.pv

    fwd, ann = hw.forward_swap(EXPIRY, TENOR)
    unit = sw.price(hw) / sw.notional
    rows = []
    for name, pvs in (("static_match", base_pvs), ("static_match_plus_payers", hedged_pvs)):
        var, es = risk.var_es(pvs)
        rows.append({"portfolio": name, "mean": pvs.mean(), "sd": pvs.std(ddof=1),
                     "var_995": var, "es_995": es, **risk.bootstrap_var_es(pvs)})
    summ = pd.DataFrame(rows)
    summ["swaption"] = f"{EXPIRY:g}y x {TENOR:g}y payer, ATM"
    summ["strike"] = fwd
    summ["notional"] = [0.0, sw.notional]
    summ["premium"] = [0.0, prem]
    summ["premium_pct_av"] = summ["premium"] / ctx.av0
    summ["normal_vol_bp"] = 1e4 * black.implied_vol(unit, fwd, fwd, EXPIRY, ann, "normal")
    summ["black_vol"] = black.implied_vol(unit, fwd, fwd, EXPIRY, ann, "black")
    save(summ, "05_hedge_summary.csv")
    print(summ.round(4).to_string(index=False))

    dist = load("04_surplus_by_path.csv")
    dist["static_match_plus_payers__dynamic"] = hedged_pvs
    save(dist, "04_surplus_by_path.csv")

    prof = load("04_surplus_vs_shift.csv")
    prof["static_match_plus_payers_surplus"] = [
        A.value(hedged_bonds, ctx.curve, s / 1e4)
        + sw.value_under(hw, ctx.curve.bumped(s / 1e4)) - liab[s] for s in prof["shift_bp"]]
    save(prof, "04_surplus_vs_shift.csv")
    print((prof.set_index("shift_bp")[["static_match_surplus", "static_match_plus_payers_surplus",
                                       "effective_match_surplus"]] / 1e6).round(1).to_string())
    pd.DataFrame([{"expiry": EXPIRY, "tenor": TENOR, "strike": fwd, "notional": sw.notional,
                   "premium": prem}]).to_csv(
        Path(__file__).resolve().parents[1] / "outputs" / "tables" / "05_hedge_position.csv",
        index=False)


if __name__ == "__main__":
    main()
