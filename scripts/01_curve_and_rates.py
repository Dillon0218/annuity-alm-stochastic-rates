"""Step 1: Treasury curve, Hull-White calibration, and checks on the rate model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import black
from alm.context import N_PATHS, build, save
from alm.curve import TENORS, par_curve_on

MC_CHECK_PATHS = 400_000


def main() -> None:
    ctx = build()
    curve, hw = ctx.curve, ctx.hw

    # curve: bootstrap and reprice the inputs
    asof, tenors, par = par_curve_on(ctx.par, curve.asof)
    rows = []
    for label, t, c in zip(TENORS, tenors, par):
        if t >= 0.5:
            times = np.arange(1, int(round(2 * t)) + 1) / 2
            price = 100 * (c / 2 * curve.df(times).sum() + curve.df(t)[0])
        else:
            price = 100 * (1 + c / 2) ** (-2 * t) / curve.df(t)[0]
        rows.append({"tenor": label, "years": t, "par_yield": c,
                     "zero_cc": float(curve.zero_cc(t)[0]), "discount_factor": float(curve.df(t)[0]),
                     "reprice_par_bond": price})
    tbl = pd.DataFrame(rows)
    save(tbl, "01_curve.csv")
    print(f"curve as of {asof.date()}: 7y par {par[list(TENORS).index('7Yr')]:.2%}; "
          f"max reprice error {np.abs(tbl['reprice_par_bond'] - 100).max():.1e}")

    # calibration
    save(ctx.vol_table, "01_vol_term_structure.csv")
    save(pd.DataFrame([{"a": hw.a, "sigma": hw.sigma,
                        "half_life_years": np.log(2) / hw.a,
                        "window_start": ctx.vol_table["start"].iloc[0],
                        "window_end": ctx.vol_table["end"].iloc[0]}]), "01_hw_parameters.csv")
    print(f"Hull-White a = {hw.a:.4f}, sigma = {hw.sigma:.4%}")

    # martingale test on the production path set, before adjustment
    raw = hw.simulate(ctx.times, N_PATHS, z=ctx.z)
    rows = []
    for T in (1, 2, 5, 10, 15, 20, 30):
        k = int(round(T / (ctx.times[1] - ctx.times[0])))
        d = raw.D[:, k]
        p0 = float(curve.df(T)[0])
        half = N_PATHS // 2
        pair = 0.5 * (d[:half] + d[half:2 * half])           # antithetic pairs
        se = pair.std(ddof=1) / np.sqrt(half)
        rows.append({"maturity": T, "P0T": p0, "mean_D": d.mean(),
                     "error_bp_of_price": 1e4 * (d.mean() / p0 - 1),
                     "stderr_bp_of_price": 1e4 * se / p0, "z": (d.mean() - p0) / se})
    mt = pd.DataFrame(rows)
    save(mt, "01_martingale_test.csv")
    print("martingale test (1,000 paths, unadjusted): max |z| = "
          f"{mt['z'].abs().max():.2f}; 30y error {mt['error_bp_of_price'].iloc[-1]:.0f} bp of price")

    # swaptions: Jamshidian closed form against a large exact simulation
    rows = []
    for expiry, tenor in ((1, 5), (2, 10), (5, 10)):
        fwd, ann = hw.forward_swap(expiry, tenor)
        p = hw.simulate(np.array([0.0, expiry]), MC_CHECK_PATHS, seed=7)
        s, a_path = hw.swap_on_path(expiry, expiry, tenor, p.x[:, 1])
        for dk in (-0.01, 0.0, 0.01):
            k = fwd + dk
            cf = hw.swaption(expiry, tenor, k)
            pay = p.D[:, 1] * a_path * np.maximum(s - k, 0.0)
            se = pay.std(ddof=1) / np.sqrt(MC_CHECK_PATHS)
            rows.append({
                "swaption": f"{expiry}y x {tenor}y", "strike_minus_forward_bp": 1e4 * dk,
                "forward": fwd, "jamshidian": cf, "monte_carlo": pay.mean(), "mc_stderr": se,
                "z": (pay.mean() - cf) / se,
                "normal_vol_bp": 1e4 * black.implied_vol(cf, fwd, k, expiry, ann, "normal"),
                "black_vol": black.implied_vol(cf, fwd, k, expiry, ann, "black"),
            })
    sw = pd.DataFrame(rows)
    save(sw, "01_swaption_check.csv")
    print(f"swaptions: max |z| Jamshidian vs {MC_CHECK_PATHS:,}-path MC = {sw['z'].abs().max():.2f}")


if __name__ == "__main__":
    main()
