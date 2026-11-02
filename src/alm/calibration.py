"""Volatility inputs for Hull-White when no swaption surface is available.

There is no free, citable USD swaption volatility surface, so the model is anchored to
the next best observable: the realized volatility of Treasury zero yields over the
trailing 252 trading days, by maturity. In one-factor Hull-White every zero yield
moves with loading B(tau)/tau on the same factor, so that term structure identifies
both parameters, and sigma (the diffusion coefficient) is the same under the real-world
and the risk-neutral measure. Implied vols normally sit above realized ones, which is
why the results are also shown at 0.5x to 1.5x sigma.

The 1-year point is left out of the fit: the front end is pinned by the policy rate, a
hump a one-factor model cannot produce, and the liability's exposure is at 2 years and
beyond.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curve import TENORS, bootstrap
from .hull_white import b_factor, fit_to_vol_term_structure

VOL_TENORS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
FIT_FROM = 2.0
WINDOW = 252


def realized_zero_vols(par: pd.DataFrame, end, window: int = WINDOW) -> pd.DataFrame:
    sub = par.loc[: pd.Timestamp(end)].iloc[-(window + 1):]
    tenors = list(TENORS.values())
    zeros = np.array([bootstrap(tenors, row.to_numpy()).zero_cc(VOL_TENORS)
                      for _, row in sub.iterrows()])
    vol = np.diff(zeros, axis=0).std(axis=0, ddof=1) * np.sqrt(252)
    return pd.DataFrame({"tenor": VOL_TENORS, "realized_vol": vol,
                         "start": sub.index[0].date(), "end": sub.index[-1].date()})


def calibrate(par: pd.DataFrame, end) -> tuple[float, float, pd.DataFrame]:
    tbl = realized_zero_vols(par, end)
    use = tbl["tenor"] >= FIT_FROM
    a, sigma, _ = fit_to_vol_term_structure(tbl.loc[use, "tenor"], tbl.loc[use, "realized_vol"])
    tbl["model_vol"] = sigma * b_factor(a, tbl["tenor"].to_numpy()) / tbl["tenor"].to_numpy()
    tbl["in_fit"] = use
    return a, sigma, tbl
