"""Treasury par yields -> zero curve.

Adapted from `rates.py` in the author's behavioural-cashflow-engine, where the bootstrap
was validated by repricing every input par bond to 100 (max error 1.4e-14). The upgrade
over the coursework version is the same one made there: CMT quotes are *par* yields,
so they are bootstrapped into zeros before anything is discounted. At 20 years the
difference is tens of basis points, which is not a rounding question for a liability
whose cash flows run 30 years.

Zeros are stored continuously compounded and interpolated linearly, flat beyond the
ends. That reproduces every input exactly and keeps discount factors monotone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAR_YIELDS_CSV = ROOT / "data" / "raw" / "treasury_par_yields.csv"

#: Column label -> tenor in years, in the order of the committed CSV.
TENORS: dict[str, float] = {
    "1Mo": 1 / 12, "3Mo": 0.25, "6Mo": 0.5, "1Yr": 1.0, "2Yr": 2.0, "3Yr": 3.0,
    "5Yr": 5.0, "7Yr": 7.0, "10Yr": 10.0, "20Yr": 20.0, "30Yr": 30.0,
}


def load_par_yields(path: str | Path | None = None) -> pd.DataFrame:
    """Daily par yields in decimals, date-indexed, complete days only."""
    p = Path(path) if path is not None else PAR_YIELDS_CSV
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return (df.loc[:, list(TENORS)] / 100.0).dropna()


def par_curve_on(par: pd.DataFrame, date) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """(as-of date, tenors, par yields) on the last date at or before `date`."""
    sub = par.loc[: pd.Timestamp(date)]
    if sub.empty:
        raise ValueError(f"no curve at or before {date}")
    return sub.index[-1], np.array(list(TENORS.values())), sub.iloc[-1].to_numpy(float)


@dataclass(frozen=True)
class ZeroCurve:
    """Discount curve held as continuously compounded zeros at knot times."""

    knot_years: np.ndarray
    knot_zeros: np.ndarray
    asof: pd.Timestamp | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.knot_years, dtype=float)
        z = np.asarray(self.knot_zeros, dtype=float)
        if len(t) != len(z) or len(t) < 2:
            raise ValueError("need matching knot times and rates, at least two")
        if (np.diff(t) <= 0).any() or (t <= 0).any():
            raise ValueError("knot times must be positive and strictly increasing")
        object.__setattr__(self, "knot_years", t)
        object.__setattr__(self, "knot_zeros", z)

    def zero_cc(self, years) -> np.ndarray:
        t = np.atleast_1d(np.asarray(years, dtype=float))
        return np.interp(t, self.knot_years, self.knot_zeros)

    def df(self, years) -> np.ndarray:
        t = np.atleast_1d(np.asarray(years, dtype=float))
        return np.exp(-self.zero_cc(t) * t)

    def par_rate(self, maturity: float, m: int = 2) -> float:
        """Coupon that prices an `m`-times-a-year bond at par."""
        n = int(round(maturity * m))
        times = np.arange(1, n + 1) / m
        dfs = self.df(times)
        return float(m * (1.0 - dfs[-1]) / dfs.sum())

    def bumped(self, bump) -> "ZeroCurve":
        """Add `bump(t)` (decimal) to every zero knot. A float is a parallel shift."""
        b = (np.full_like(self.knot_zeros, float(bump)) if np.isscalar(bump)
             else np.asarray(bump(self.knot_years), dtype=float))
        return ZeroCurve(self.knot_years, self.knot_zeros + b, asof=self.asof,
                         meta={**self.meta, "bumped": True})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"years": self.knot_years, "zero_cc": self.knot_zeros,
                             "df": self.df(self.knot_years)})


def bootstrap(years, par_rates, *, m: int = 2, asof=None) -> ZeroCurve:
    """Bootstrap par yields into zeros along the coupon grid.

        DF(T) = (1 - (c/m) * sum_{i<n} DF(i/m)) / (1 + c/m)

    Par yields are interpolated linearly onto every coupon date first (the standard
    order; interpolating zeros afterwards would not reprice the inputs). Tenors inside
    one coupon period have a single cash flow, so the par quote already is the zero.
    """
    years = np.asarray(years, dtype=float)
    par = np.asarray(par_rates, dtype=float)
    order = np.argsort(years)
    years, par = years[order], par[order]
    step = 1.0 / m
    knot_t, knot_df = [], []
    for t, c in zip(years, par):
        if t < step - 1e-12:
            knot_t.append(float(t))
            knot_df.append(float((1.0 + c / m) ** (-m * t)))
    grid = np.arange(1, int(round(years[-1] * m)) + 1) * step
    acc = 0.0
    for t, c in zip(grid, np.interp(grid, years, par)):
        df = (1.0 - (c / m) * acc) / (1.0 + c / m)
        if df <= 0:
            raise ValueError(f"non-positive discount factor at {t}y")
        acc += df
        knot_t.append(float(t))
        knot_df.append(float(df))
    t_arr, df_arr = np.array(knot_t), np.array(knot_df)
    return ZeroCurve(t_arr, -np.log(df_arr) / t_arr,
                     asof=None if asof is None else pd.Timestamp(asof))


def curve_on(date, par: pd.DataFrame | None = None) -> ZeroCurve:
    par = load_par_yields() if par is None else par
    asof, tenors, rates = par_curve_on(par, date)
    return bootstrap(tenors, rates, asof=asof)


def key_rate_bump(key_years, k: int, size: float):
    """Triangular bump of `size` peaking at key `k`, zero at its neighbours.

    The bumps over all keys sum to a parallel shift (flat beyond the end keys), so the
    key rate durations sum to the effective duration.
    """
    keys = np.asarray(key_years, dtype=float)

    def bump(t):
        t = np.asarray(t, dtype=float)
        w = np.zeros_like(t)
        if k == 0:
            w = np.where(t <= keys[0], 1.0, w)
        if k == len(keys) - 1:
            w = np.where(t >= keys[-1], 1.0, w)
        if k > 0:
            lo, mid = keys[k - 1], keys[k]
            m = (t > lo) & (t <= mid)
            w = np.where(m, (t - lo) / (mid - lo), w)
        if k < len(keys) - 1:
            mid, hi = keys[k], keys[k + 1]
            m = (t > mid) & (t < hi)
            w = np.where(m, (hi - t) / (hi - mid), w)
        return size * w

    return bump
