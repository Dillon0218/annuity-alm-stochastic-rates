"""The asset side: option-free bonds priced at a constant spread over Treasuries.

Each bond is a par bond today at Treasury zeros plus `spread` (continuously compounded),
paying semi-annual coupons. Holding the spread fixed means every change in asset value
comes from the rate curve, which is the risk this project is about. The spread is the
compensation for credit risk and is treated as such: it is not counted as free excess
return in any valuation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curve import ZeroCurve
from .hull_white import Paths


@dataclass(frozen=True)
class Bond:
    maturity: float
    coupon: float
    face: float
    spread: float

    def schedule(self, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
        times = np.arange(1, int(round(self.maturity * m)) + 1) / m
        cf = np.full(len(times), self.face * self.coupon / m)
        cf[-1] += self.face
        return times, cf


def par_coupon(curve: ZeroCurve, maturity: float, spread: float, m: int = 2) -> float:
    times = np.arange(1, int(round(maturity * m)) + 1) / m
    dfs = curve.df(times) * np.exp(-spread * times)
    return float(m * (1.0 - dfs[-1]) / dfs.sum())


def value(bonds, curve: ZeroCurve, shift: float = 0.0) -> float:
    """Market value under `curve` shifted in parallel by `shift` (decimal)."""
    total = 0.0
    for b in bonds:
        t, cf = b.schedule()
        total += float(np.sum(cf * curve.df(t) * np.exp(-(b.spread + shift) * t)))
    return total


def duration(bonds, curve: ZeroCurve) -> float:
    """Effective (= modified, continuous) duration from the analytic derivative."""
    pv, tpv = 0.0, 0.0
    for b in bonds:
        t, cf = b.schedule()
        d = cf * curve.df(t) * np.exp(-b.spread * t)
        pv += d.sum()
        tpv += (t * d).sum()
    return tpv / pv


def barbell(curve: ZeroCurve, market_value: float, dollar_duration: float,
            short: float, long: float, spread: float) -> list[Bond]:
    """Two par bonds whose value and dollar duration match the targets."""
    legs = []
    for T in (short, long):
        c = par_coupon(curve, T, spread)
        legs.append(Bond(T, c, 1.0, spread))
    d1, d2 = duration([legs[0]], curve), duration([legs[1]], curve)
    w2 = (dollar_duration - market_value * d1) / (d2 - d1)
    w1 = market_value - w2
    if w1 < 0 or w2 < 0:
        raise ValueError(f"target duration {dollar_duration / market_value:.2f} is "
                         f"outside [{d1:.2f}, {d2:.2f}]; pick other maturities")
    return [Bond(short, legs[0].coupon, w1, spread), Bond(long, legs[1].coupon, w2, spread)]


def cash_flows_on_grid(bonds, times: np.ndarray) -> np.ndarray:
    """Spread-adjusted asset cash flows on the projection grid.

    Each coupon is multiplied by exp(-spread * t), the risk-neutral survival implied by
    pricing at a constant spread, so that deflating these along a path and averaging
    gives back the market value.
    """
    out = np.zeros(len(times))
    for b in bonds:
        t, cf = b.schedule()
        idx = np.searchsorted(times, t - 1e-9)
        if not np.allclose(times[idx], t):
            raise ValueError("bond coupon dates must lie on the projection grid")
        np.add.at(out, idx, cf * np.exp(-b.spread * t))
    return out


def pathwise_pv(bonds, paths: Paths) -> np.ndarray:
    """PV of the asset cash flows along each path, deflated at the path's short rate."""
    return paths.D @ cash_flows_on_grid(bonds, paths.times)
