"""Project the block's benefit cash flows along rate paths and value them.

Per quarter, per path, per model point:

    AV grows at the credited rate
    deaths    = AV * q                 (death benefit = account value)
    surrender = (AV - deaths) * w      (paid at AV less the surrender charge)
    AV is paid out at attained age 95, or at the 30-year horizon if that comes first

The credited rate is declared once a year (every policy's anniversary is the
valuation date) as

    credited = max(minimum guaranteed rate, portfolio yield - pricing spread)

and the portfolio yield is a book yield that rolls toward the new money rate: a
fraction `turnover` of the portfolio is reinvested each year at the 7-year par yield on
that path plus the asset spread. That lag is the whole mechanism of the project. When
market rates jump, a new contract can credit the new money rate less the spread while
this block is still credited off last decade's book yield, and the gap is what the
dynamic surrender formula responds to.

Liability value = average over paths of the path-deflated benefit cash flows.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import lapse, mortality
from .block import SC_SCHEDULE, VALUATION_YEAR, ModelPoints
from .hull_white import HullWhite, Paths


@dataclass(frozen=True)
class Assumptions:
    # crediting strategy
    crediting: str = "portfolio"      # "portfolio" or "fixed"
    pricing_spread: float = 0.0175    # credited = portfolio yield - spread
    asset_spread: float = 0.0100      # new money = 7y Treasury par + spread
    nmr_tenor: float = 7.0
    turnover: float = 1.0 / 7.0       # share of the portfolio reinvested per year
    fixed_rate: float = 0.03          # credited rate when crediting == "fixed"
    initial_book_yield: float | None = None   # None: equal to today's new money rate
    # policyholder behaviour
    dynamic: str = "academy"          # "academy", "phbdg" or "none"
    strength: float = 1.0
    base_lapse: bool = True
    mortality: bool = True
    maturity_age: int = 95            # account value paid out at this attained age
    # grid
    horizon: float = 30.0
    dt: float = 0.25

    def but(self, **kw) -> "Assumptions":
        return replace(self, **kw)


def grid(asm: Assumptions) -> np.ndarray:
    return np.arange(int(round(asm.horizon / asm.dt)) + 1) * asm.dt


@dataclass
class Result:
    times: np.ndarray
    cf: np.ndarray                    # (n_paths, K+1) total benefit cash flow
    pv: np.ndarray                    # (n_paths,) deflated PV per path
    detail: dict = field(default_factory=dict)

    @property
    def value(self) -> float:
        return float(self.pv.mean())

    @property
    def stderr(self) -> float:
        return float(self.pv.std(ddof=1) / np.sqrt(len(self.pv))) if len(self.pv) > 1 else 0.0


def new_money_rate(hw: HullWhite, paths: Paths, asm: Assumptions) -> np.ndarray:
    """(n_paths, K+1) new money yield: HW par yield at the reinvestment tenor + spread."""
    out = np.empty_like(paths.x)
    for k, t in enumerate(paths.times):
        out[:, k] = hw.par_yield(t, asm.nmr_tenor, paths.x[:, k])
    return out + asm.asset_spread


def project(mp: ModelPoints, hw: HullWhite, paths: Paths, asm: Assumptions,
            keep_detail: bool = False) -> Result:
    times = paths.times
    dt = asm.dt
    if not np.allclose(np.diff(times), dt):
        raise ValueError("paths must be on the projection grid")
    P, K = paths.n_paths, len(times) - 1

    nmr = new_money_rate(hw, paths, asm)
    book = np.empty_like(nmr)
    book[:, 0] = nmr[:, 0] if asm.initial_book_yield is None else asm.initial_book_yield
    w_roll = 1.0 - np.exp(-asm.turnover * dt) if np.isfinite(asm.turnover) else 1.0
    for k in range(K):
        book[:, k + 1] = book[:, k] + w_roll * (nmr[:, k + 1] - book[:, k])
    competitor = nmr - asm.pricing_spread

    V = np.broadcast_to(mp.av, (P, mp.n_cells)).copy()
    cf = np.zeros((P, K + 1))
    deaths_t = np.zeros((P, K + 1)) if keep_detail else None
    surr_t = np.zeros((P, K + 1)) if keep_detail else None
    sc_t = np.zeros((P, K + 1)) if keep_detail else None
    cred_t = np.zeros((P, K + 1)) if keep_detail else None
    gap_t = np.zeros((P, K + 1)) if keep_detail else None

    sc_arr = np.array(SC_SCHEDULE)
    cr = None
    for k in range(K):
        t = times[k]
        n = int(np.floor(t + 1e-9))
        if k == 0 or abs(t - n) < 1e-9:            # anniversary: declare the rate
            if asm.crediting == "fixed":
                cr = np.maximum(mp.gmir[None, :], asm.fixed_rate) * np.ones((P, 1))
            else:
                cr = np.maximum(mp.gmir[None, :], (book[:, k] - asm.pricing_spread)[:, None])
            py = mp.duration + n + 1
            att = mp.age + n
            sc = np.where(py <= len(sc_arr), sc_arr[np.clip(py - 1, 0, len(sc_arr) - 1)], 0.0)
            ph = lapse.phase(py, len(sc_arr))
            base = lapse.base_rate(py, att, len(sc_arr)) if asm.base_lapse else np.zeros(mp.n_cells)
            q = (mortality.qx(att, mp.female, VALUATION_YEAR + n) if asm.mortality
                 else np.zeros(mp.n_cells))
            q_step = 1.0 - (1.0 - q) ** dt
            growth = (1.0 + cr) ** dt
            matured = att >= asm.maturity_age
            if matured.any():
                cf[:, k] += V[:, matured].sum(axis=1)
                V[:, matured] = 0.0

        w = lapse.total_rate(base[None, :], cr, competitor[:, k][:, None], sc[None, :],
                             ph[None, :], asm.dynamic, asm.strength)
        w_step = 1.0 - (1.0 - w) ** dt
        V = V * growth
        d = V * q_step[None, :]
        s = (V - d) * w_step
        paid = d + s * (1.0 - sc)[None, :]
        cf[:, k + 1] = paid.sum(axis=1)
        V = V - d - s
        if keep_detail:
            deaths_t[:, k + 1] = d.sum(axis=1)
            surr_t[:, k + 1] = s.sum(axis=1)
            sc_t[:, k + 1] = (s * sc[None, :]).sum(axis=1)
            cred_t[:, k] = (cr * V).sum(axis=1) / np.maximum(V.sum(axis=1), 1e-300)
            gap_t[:, k] = competitor[:, k] - (book[:, k] - asm.pricing_spread)
    terminal = V.sum(axis=1)
    cf[:, K] += terminal
    pv = (paths.D * cf).sum(axis=1)
    detail = {}
    if keep_detail:
        detail = {"deaths": deaths_t, "surrenders": surr_t, "surrender_charges": sc_t,
                  "credited": cred_t, "rate_gap": gap_t, "terminal": terminal,
                  "nmr": nmr, "book": book}
    return Result(times, cf, pv, detail)
