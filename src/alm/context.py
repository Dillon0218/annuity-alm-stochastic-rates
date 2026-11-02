"""Everything the pipeline steps share, built once from the committed data.

Every stochastic valuation reuses the same 1,000 x 120 x 2 normal draws. A curve shift
changes P(0,t), not the draws, so bumped revaluations differ only by the bump: that is
what keeps finite-difference durations, convexities and key rate durations free of
Monte Carlo noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import assets as A
from .block import VALUATION_DATE, ModelPoints, compress, generate_inforce
from .calibration import calibrate
from .curve import ZeroCurve, curve_on, load_par_yields
from .hull_white import HullWhite, Paths, standard_normals
from .projection import Assumptions, Result, grid, project

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

N_PATHS = 1_000
SEED = 20261030
ASSET_SPREAD = 0.0100
BARBELL = (2.0, 20.0)
SHIFT = 0.0050                     # +-50 bp for effective duration and convexity
KEY_RATES = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
KRD_BUMP = 0.0025


@dataclass(frozen=True)
class Context:
    par: pd.DataFrame
    curve: ZeroCurve
    hw: HullWhite
    vol_table: pd.DataFrame
    inforce: pd.DataFrame
    mp: ModelPoints
    asm: Assumptions
    times: np.ndarray
    z: np.ndarray

    @property
    def av0(self) -> float:
        return self.mp.total_av

    def model(self, curve: ZeroCurve | None = None, sigma: float | None = None,
              a: float | None = None) -> HullWhite:
        return HullWhite(curve or self.curve, self.hw.a if a is None else a,
                         self.hw.sigma if sigma is None else sigma)

    def paths(self, hw: HullWhite) -> Paths:
        """The shared draws under `hw`, martingale-adjusted to its curve."""
        return hw.simulate(self.times, N_PATHS, z=self.z).martingale_adjusted(hw.curve)

    def liability(self, curve: ZeroCurve | None = None, asm: Assumptions | None = None,
                  sigma: float | None = None, a: float | None = None,
                  keep_detail: bool = False) -> tuple[HullWhite, Paths, Result]:
        hw = self.model(curve, sigma, a)
        p = self.paths(hw)
        return hw, p, project(self.mp, hw, p, asm or self.asm, keep_detail)

    def deterministic_liability(self, curve: ZeroCurve | None = None,
                                asm: Assumptions | None = None) -> Result:
        from .scenarios import deterministic_paths
        hw0 = self.model(curve, sigma=0.0)
        return project(self.mp, hw0, deterministic_paths(hw0, self.times), asm or self.asm)

    def static_duration(self) -> tuple[float, float]:
        """(PV, duration) of the base-scenario cash flows held fixed."""
        r = self.deterministic_liability()
        pv = r.cf[0] * self.curve.df(self.times)
        return float(pv.sum()), float((self.times * pv).sum() / pv.sum())

    def barbell(self, dollar_duration: float, market_value: float | None = None):
        mv = self.av0 if market_value is None else market_value
        return A.barbell(self.curve, mv, dollar_duration, *BARBELL, ASSET_SPREAD)


@lru_cache(maxsize=1)
def build() -> Context:
    par = load_par_yields()
    curve = curve_on(VALUATION_DATE, par)
    a, sigma, vol_table = calibrate(par, VALUATION_DATE)
    inforce = generate_inforce(par)
    mp = compress(inforce)
    # The book yield of the existing portfolio is history: it is today's new money
    # rate and it does not move when the curve is shocked.
    book0 = curve.par_rate(7.0) + ASSET_SPREAD
    asm = Assumptions(asset_spread=ASSET_SPREAD, initial_book_yield=book0)
    times = grid(asm)
    z = standard_normals(N_PATHS, len(times) - 1, SEED)
    return Context(par, curve, HullWhite(curve, a, sigma), vol_table, inforce, mp, asm,
                   times, z)


def save(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    df.to_csv(path, index=False)
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)
