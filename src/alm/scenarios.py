"""Deterministic rate paths: parallel shocks and a replay of the 2022-2023 rise.

A deterministic path lives in the same model with sigma = 0: the curve at time t is
today's forward curve moved by a prescribed factor x(t), with Hull-White loadings, and
the path deflator is P(0,t) * exp(-integral of x). That keeps the stress paths and the
1,000 stochastic paths on one set of pricing formulas.

The 2022-style path replays history on the 7-year point: x(t) is solved quarter by
quarter so that the path's 7-year par yield sits as far above its forward as the
actual 7-year Treasury had risen since 31 Dec 2021, through 29 Sep 2023 (+317 bp in
21 months), and holds there afterwards. With a one-factor model the rest of the curve
moves almost in parallel, so the bear flattening of 2022 is not reproduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .hull_white import HullWhite, Paths

REPLAY_START = pd.Timestamp("2021-12-31")
REPLAY_END = pd.Timestamp("2023-09-29")


def deterministic_paths(hw0: HullWhite, times: np.ndarray, x_path=None) -> Paths:
    if hw0.sigma != 0:
        raise ValueError("deterministic paths need a sigma = 0 model")
    x = np.zeros(len(times)) if x_path is None else np.asarray(x_path, dtype=float)
    inc = 0.5 * np.diff(times) * (x[1:] + x[:-1])
    integral = np.concatenate([[0.0], np.cumsum(inc)])
    D = hw0.curve.df(times) * np.exp(-integral)
    D[0] = 1.0
    return Paths(times, x[None, :], D[None, :], meta={"deterministic": True})


def replay_7y_changes(par: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Change in the 7-year par yield since REPLAY_START, on each grid time."""
    y7 = par["7Yr"]
    base = y7.loc[:REPLAY_START].iloc[-1]
    out = np.zeros(len(times))
    last = 0.0
    for k, t in enumerate(times):
        date = REPLAY_START + pd.DateOffset(months=int(round(t * 12)))
        if date <= REPLAY_END + pd.Timedelta(days=3):
            last = float(y7.loc[:date].iloc[-1] - base)
        out[k] = last
    out[0] = 0.0
    return out


def replay_2022_x(hw0: HullWhite, par: pd.DataFrame, times: np.ndarray,
                  tenor: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    """x(t) whose 7-year par yield move matches the 2022-2023 history. Returns (x, moves)."""
    moves = replay_7y_changes(par, times)
    x = np.zeros(len(times))
    for k, (t, dy) in enumerate(zip(times, moves)):
        if dy == 0.0:
            continue
        y0 = hw0.par_yield(t, tenor, 0.0)[0]
        x[k] = brentq(lambda v: hw0.par_yield(t, tenor, v)[0] - y0 - dy, -0.5, 0.5, xtol=1e-14)
    return x, moves
