"""Surrender assumptions: a base table plus a dynamic adjustment for the rate incentive.

Base rates
----------
The NAIC VM-22 Standard Projection Amount proposal for fixed indexed annuities without
a living benefit, developed from the 2019/2020 LIMRA FIA study, by attained age group
and position relative to the end of the surrender charge period. Source:
"Standard Projection Amount: Surrenders, Withdrawals and Dynamic Lapse Assumptions for
FIA", NAIC VM-22 Subgroup call materials. The SOA/LIMRA fixed rate deferred study has
the same shape (low inside the charge period, a shock in the year it expires, elevated
afterwards) but publishes its rates only to subscribers, so this is the closest public
table with the same granularity.

Dynamic adjustment
------------------
The American Academy of Actuaries' linear formula from the same NAIC materials:

    total lapse  = base + rate factor
    rate factor  = market factor * max(0, 1 - 10 * surrender charge)
    market factor = Y * (MR - BF - CR)    if CR <= MR - BF     (credited rate behind)
                  = 0                     if MR - BF < CR < MR (inside the buffer)
                  = -X * (CR - MR)        if CR >= MR          (credited rate ahead)

with X = 1 / 5 / 3 and Y = 3 / 5 / 6 inside the charge period / in the shock year /
after, buffer BF = 25 bp, and the rate factor bounded to [-2%, +10%] inside the charge
period, [-4%, +60%] in the shock year and [-2%, +35%] after. The NAIC page prints the
CR >= MR branch as "X * (CR - MR)"; it is read here with a minus sign, which is what the
-2% floor implies. The CR < MR branch reproduces the NAIC worked examples exactly
(tests/test_lapse.py): 5.13%, 43.75% and 20.50% at a 2% rate gap.

The drafting group's alternative (PHBDG), a power form with exponent 2.5, is kept as a
sensitivity.
"""
from __future__ import annotations

import numpy as np

SC_PERIOD = 7

#: Rows: 5+ yrs to expiry, 4, 3, 2, 1 yr to expiry, year SC expires (shock),
#: 1, 2, 3, 4, 5+ yrs after. Columns: attained age <60, 60-69, 70-79, 80+.
BASE_TABLE = np.array([
    [0.020, 0.020, 0.020, 0.020],
    [0.025, 0.025, 0.025, 0.020],
    [0.030, 0.030, 0.030, 0.025],
    [0.040, 0.035, 0.030, 0.030],
    [0.045, 0.035, 0.040, 0.040],
    [0.335, 0.415, 0.370, 0.235],
    [0.150, 0.175, 0.135, 0.090],
    [0.110, 0.120, 0.090, 0.070],
    [0.085, 0.095, 0.070, 0.055],
    [0.080, 0.085, 0.065, 0.050],
    [0.065, 0.070, 0.060, 0.050],
])
SHOCK_ROW = 5

# phase index: 0 inside the surrender charge period, 1 shock year, 2 after
_X = np.array([1.0, 5.0, 3.0])
_Y = np.array([3.0, 5.0, 6.0])
_RF_MIN = np.array([-0.02, -0.04, -0.02])
_RF_MAX = np.array([0.10, 0.60, 0.35])
ACADEMY_BUFFER = 0.0025
PHBDG_BUFFER = 0.0050


def table_row(policy_year, sc_period: int = SC_PERIOD) -> np.ndarray:
    """Row of BASE_TABLE for a (1-based) policy year."""
    py = np.asarray(policy_year, dtype=int)
    years_to_expiry = sc_period + 1 - py          # 1 in the last charge year
    in_sc = np.clip(SHOCK_ROW - years_to_expiry, 0, SHOCK_ROW - 1)
    after = np.clip(SHOCK_ROW + (py - sc_period - 1), SHOCK_ROW, len(BASE_TABLE) - 1)
    return np.where(py <= sc_period, in_sc, after)


def phase(policy_year, sc_period: int = SC_PERIOD) -> np.ndarray:
    py = np.asarray(policy_year, dtype=int)
    return np.where(py <= sc_period, 0, np.where(py == sc_period + 1, 1, 2))


def age_group(attained_age) -> np.ndarray:
    a = np.asarray(attained_age)
    return np.where(a < 60, 0, np.where(a < 70, 1, np.where(a < 80, 2, 3)))


def base_rate(policy_year, attained_age, sc_period: int = SC_PERIOD) -> np.ndarray:
    return BASE_TABLE[table_row(policy_year, sc_period), age_group(attained_age)]


def academy_rate_factor(cr, mr, surrender_charge, ph) -> np.ndarray:
    """Academy rate factor, decimal. Broadcasts over all inputs."""
    cr, mr = np.asarray(cr, dtype=float), np.asarray(mr, dtype=float)
    ph = np.asarray(ph, dtype=int)
    behind = _Y[ph] * np.maximum(mr - ACADEMY_BUFFER - cr, 0.0)
    ahead = -_X[ph] * np.maximum(cr - mr, 0.0)
    mf = np.where(cr >= mr, ahead, behind)
    rf = mf * np.maximum(0.0, 1.0 - 10.0 * np.asarray(surrender_charge, dtype=float))
    return np.clip(rf, _RF_MIN[ph], _RF_MAX[ph])


def phbdg_total(base, cr, mr, surrender_charge) -> np.ndarray:
    """PHBDG total lapse, decimal: power 2.5 on the gap in percent, total in [1%, 60%]."""
    cr, mr = np.asarray(cr, dtype=float), np.asarray(mr, dtype=float)
    gap_behind = np.maximum(mr - PHBDG_BUFFER - cr, 0.0) * 100
    gap_ahead = np.maximum(cr - mr, 0.0) * 100
    mf = (1.25 * gap_behind ** 2.5 - 1.25 * gap_ahead ** 2.5) / 100
    rf = mf * np.maximum(0.0, 1.0 - 5.0 * np.asarray(surrender_charge, dtype=float))
    return np.clip(np.asarray(base) + rf, 0.01, 0.60)


def total_rate(base, cr, mr, surrender_charge, ph, model: str = "academy",
               strength: float = 1.0) -> np.ndarray:
    """Annual surrender rate under the chosen dynamic model.

    `strength` scales the rate factor (0 switches dynamic lapse off). For PHBDG it
    scales the difference between the PHBDG total and the base rate.
    """
    base = np.asarray(base, dtype=float)
    if model == "none" or strength == 0:
        return np.clip(base + 0.0 * np.asarray(cr), 0.0, 1.0)
    if model == "academy":
        rf = academy_rate_factor(cr, mr, surrender_charge, ph)
        return np.clip(base + strength * rf, 0.0, 1.0)
    if model == "phbdg":
        tot = phbdg_total(base, cr, mr, surrender_charge)
        return np.clip(base + strength * (tot - base), 0.0, 1.0)
    raise ValueError(f"unknown dynamic lapse model {model!r}")
