import numpy as np
import pytest

from alm.block import VALUATION_DATE, compress, generate_inforce, snfl_rate
from alm.curve import curve_on, load_par_yields
from alm.hull_white import HullWhite
from alm.projection import Assumptions, grid, project

PAR = load_par_yields()
CURVE = curve_on(VALUATION_DATE, PAR)
MP = compress(generate_inforce(PAR))

# credited 3% a year, no surrender, no death, no maturity age, 10-year horizon
FIXED = Assumptions(crediting="fixed", fixed_rate=0.03, base_lapse=False, dynamic="none",
                    mortality=False, maturity_age=999, horizon=10.0)


def _deterministic_answer() -> float:
    credited = np.maximum(MP.gmir, FIXED.fixed_rate)
    return float(np.sum(MP.av * (1 + credited) ** FIXED.horizon) * CURVE.df(FIXED.horizon)[0])


def test_no_lapse_no_mortality_equals_deterministic_discounting_at_zero_vol():
    hw = HullWhite(CURVE, 0.0146, 0.0)
    r = project(MP, hw, hw.simulate(grid(FIXED), 4, seed=0), FIXED)
    assert r.value == pytest.approx(_deterministic_answer(), rel=1e-12)
    # the whole account value comes out at the horizon and nowhere else
    assert np.all(r.cf[:, :-1] == 0.0)


def test_no_lapse_no_mortality_equals_deterministic_discounting_with_volatility():
    """With sigma > 0 the only difference is sampling noise in the deflator."""
    hw = HullWhite(CURVE, 0.0146, 0.0077)
    r = project(MP, hw, hw.simulate(grid(FIXED), 2000, seed=4), FIXED)
    half = len(r.pv) // 2
    pair_se = (0.5 * (r.pv[:half] + r.pv[half:])).std(ddof=1) / np.sqrt(half)
    assert abs(r.value - _deterministic_answer()) < 4 * pair_se


def test_account_value_is_conserved():
    """Credit every cell the same rate c and the account value cannot leak.

    Whatever leaves (deaths, surrenders before the charge, the balance at the horizon)
    left at its own time, so discounting each exit back at c must give exactly the
    starting account value. A decrement applied twice, or a balance never paid out,
    breaks this.
    """
    c = 0.03
    asm = Assumptions(crediting="fixed", fixed_rate=c, maturity_age=999)
    hw = HullWhite(CURVE, 0.0146, 0.0)
    r = project(MP, hw, hw.simulate(grid(asm), 1, seed=0), asm, keep_detail=True)
    d = r.detail
    disc = (1 + c) ** -r.times
    removed = ((d["deaths"][0] + d["surrenders"][0]) * disc).sum() + d["terminal"][0] * disc[-1]
    assert removed == pytest.approx(MP.total_av, rel=1e-12)
    # and the surrender charge is the only gap between account value out and cash paid
    paid = (r.cf[0] * disc).sum()
    assert removed - paid == pytest.approx((d["surrender_charges"][0] * disc).sum(), rel=1e-10)


def test_surrenders_rise_with_rates():
    base = Assumptions(initial_book_yield=CURVE.par_rate(7.0) + 0.01)
    hw0 = HullWhite(CURVE, 0.0146, 0.0)
    out = []
    for bp in (-100, 0, 100, 200):
        c = CURVE.bumped(bp / 1e4)
        h = hw0.with_curve(c)
        r = project(MP, h, h.simulate(grid(base), 1, seed=0), base, keep_detail=True)
        out.append(r.detail["surrenders"][0, 1:13].sum())
    assert np.all(np.diff(out) > 0)


def test_snfl_minimum_rate():
    assert snfl_rate(0.0406) == pytest.approx(0.0280)
    assert snfl_rate(0.0050) == pytest.approx(0.0100)    # floor
    assert snfl_rate(0.0600) == pytest.approx(0.0300)    # cap
