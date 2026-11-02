import numpy as np
import pytest

from alm.block import VALUATION_DATE
from alm.curve import curve_on, load_par_yields
from alm.hull_white import HullWhite

CURVE = curve_on(VALUATION_DATE, load_par_yields())
GRID = np.arange(121) * 0.25


def test_zero_volatility_reproduces_the_initial_curve():
    """sigma = 0: every path's deflator is today's discount factor, exactly."""
    hw = HullWhite(CURVE, 0.0146, 0.0)
    paths = hw.simulate(GRID, 50, seed=1)
    assert np.all(paths.x == 0.0)
    np.testing.assert_allclose(paths.D, np.broadcast_to(CURVE.df(GRID), paths.D.shape),
                               rtol=1e-14, atol=0)


def test_zero_volatility_bond_prices_are_forward_prices():
    hw = HullWhite(CURVE, 0.0146, 0.0)
    for t in (1.0, 5.0, 12.5):
        T = t + np.array([0.5, 3.0, 10.0])
        np.testing.assert_allclose(hw.zero_bond(t, T, 0.0), CURVE.df(T) / CURVE.df(t)[0],
                                   rtol=1e-14)


@pytest.mark.parametrize("t,T", [(2.0, 12.0), (10.0, 17.0), (20.0, 27.0)])
def test_path_bond_prices_are_martingales(t, T):
    """E[D(t) P(t,T)] = P(0,T). Guards the alpha variance term in the bond formula."""
    hw = HullWhite(CURVE, 0.0146, 0.0077)
    p = hw.simulate(np.array([0.0, t]), 400_000, seed=3)
    disc_bond = p.D[:, 1] * hw.zero_bond(t, np.array([T]), p.x[:, 1][:, None])[:, 0]
    se = disc_bond.std(ddof=1) / np.sqrt(len(disc_bond))
    assert abs(disc_bond.mean() - CURVE.df(T)[0]) < 4 * se


def test_exact_simulation_matches_the_ou_moments():
    hw = HullWhite(CURVE, 0.05, 0.01)
    p = hw.simulate(np.array([0.0, 3.0, 10.0]), 400_000, seed=5, antithetic=False)
    for k, t in ((1, 3.0), (2, 10.0)):
        assert np.var(p.x[:, k]) == pytest.approx(float(hw.var_x(t)), rel=0.01)


def test_jamshidian_swaption_matches_simulation():
    hw = HullWhite(CURVE, 0.0146, 0.0077)
    expiry, tenor = 5.0, 10.0
    fwd, _ = hw.forward_swap(expiry, tenor)
    p = hw.simulate(np.array([0.0, expiry]), 400_000, seed=11)
    s, ann = hw.swap_on_path(expiry, expiry, tenor, p.x[:, 1])
    for k in (fwd - 0.01, fwd, fwd + 0.01):
        pay = p.D[:, 1] * ann * np.maximum(s - k, 0.0)
        se = pay.std(ddof=1) / np.sqrt(len(pay))
        assert abs(pay.mean() - hw.swaption(expiry, tenor, k)) < 4 * se


def test_payer_receiver_parity():
    hw = HullWhite(CURVE, 0.0146, 0.0077)
    for expiry, tenor in ((1.0, 5.0), (2.0, 10.0), (5.0, 10.0)):
        fwd, ann = hw.forward_swap(expiry, tenor)
        k = fwd + 0.004
        lhs = hw.swaption(expiry, tenor, k) - hw.swaption(expiry, tenor, k, "receiver")
        assert lhs == pytest.approx(ann * (fwd - k), abs=1e-12)


def test_martingale_adjustment_reprices_the_curve():
    hw = HullWhite(CURVE, 0.0146, 0.0077)
    p = hw.simulate(GRID, 1000, seed=2).martingale_adjusted(CURVE)
    np.testing.assert_allclose(p.D.mean(axis=0), CURVE.df(GRID) * (GRID > 0) + (GRID == 0),
                               rtol=1e-12)
