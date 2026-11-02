import numpy as np
import pytest

from alm import assets as A
from alm import black, risk
from alm.block import VALUATION_DATE
from alm.curve import TENORS, bootstrap, curve_on, key_rate_bump, load_par_yields, par_curve_on
from alm.hull_white import HullWhite

PAR = load_par_yields()
CURVE = curve_on(VALUATION_DATE, PAR)


def test_bootstrap_reprices_every_input_par_bond():
    _, tenors, par = par_curve_on(PAR, VALUATION_DATE)
    for t, c in zip(tenors, par):
        if t >= 0.5:
            times = np.arange(1, int(round(2 * t)) + 1) / 2
            assert c / 2 * CURVE.df(times).sum() + CURVE.df(t)[0] == pytest.approx(1.0, abs=1e-12)


def test_key_rate_bumps_add_up_to_a_parallel_shift():
    keys = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
    t = np.linspace(0.05, 30, 400)
    total = sum(key_rate_bump(keys, k, 1.0)(t) for k in range(len(keys)))
    np.testing.assert_allclose(total, 1.0)


def test_barbell_matches_value_and_dollar_duration():
    bonds = A.barbell(CURVE, 1e9, 5.0e9, 2.0, 20.0, 0.01)
    assert A.value(bonds, CURVE) == pytest.approx(1e9, rel=1e-12)
    assert A.value(bonds, CURVE) * A.duration(bonds, CURVE) == pytest.approx(5.0e9, rel=1e-12)
    h = 1e-5
    num = (A.value(bonds, CURVE, -h) - A.value(bonds, CURVE, h)) / (2 * h * 1e9)
    assert num == pytest.approx(5.0, rel=1e-8)


def test_asset_cash_flows_average_back_to_market_value():
    bonds = A.barbell(CURVE, 1e9, 5.0e9, 2.0, 20.0, 0.01)
    hw = HullWhite(CURVE, 0.0146, 0.0077)
    p = hw.simulate(np.arange(121) * 0.25, 1000, seed=0).martingale_adjusted(CURVE)
    assert A.pathwise_pv(bonds, p).mean() == pytest.approx(A.value(bonds, CURVE), rel=1e-12)


def test_black76_parity_and_round_trip():
    f, k, t, ann = 0.05, 0.055, 2.0, 7.5
    pay = black.black76(f, k, t, 0.2, ann, "payer")
    rec = black.black76(f, k, t, 0.2, ann, "receiver")
    assert pay - rec == pytest.approx(ann * (f - k), abs=1e-14)
    assert black.implied_vol(pay, f, k, t, ann, "black") == pytest.approx(0.2, abs=1e-10)


def test_bachelier_at_the_money_closed_form():
    f, t, ann, vol = 0.05, 3.0, 6.0, 0.008
    assert black.bachelier(f, f, t, vol, ann) == pytest.approx(ann * vol * np.sqrt(t / (2 * np.pi)))
    assert black.implied_vol(black.bachelier(f, f, t, vol, ann), f, f, t, ann, "normal") \
        == pytest.approx(vol, abs=1e-12)


def test_var_es_on_a_normal_sample():
    x = np.random.default_rng(0).standard_normal(2_000_000)
    var, es = risk.var_es(x, 0.995)
    assert var == pytest.approx(2.5758, abs=0.01)
    assert es == pytest.approx(2.8919, abs=0.01)     # phi(z)/(1-q)
    assert es > var
