"""The dynamic surrender formulas against the NAIC worked examples.

Source: NAIC VM-22 Subgroup, "Dynamic Lapse Comparison - SCL vs Academy vs NAIC
Drafting Group" (13 Dec 2023): base 2.5% with a 5% surrender charge, 35% in the shock
year, 10% after, at market-less-credited gaps of 2% to 5%.
"""
import numpy as np
import pytest

from alm import lapse

GAPS = np.array([0.02, 0.03, 0.04, 0.05])
CR = 0.03


@pytest.mark.parametrize("base,sc,phase,expected", [
    (0.025, 0.05, 0, [0.0513, 0.0663, 0.0813, 0.0963]),
    (0.350, 0.00, 1, [0.4375, 0.4875, 0.5375, 0.5875]),
    (0.100, 0.00, 2, [0.2050, 0.2650, 0.3250, 0.3850]),
])
def test_academy_reproduces_naic_examples(base, sc, phase, expected):
    got = lapse.total_rate(base, CR, CR + GAPS, sc, phase, "academy")
    np.testing.assert_allclose(got, expected, atol=5e-5)


@pytest.mark.parametrize("base,sc,expected", [
    (0.025, 0.05, [0.0508, 0.1176, 0.2399, 0.4277]),
    (0.350, 0.00, [0.3844, 0.4735, 0.6000, 0.6000]),
    (0.100, 0.00, [0.1344, 0.2235, 0.3865, 0.6000]),
])
def test_phbdg_reproduces_naic_examples(base, sc, expected):
    got = lapse.total_rate(base, CR, CR + GAPS, sc, 0, "phbdg")
    np.testing.assert_allclose(got, expected, atol=5e-5)


def test_credited_ahead_of_market_lowers_surrenders_within_floor():
    rf = lapse.academy_rate_factor(0.05, 0.03, 0.0, 2)
    assert rf == pytest.approx(-0.02)        # -3 x 2% = -6%, floored at -2%


def test_inside_buffer_no_adjustment():
    assert lapse.academy_rate_factor(0.0480, 0.0500, 0.0, 2) == 0.0


def test_base_table_positions():
    # 7-year charge: years 1-3 are "5+ years to expiry", year 7 is "1 year to expiry",
    # year 8 is the shock year, year 13+ is "5+ years after"
    rows = lapse.table_row(np.arange(1, 15))
    np.testing.assert_array_equal(rows, [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    assert lapse.base_rate(8, 65) == pytest.approx(0.415)
    assert lapse.base_rate(2, 85) == pytest.approx(0.020)
