"""Black-76 and Bachelier swaption formulas, and implied volatility.

Carried over from the author's fixed income coursework, where caps and swaptions were
priced under Black-76 from a quoted volatility. Here the direction is reversed: the
swaption is priced inside the Hull-White model (Jamshidian), and these formulas turn
that price back into the two volatilities a rates desk would quote it in. Black-76 is
lognormal in the forward swap rate; Bachelier (normal) vol, in basis points a year,
is the convention the USD swaption market actually trades.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76(forward: float, strike: float, expiry: float, vol: float, annuity: float,
            kind: str = "payer") -> float:
    if vol <= 0 or expiry <= 0:
        intrinsic = forward - strike if kind == "payer" else strike - forward
        return annuity * max(intrinsic, 0.0)
    sd = vol * np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if kind == "payer":
        return annuity * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return annuity * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def bachelier(forward: float, strike: float, expiry: float, vol: float, annuity: float,
              kind: str = "payer") -> float:
    if vol <= 0 or expiry <= 0:
        intrinsic = forward - strike if kind == "payer" else strike - forward
        return annuity * max(intrinsic, 0.0)
    sd = vol * np.sqrt(expiry)
    d = (forward - strike) / sd
    sign = 1.0 if kind == "payer" else -1.0
    return annuity * (sign * (forward - strike) * norm.cdf(sign * d) + sd * norm.pdf(d))


def implied_vol(price: float, forward: float, strike: float, expiry: float,
                annuity: float, model: str = "black", kind: str = "payer") -> float:
    f = black76 if model == "black" else bachelier
    hi = 5.0 if model == "black" else 0.10
    return float(brentq(lambda v: f(forward, strike, expiry, v, annuity, kind) - price,
                        1e-9, hi, xtol=1e-14))
