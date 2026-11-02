"""Hull-White one-factor, fitted exactly to today's curve.

    dr = (theta(t) - a r) dt + sigma dW

written as r(t) = x(t) + alpha(t), with x a zero-mean Ornstein-Uhlenbeck process that
carries all the randomness and alpha(t) a deterministic shift that reprices the curve.

Two choices make the model exact rather than approximately right:

1. **x and its time integral are simulated jointly and exactly.** Given x(t), the pair
   (x(t+h), integral of x over [t, t+h]) is bivariate normal with known moments, so the
   path discount factor D(t) = exp(-integral of r) is drawn from its true distribution
   at any step size. There is no Euler error and no trapezoid error; the only error in
   E[D(t)] is sampling noise. (The behavioural-cashflow-engine version integrated x by
   trapezoid and carried a 0.002 bp bias at monthly steps; this removes it.)

2. **Bond prices are written in x, not in r.** The textbook formula needs the
   instantaneous forward f(0,t), which is a numerical derivative of an interpolated
   curve. Substituting r = x + alpha, the forward cancels and what is left is

       P(t,T) = P(0,T)/P(0,t) * exp(-B x(t) - (sigma^2/4a)(1 - e^{-2at}) B^2
                                     - (sigma^2/2a^2)(1 - e^{-at})^2 B),   B = B(T-t)

   so every path reprices off P(0,.) directly. The last term is the part of alpha that
   does not cancel, and it is easy to drop: without it E[D(t) P(t,T)] misses P(0,T) by
   exp(B * Cov(integral of x, x(t))), which in an early version put the 7-year yield on
   the paths 25 bp too low at year 10 and 88 bp too low at year 20. tests/test_hull_white.py now checks the
   identity by simulation.

With sigma = 0, x is identically zero and D(t) = P(0,t) on every path. That is the
first test in tests/test_hull_white.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from .curve import ZeroCurve

_A_SMALL = 1e-8


def b_factor(a: float, tau) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    return tau if abs(a) < _A_SMALL else (1.0 - np.exp(-a * tau)) / a


@dataclass(frozen=True)
class Paths:
    """Simulated state on a time grid.

    times  (n_steps + 1,)        years from the valuation date, starting at 0
    x      (n_paths, n_steps+1)  the OU factor
    D      (n_paths, n_steps+1)  money-market deflator exp(-integral of r)
    """

    times: np.ndarray
    x: np.ndarray
    D: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return self.x.shape[0]

    def martingale_adjusted(self, curve: ZeroCurve) -> "Paths":
        """Rescale D(t) so the path average reprices every zero bond on `curve` exactly.

        The simulation is exact, so E[D(t)] = P(0,t) already; what this removes is the
        sampling noise in a 1,000 path set, which at a = 1.5% reaches one to two
        percent of the 30-year discount factor and would otherwise be read as option
        value. It is the standard last step of an economic scenario generator, and the
        factor is common to all paths, so it shifts no path relative to another.
        """
        target = curve.df(self.times)
        target[0] = 1.0
        scale = target / self.D.mean(axis=0)
        return Paths(self.times, self.x, self.D * scale[None, :],
                     meta={**self.meta, "martingale_adjusted": True})


@dataclass(frozen=True)
class HullWhite:
    curve: ZeroCurve
    a: float
    sigma: float

    def __post_init__(self) -> None:
        if self.a < 0 or self.sigma < 0:
            raise ValueError("a and sigma must be non-negative")

    # -- moments --------------------------------------------------------------

    def var_x(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if abs(self.a) < _A_SMALL:
            return self.sigma ** 2 * t
        return self.sigma ** 2 * (1 - np.exp(-2 * self.a * t)) / (2 * self.a)

    def var_int(self, t) -> np.ndarray:
        """Var of the integral of x over [0, t]; half of it is the convexity term."""
        t = np.asarray(t, dtype=float)
        a, s2 = self.a, self.sigma ** 2
        if abs(a) < _A_SMALL:
            return s2 * t ** 3 / 3.0
        return (s2 / a ** 2) * (t - 2 * b_factor(a, t) + (1 - np.exp(-2 * a * t)) / (2 * a))

    def zero_yield_vol(self, tau) -> np.ndarray:
        """Instantaneous normal vol of the tau-year zero yield: sigma * B(tau) / tau."""
        tau = np.asarray(tau, dtype=float)
        return self.sigma * b_factor(self.a, tau) / tau

    # -- bond prices on a path ------------------------------------------------

    def zero_bond(self, t: float, T, x_t) -> np.ndarray:
        """P(t,T) given x(t). Broadcasts over T (array) and x_t (array)."""
        T = np.asarray(T, dtype=float)
        x_t = np.asarray(x_t, dtype=float)
        if np.any(T < t - 1e-12):
            raise ValueError("T must be at or after t")
        b = b_factor(self.a, T - t)
        p0t = 1.0 if t <= 0 else float(self.curve.df(t)[0])
        p0T = self.curve.df(np.atleast_1d(T)).reshape(T.shape)
        s2 = self.sigma ** 2
        if abs(self.a) < _A_SMALL:
            conv = 0.5 * s2 * t * b ** 2 + 0.5 * s2 * t ** 2 * b
        else:
            a = self.a
            conv = ((s2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * b ** 2
                    + (s2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2 * b)
        return (p0T / p0t) * np.exp(-b * x_t - conv)

    def par_yield(self, t: float, tenor: float, x_t, m: int = 2) -> np.ndarray:
        """Par coupon of a `tenor`-year bond seen at t, for each x_t."""
        x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
        Ts = t + np.arange(1, int(round(tenor * m)) + 1) / m
        dfs = self.zero_bond(t, Ts[None, :], x_t[:, None])        # (n, k)
        return m * (1.0 - dfs[:, -1]) / dfs.sum(axis=1)

    # -- simulation -------------------------------------------------------------

    def simulate(self, times, n_paths: int, seed: int = 0, antithetic: bool = True,
                 z=None) -> Paths:
        """Exact joint simulation of (x, integral of x) on an arbitrary grid.

        `z` (n_paths, n_steps, 2) lets callers reuse the same draws across curve
        shifts, which is what makes bumped revaluations smooth.
        """
        times = np.asarray(times, dtype=float)
        if times[0] != 0.0 or (np.diff(times) <= 0).any():
            raise ValueError("times must start at 0 and increase")
        n_steps = len(times) - 1
        if z is None:
            z = standard_normals(n_paths, n_steps, seed, antithetic)
        x = np.zeros((n_paths, n_steps + 1))
        integral = np.zeros((n_paths, n_steps + 1))
        a, s = self.a, self.sigma
        for k in range(n_steps):
            h = times[k + 1] - times[k]
            if abs(a) < _A_SMALL:
                decay, bh = 1.0, h
                v1, v2, c12 = s * s * h, s * s * h ** 3 / 3, s * s * h * h / 2
            else:
                decay, bh = np.exp(-a * h), (1 - np.exp(-a * h)) / a
                v1 = s * s * (1 - np.exp(-2 * a * h)) / (2 * a)
                v2 = (s * s / a ** 2) * (h - 2 * bh + (1 - np.exp(-2 * a * h)) / (2 * a))
                c12 = (s * s / (2 * a ** 2)) * (1 - np.exp(-a * h)) ** 2
            if v1 > 0:
                e1 = np.sqrt(v1) * z[:, k, 0]
                e2 = (c12 / np.sqrt(v1)) * z[:, k, 0] + np.sqrt(max(v2 - c12 ** 2 / v1, 0.0)) * z[:, k, 1]
            else:
                e1 = e2 = 0.0
            integral[:, k + 1] = integral[:, k] + x[:, k] * bh + e2
            x[:, k + 1] = x[:, k] * decay + e1
        D = self.curve.df(times)[None, :] * np.exp(-integral - 0.5 * self.var_int(times)[None, :])
        D[:, 0] = 1.0
        return Paths(times, x, D, meta={"a": a, "sigma": s, "seed": seed})

    # -- swaptions (Jamshidian) -------------------------------------------------

    def zbo(self, T: float, S: float, K, kind: str) -> np.ndarray:
        """Option at T on the zero bond maturing at S, strike K (t=0 value)."""
        K = np.asarray(K, dtype=float)
        S = np.asarray(S, dtype=float)
        p_T = float(self.curve.df(T)[0])
        p_S = self.curve.df(np.atleast_1d(S)).reshape(S.shape)
        sp = np.sqrt(self.var_x(T)) * b_factor(self.a, S - T)
        if np.all(sp <= 0):
            fwd = p_S - K * p_T
            return np.maximum(fwd, 0.0) if kind == "call" else np.maximum(-fwd, 0.0)
        h = np.log(p_S / (p_T * K)) / sp + sp / 2
        if kind == "call":
            return p_S * norm.cdf(h) - K * p_T * norm.cdf(h - sp)
        return K * p_T * norm.cdf(-h + sp) - p_S * norm.cdf(-h)

    def swaption(self, expiry: float, tenor: float, strike: float, kind: str = "payer",
                 m: int = 2) -> float:
        """European swaption per unit notional, fixed leg paid `m` times a year.

        Jamshidian: a payer swaption is a put on the coupon bond paying K/m per period
        plus 1 at the end, struck at par. Find the x* that prices that bond at par at
        expiry; the option then splits into a strip of zero-bond puts.
        """
        Ts = expiry + np.arange(1, int(round(tenor * m)) + 1) / m
        c = np.full(len(Ts), strike / m)
        c[-1] += 1.0

        def bond_minus_par(xs):
            return float(np.sum(c * self.zero_bond(expiry, Ts, xs))) - 1.0

        if self.sigma == 0:
            fwd = (float(self.curve.df(expiry)[0]) - np.sum(c * self.curve.df(Ts)))
            return max(fwd, 0.0) if kind == "payer" else max(-fwd, 0.0)
        x_star = brentq(bond_minus_par, -2.0, 2.0, xtol=1e-14)
        X = self.zero_bond(expiry, Ts, x_star)
        leg = "put" if kind == "payer" else "call"
        return float(np.sum(c * self.zbo(expiry, Ts, X, leg)))

    def forward_swap(self, expiry: float, tenor: float, m: int = 2) -> tuple[float, float]:
        """(forward par swap rate, annuity) at t = 0."""
        Ts = expiry + np.arange(1, int(round(tenor * m)) + 1) / m
        ann = float(np.sum(self.curve.df(Ts))) / m
        rate = (float(self.curve.df(expiry)[0]) - float(self.curve.df(Ts[-1])[0])) / ann
        return rate, ann

    def swap_on_path(self, t: float, expiry: float, tenor: float, x_t, m: int = 2):
        """(swap rate, annuity) at time t for the swap starting at `expiry`, per path."""
        x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
        Ts = expiry + np.arange(1, int(round(tenor * m)) + 1) / m
        p_start = self.zero_bond(t, np.array([expiry]), x_t[:, None])[:, 0]
        dfs = self.zero_bond(t, Ts[None, :], x_t[:, None])
        ann = dfs.sum(axis=1) / m
        return (p_start - dfs[:, -1]) / ann, ann

    def with_curve(self, curve: ZeroCurve) -> "HullWhite":
        return HullWhite(curve, self.a, self.sigma)

    def with_sigma(self, sigma: float) -> "HullWhite":
        return HullWhite(self.curve, self.a, sigma)


def standard_normals(n_paths: int, n_steps: int, seed: int, antithetic: bool = True):
    """(n_paths, n_steps, 2) draws; antithetic pairs cancel the first-order noise."""
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps, 2))
        return np.concatenate([z, -z], axis=0)[:n_paths]
    return rng.standard_normal((n_paths, n_steps, 2))


def fit_to_vol_term_structure(tenors, vols, weights=None) -> tuple[float, float, np.ndarray]:
    """Fit (a, sigma) so that sigma B(tau)/tau matches a term structure of yield vols.

    In one-factor Hull-White every zero yield loads on the same factor with loading
    B(tau)/tau, so the term structure of yield volatilities pins down both parameters:
    sigma sets the level, a sets how fast vol decays with maturity. sigma is the
    diffusion coefficient, which is the same under the real-world and risk-neutral
    measures; that is what makes realized volatility a legitimate anchor for it.
    """
    tau = np.asarray(tenors, dtype=float)
    v = np.asarray(vols, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)

    def resid(p):
        a, s = p
        return w * (s * b_factor(a, tau) / tau - v)

    sol = least_squares(resid, x0=[0.05, float(v.max())], bounds=([0.0, 0.0], [2.0, 0.1]))
    a, s = sol.x
    return float(a), float(s), s * b_factor(a, tau) / tau
