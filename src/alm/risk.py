"""Risk measures: effective duration and convexity, VaR and ES of surplus."""
from __future__ import annotations

import numpy as np


def duration_convexity(v_down: float, v0: float, v_up: float, h: float) -> tuple[float, float]:
    """From values at -h, 0, +h (decimal shift of the zero curve)."""
    return (v_down - v_up) / (2 * v0 * h), (v_down + v_up - 2 * v0) / (v0 * h * h)


def var_es(pvs: np.ndarray, q: float = 0.995) -> tuple[float, float]:
    """VaR and ES of the shortfall of surplus below its mean, at level q.

    Loss on a path = mean surplus - surplus on that path, so VaR is the unexpected loss
    and does not reward a hedge for moving the mean. ES is the average loss at or beyond
    VaR. With 1,000 paths the 99.5% tail is five paths; bootstrap_var_es says how much
    that matters.
    """
    loss = pvs.mean() - pvs
    v = float(np.quantile(loss, q))
    return v, float(loss[loss >= v].mean())


def bootstrap_var_es(pvs: np.ndarray, q: float = 0.995, n_boot: int = 2000,
                     seed: int = 0, ci: float = 0.90) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pvs)
    vs, es = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        vs[i], es[i] = var_es(pvs[rng.integers(0, n, n)], q)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {"var_lo": float(np.quantile(vs, lo)), "var_hi": float(np.quantile(vs, hi)),
            "es_lo": float(np.quantile(es, lo)), "es_hi": float(np.quantile(es, hi))}
