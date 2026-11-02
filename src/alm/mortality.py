"""SOA 2012 IAM Basic Table with generational improvement under Projection Scale G2.

    q_x(Y) = q_x^{2012} * (1 - G2_x)^(Y - 2012)

The 2012 IAM is a payout annuitant table, so it is lighter than deferred annuity
accumulation phase experience would be. That understates deaths slightly, which
matters little here: in a fixed deferred annuity the death benefit is the account
value, so a death is a surrender without the surrender charge.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IAM_CSV = ROOT / "data" / "raw" / "soa_2012iam_basic_g2.csv"
BASE_YEAR = 2012
MAX_AGE = 120


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(IAM_CSV).set_index("age")


def qx(age, female, calendar_year) -> np.ndarray:
    """Annual mortality rate, ANB, broadcast over the three inputs."""
    t = _table()
    age = np.clip(np.asarray(age, dtype=int), 0, MAX_AGE)
    female = np.asarray(female, dtype=bool)
    yrs = np.asarray(calendar_year, dtype=float) - BASE_YEAR
    q = np.where(female, t["q_female"].to_numpy()[age], t["q_male"].to_numpy()[age])
    g = np.where(female, t["g2_female"].to_numpy()[age], t["g2_male"].to_numpy()[age])
    return np.where(age >= MAX_AGE, 1.0, np.clip(q * (1.0 - g) ** yrs, 0.0, 1.0))
