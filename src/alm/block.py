"""The in-force block: 10,000 single premium deferred annuities, and its model points.

The block is synthetic (seeded) because no public SPDA seriatim file exists. What is
not invented is taken from public sources:

* sex mix: 55% female, inside the 52-57% the 2023-2024 SOA/LIMRA fixed rate deferred
  surrender study reports by market type;
* minimum guaranteed rate by issue year: the NAIC Standard Nonforfeiture Law rate,
  the 5-year CMT averaged over the issue year less 125 bp, rounded to 0.05%, floored
  at 1% and capped at 3% (the common product design is to guarantee exactly this).
  2023-2026 issues carry 2.70%-3.00%, so the guarantee is close to the money today;
* surrender charges 7, 6, 5, 4, 3, 2, 1% over policy years 1-7.

Policies were issued evenly over the ten years 2017-2026 and the valuation date is
treated as every policy's anniversary, so durations are whole years.

Policies with the same attained age, sex and duration project identically per dollar
of account value, so grouping on those three keys is exact compression, not an
approximation: 10,000 policies collapse to a few hundred cells with no loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VALUATION_DATE = pd.Timestamp("2026-10-30")
VALUATION_YEAR = 2026
ISSUE_YEARS = np.arange(2017, 2027)
SC_SCHEDULE = (0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)


def snfl_rate(cmt5_annual_avg: float) -> float:
    """NAIC Model 805 minimum nonforfeiture rate from the 5-year CMT."""
    r = round((cmt5_annual_avg - 0.0125) / 0.0005) * 0.0005
    return float(np.clip(r, 0.01, 0.03))


def gmir_by_issue_year(par: pd.DataFrame) -> pd.Series:
    """Minimum guaranteed rate for each issue year from the 5-year par yield history."""
    cmt5 = par["5Yr"].loc[:VALUATION_DATE]
    avg = cmt5.groupby(cmt5.index.year).mean()
    return pd.Series({y: snfl_rate(avg.loc[y]) for y in ISSUE_YEARS}, name="gmir")


def generate_inforce(par: pd.DataFrame, n: int = 10_000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    issue_year = rng.choice(ISSUE_YEARS, size=n)
    duration = VALUATION_YEAR - issue_year                 # completed policy years
    issue_age = np.clip(np.rint(rng.normal(62, 8, size=n)), 45, 80).astype(int)
    female = rng.random(n) < 0.55
    # account value: lognormal, median $90k; mean about $115k
    av = np.round(np.exp(rng.normal(np.log(90_000), 0.70, size=n)), 2)
    gmir = gmir_by_issue_year(par).reindex(issue_year).to_numpy()
    return pd.DataFrame({
        "policy_id": np.arange(1, n + 1), "issue_year": issue_year, "duration": duration,
        "age": issue_age + duration, "female": female, "av": av, "gmir": gmir,
    })


@dataclass(frozen=True)
class ModelPoints:
    age: np.ndarray        # attained age at valuation, ANB
    female: np.ndarray
    duration: np.ndarray   # completed policy years at valuation
    av: np.ndarray         # total account value in the cell
    count: np.ndarray
    gmir: np.ndarray

    @property
    def n_cells(self) -> int:
        return len(self.av)

    @property
    def total_av(self) -> float:
        return float(self.av.sum())


def compress(inforce: pd.DataFrame) -> ModelPoints:
    g = (inforce.groupby(["age", "female", "duration"], as_index=False)
         .agg(av=("av", "sum"), count=("policy_id", "size"), gmir=("gmir", "first")))
    return ModelPoints(g["age"].to_numpy(), g["female"].to_numpy(bool),
                       g["duration"].to_numpy(), g["av"].to_numpy(float),
                       g["count"].to_numpy(), g["gmir"].to_numpy(float))
