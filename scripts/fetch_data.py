"""Rebuild data/raw/ from the public sources. Not part of run_all.

run_all.py reads the committed snapshots so that every number in the README is
reproducible. Run this only to refresh them, and expect the numbers to move.

    python scripts/fetch_data.py

Sources
-------
Treasury par yields   U.S. Treasury, Daily Treasury Par Yield Curve Rates (public domain)
                      https://home.treasury.gov/resource-center/data-chart-center/interest-rates
2012 IAM Basic        Society of Actuaries, mort.soa.org tables 2581 (male) and 2582 (female), ANB
Projection Scale G2   Society of Actuaries, mort.soa.org tables 2583 (male) and 2584 (female), ANB
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FIRST_YEAR, LAST_YEAR = 2017, 2026

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={y}&page&_format=csv"
)
SOA_URL = "https://mort.soa.org/data/t{tid}.xml"

#: Treasury column label -> tenor in years. The eleven tenors with a full history
#: since 2017 (2M, 4M and 1.5M were added later and are dropped).
TENORS = {
    "1 Mo": 1 / 12, "3 Mo": 0.25, "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
    "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def fetch_treasury() -> pd.DataFrame:
    frames = []
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        df = pd.read_csv(io.BytesIO(_get(TREASURY_URL.format(y=y))))
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        frames.append(df[["Date", *TENORS]])
        print(f"  treasury {y}: {len(df)} days")
    out = pd.concat(frames).sort_values("Date").drop_duplicates("Date")
    out.columns = ["date", *[k.replace(" ", "") for k in TENORS]]
    return out


def _xtbml_rates(tid: int) -> pd.Series:
    text = _get(SOA_URL.format(tid=tid)).decode("utf-8-sig")
    pairs = re.findall(r'<Y t="(\d+)">([^<]+)</Y>', text)
    if not pairs:
        raise ValueError(f"no rates parsed from SOA table {tid}")
    return pd.Series({int(a): float(v) for a, v in pairs}, name=tid)


def fetch_mortality() -> pd.DataFrame:
    q_m, q_f, g_m, g_f = (_xtbml_rates(t) for t in (2581, 2582, 2583, 2584))
    out = pd.DataFrame({"q_male": q_m, "q_female": q_f})
    out["g2_male"] = g_m.reindex(out.index)
    out["g2_female"] = g_f.reindex(out.index)
    # Scale G2 is published to age 105; the SOA convention holds it flat beyond.
    out[["g2_male", "g2_female"]] = out[["g2_male", "g2_female"]].ffill()
    out.index.name = "age"
    return out


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    print("Treasury par yields")
    fetch_treasury().to_csv(RAW / "treasury_par_yields.csv", index=False)
    print("SOA 2012 IAM Basic + Scale G2")
    fetch_mortality().to_csv(RAW / "soa_2012iam_basic_g2.csv")
    print(f"written to {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
