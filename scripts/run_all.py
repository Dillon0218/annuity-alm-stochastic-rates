"""Rerun the whole pipeline from the committed data: python scripts/run_all.py"""
from __future__ import annotations

import runpy
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_curve_and_rates.py",
    "02_liability.py",
    "03_assets_and_krd.py",
    "04_surplus.py",
    "05_hedge.py",
    "06_stress.py",
    "07_sensitivity.py",
    "08_figures.py",
    "09_headline.py",
]


def main() -> None:
    start = time.time()
    for step in STEPS:
        t0 = time.time()
        print(f"\n=== {step} ===")
        runpy.run_path(str(HERE / step), run_name="__main__")
        print(f"--- {step} done in {time.time() - t0:.0f}s")
    print(f"\nall steps done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
