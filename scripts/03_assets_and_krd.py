"""Step 3: the asset portfolios and key rate durations against the liability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import assets as A
from alm.context import KEY_RATES, KRD_BUMP, build, load, save
from alm.curve import key_rate_bump


def portfolios(ctx):
    """Static match (the brief's t = 0 deterministic duration) and effective match."""
    dur = load("02_duration_convexity.csv").set_index("variant").loc["dynamic"]
    static_pv, static_dur = ctx.static_duration()
    return {
        "static_match": (ctx.barbell(static_pv * static_dur), static_pv * static_dur),
        "effective_match": (ctx.barbell(dur.stochastic_value * dur.effective_duration),
                            dur.stochastic_value * dur.effective_duration),
    }


def main() -> None:
    ctx = build()
    ports = portfolios(ctx)
    rows = []
    for name, (bonds, target) in ports.items():
        for b in bonds:
            rows.append({"portfolio": name, "maturity": b.maturity, "coupon": b.coupon,
                         "face": b.face, "spread": b.spread,
                         "market_value": A.value([b], ctx.curve),
                         "duration": A.duration([b], ctx.curve)})
        rows.append({"portfolio": name, "maturity": np.nan, "coupon": np.nan,
                     "face": sum(b.face for b in bonds), "spread": np.nan,
                     "market_value": A.value(bonds, ctx.curve),
                     "duration": A.duration(bonds, ctx.curve),
                     "target_dollar_duration": target})
    pf = pd.DataFrame(rows)
    save(pf, "03_asset_portfolios.csv")
    print(pf.round(4).to_string(index=False))

    # key rate durations: triangular bumps on the zero curve, same draws everywhere
    l0 = load("02_duration_convexity.csv").set_index("variant").loc["dynamic", "stochastic_value"]
    rows = []
    for k, key in enumerate(KEY_RATES):
        up = ctx.curve.bumped(key_rate_bump(KEY_RATES, k, KRD_BUMP))
        dn = ctx.curve.bumped(key_rate_bump(KEY_RATES, k, -KRD_BUMP))
        l_up = ctx.liability(up)[2].value
        l_dn = ctx.liability(dn)[2].value
        row = {"key_rate": key, "liability_krd": (l_dn - l_up) / (2 * l0 * KRD_BUMP),
               "liability_dv01": (l_dn - l_up) / (2 * KRD_BUMP) * 1e-4}
        for name, (bonds, _) in ports.items():
            a0 = A.value(bonds, ctx.curve)
            a_up, a_dn = A.value(bonds, up), A.value(bonds, dn)
            row[f"{name}_krd"] = (a_dn - a_up) / (2 * a0 * KRD_BUMP)
            row[f"{name}_dv01"] = (a_dn - a_up) / (2 * KRD_BUMP) * 1e-4
        rows.append(row)
        print(f"key {key:>4}: liability KRD {row['liability_krd']:.3f}")
    krd = pd.DataFrame(rows)
    total = krd.drop(columns="key_rate").sum().to_dict()
    total["key_rate"] = "sum"
    krd = pd.concat([krd, pd.DataFrame([total])], ignore_index=True)
    save(krd, "03_key_rate_durations.csv")
    print(krd.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
