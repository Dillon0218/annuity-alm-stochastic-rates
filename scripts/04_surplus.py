"""Step 4: the surplus, by path and under instantaneous shocks.

Surplus on a path is the present value of asset cash flows less benefit cash flows over
the 30-year run-off, deflated at that path's own short rate. That is the same as
carrying any shortfall or excess in cash at the short rate and discounting the ending
balance, so it needs no reinvestment assumption: it measures what the t = 0 portfolio
does and does not protect. Averaged over paths it equals assets less the liability
value; its spread across paths is the asset-liability mismatch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm import assets as A
from alm import risk
from alm.context import build, load, save

_spec = importlib.util.spec_from_file_location(
    "step3", Path(__file__).with_name("03_assets_and_krd.py"))
step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step3)


def main() -> None:
    ctx = build()
    ports = step3.portfolios(ctx)
    hw, paths, r_dyn = ctx.liability()
    _, _, r_nod = ctx.liability(asm=ctx.asm.but(dynamic="none"))

    dist = pd.DataFrame({"path": np.arange(len(r_dyn.pv))})
    rows = []
    avg_x10 = paths.x[:, : int(10 / (ctx.times[1] - ctx.times[0])) + 1].mean(axis=1)
    for pname, (bonds, _) in ports.items():
        apv = A.pathwise_pv(bonds, paths)
        for lname, r in (("dynamic", r_dyn), ("no_dynamic", r_nod)):
            pvs = apv - r.pv
            dist[f"{pname}__{lname}"] = pvs
            var, es = risk.var_es(pvs)
            ci = risk.bootstrap_var_es(pvs)
            worst = np.argsort(pvs)[:5]
            rows.append({"portfolio": pname, "liability": lname, "mean": pvs.mean(),
                         "sd": pvs.std(ddof=1), "var_995": var, "es_995": es, **ci,
                         "worst5_avg_factor_0_10y": avg_x10[worst].mean(),
                         "worst5_rates_up": int((avg_x10[worst] > 0).sum())})
    dist["avg_factor_0_10y"] = avg_x10
    save(dist, "04_surplus_by_path.csv")

    # how often the Gaussian model takes the 3-month rate below zero
    short = np.column_stack([
        -np.log(hw.zero_bond(t, np.array([t + 0.25]), paths.x[:, k][:, None])[:, 0]) / 0.25
        for k, t in enumerate(ctx.times)])
    eff = dist["effective_match__dynamic"].to_numpy()
    worst_eff = np.argsort(eff)[:5]
    q10 = ctx.times <= 10 + 1e-9
    save(pd.DataFrame([{
        "share_negative_within_10y": float((short[:, q10].min(axis=1) < 0).mean()),
        "share_negative_within_30y": float((short.min(axis=1) < 0).mean()),
        "effective_match_worst5_negative_within_10y": int((short[worst_eff][:, q10].min(axis=1) < 0).sum()),
    }]), "04_negative_rates.csv")
    ve = pd.DataFrame(rows)
    save(ve, "04_var_es.csv")
    print(ve.round(1).to_string(index=False))

    # economic surplus under instantaneous shocks, from the step 2 liability values
    prof = load("02_value_vs_shift.csv")
    for pname, (bonds, _) in ports.items():
        prof[f"{pname}_assets"] = [A.value(bonds, ctx.curve, s / 1e4) for s in prof["shift_bp"]]
        prof[f"{pname}_surplus"] = prof[f"{pname}_assets"] - prof["dynamic"]
        prof[f"{pname}_surplus_no_dynamic"] = prof[f"{pname}_assets"] - prof["no_dynamic"]
    save(prof, "04_surplus_vs_shift.csv")
    show = prof.set_index("shift_bp")[[c for c in prof.columns if c.endswith("_surplus")]]
    print((show / 1e6).round(1).to_string())


if __name__ == "__main__":
    main()
