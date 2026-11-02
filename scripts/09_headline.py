"""Step 9: every number quoted in the README, in one table, read from steps 1-7."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm.context import build, load, save


def main() -> None:
    ctx = build()
    av0 = ctx.av0
    hwp = load("01_hw_parameters.csv").iloc[0]
    curve = load("01_curve.csv").set_index("tenor")
    mt = load("01_martingale_test.csv")
    sw = load("01_swaption_check.csv")
    dur = load("02_duration_convexity.csv").set_index("variant")
    prof = load("04_surplus_vs_shift.csv").set_index("shift_bp")
    ve = load("04_var_es.csv").set_index(["portfolio", "liability"])
    hs = load("05_hedge_summary.csv").set_index("portfolio")
    st = load("06_stress.csv").set_index("scenario")
    krd = load("03_key_rate_durations.csv")
    krd = krd[krd["key_rate"] != "sum"].astype(float).set_index("key_rate")
    pf = load("03_asset_portfolios.csv")
    static_legs = pf[(pf["portfolio"] == "static_match") & pf["maturity"].notna()]
    sens = load("07_sensitivity.csv").set_index("variant")
    neg = load("04_negative_rates.csv").iloc[0]

    d, n = dur.loc["dynamic"], dur.loc["no_dynamic"]
    sm, sh, em = "static_match_surplus", "static_match_plus_payers_surplus", "effective_match_surplus"
    rows = {
        "valuation date": "2026-10-30",
        "7y Treasury par yield": curve.loc["7Yr", "par_yield"],
        "HW mean reversion a": hwp.a,
        "HW sigma": hwp.sigma,
        "martingale test max |z| (1,000 raw paths)": mt["z"].abs().max(),
        "swaption Jamshidian vs MC max |z|": sw["z"].abs().max(),
        "policies": len(ctx.inforce),
        "account value": av0,
        "model points": ctx.mp.n_cells,
        "liability value, stochastic": d.stochastic_value,
        "liability value, % of AV": d.stochastic_pct_av,
        "liability value, deterministic": d.deterministic_value,
        "dynamic surrender cost": d.stochastic_value - n.stochastic_value,
        "dynamic surrender cost, % of AV": (d.stochastic_value - n.stochastic_value) / av0,
        "static cash flow duration": d.static_duration,
        "effective duration": d.effective_duration,
        "effective duration, no dynamic surrender": n.effective_duration,
        "deterministic effective duration": d.deterministic_effective_duration,
        "effective convexity": d.effective_convexity,
        "effective convexity, no dynamic surrender": n.effective_convexity,
        "duration at -200 bp": load("02_value_vs_shift.csv").set_index("shift_bp").loc[-200, "dynamic_duration"],
        "duration at +200 bp": load("02_value_vs_shift.csv").set_index("shift_bp").loc[200, "dynamic_duration"],
        "duration at -200 bp, no dynamic": load("02_value_vs_shift.csv").set_index("shift_bp").loc[-200, "no_dynamic_duration"],
        "duration at +200 bp, no dynamic": load("02_value_vs_shift.csv").set_index("shift_bp").loc[200, "no_dynamic_duration"],
        "static match 2y bond": static_legs.iloc[0].market_value,
        "static match 20y bond": static_legs.iloc[1].market_value,
        "static match asset duration": pf[(pf.portfolio == "static_match") & pf.maturity.isna()].duration.iloc[0],
        "effective match asset duration": pf[(pf.portfolio == "effective_match") & pf.maturity.isna()].duration.iloc[0],
        "surplus at t=0": prof.loc[0, sm],
        "static match surplus change +200 bp": prof.loc[200, sm] - prof.loc[0, sm],
        "static match surplus change -200 bp": prof.loc[-200, sm] - prof.loc[0, sm],
        "static match surplus change +200 bp, no dynamic": (prof.loc[200, "static_match_surplus_no_dynamic"]
                                                            - prof.loc[0, "static_match_surplus_no_dynamic"]),
        "effective match surplus change +200 bp": prof.loc[200, em] - prof.loc[0, em],
        "effective match surplus change -200 bp": prof.loc[-200, em] - prof.loc[0, em],
        "hedged surplus change +200 bp": prof.loc[200, sh] - prof.loc[0, sh],
        "hedged surplus change -200 bp": prof.loc[-200, sh] - prof.loc[0, sh],
        "VaR static": ve.loc[("static_match", "dynamic"), "var_995"],
        "ES static": ve.loc[("static_match", "dynamic"), "es_995"],
        "ES static CI low": ve.loc[("static_match", "dynamic"), "es_lo"],
        "ES static CI high": ve.loc[("static_match", "dynamic"), "es_hi"],
        "VaR static, no dynamic": ve.loc[("static_match", "no_dynamic"), "var_995"],
        "ES static, no dynamic": ve.loc[("static_match", "no_dynamic"), "es_995"],
        "VaR effective": ve.loc[("effective_match", "dynamic"), "var_995"],
        "ES effective": ve.loc[("effective_match", "dynamic"), "es_995"],
        "ES effective CI low": ve.loc[("effective_match", "dynamic"), "es_lo"],
        "ES effective CI high": ve.loc[("effective_match", "dynamic"), "es_hi"],
        "worst 5 paths with rates up, static": ve.loc[("static_match", "dynamic"), "worst5_rates_up"],
        "worst 5 paths with rates up, effective": ve.loc[("effective_match", "dynamic"), "worst5_rates_up"],
        "VaR hedged": hs.loc["static_match_plus_payers", "var_995"],
        "ES hedged": hs.loc["static_match_plus_payers", "es_995"],
        "ES hedged CI low": hs.loc["static_match_plus_payers", "es_lo"],
        "ES hedged CI high": hs.loc["static_match_plus_payers", "es_hi"],
        "hedge notional": hs.loc["static_match_plus_payers", "notional"],
        "hedge notional, x AV": hs.loc["static_match_plus_payers", "notional"] / av0,
        "hedge premium": hs.loc["static_match_plus_payers", "premium"],
        "hedge premium, % of AV": hs.loc["static_match_plus_payers", "premium_pct_av"],
        "hedge strike": hs.loc["static_match_plus_payers", "strike"],
        "hedge normal vol (bp)": hs.loc["static_match_plus_payers", "normal_vol_bp"],
        "hedge Black vol": hs.loc["static_match_plus_payers", "black_vol"],
        "ES reduction from hedge": 1 - hs.loc["static_match_plus_payers", "es_995"] / hs.loc["static_match", "es_995"],
        "VaR reduction from hedge": 1 - hs.loc["static_match_plus_payers", "var_995"] / hs.loc["static_match", "var_995"],
        "2022 replay impact, static": st.loc["2022 replay", "static_match_impact"],
        "2022 replay impact, hedged": st.loc["2022 replay", "static_match_plus_payers_impact"],
        "2022 replay impact, effective": st.loc["2022 replay", "effective_match_impact"],
        "2022 replay surrenders yr1-3 vs base": st.loc["2022 replay", "surrenders_yr1_3_vs_base"],
        "2022 replay surrenders yr1-3": st.loc["2022 replay", "surrenders_yr1_3"],
        "2022 replay max rate gap (bp)": st.loc["2022 replay", "max_rate_gap_bp"],
        "static match +200 bp loss from dynamic surrender": (
            (prof.loc[200, sm] - prof.loc[0, sm])
            - (prof.loc[200, "static_match_surplus_no_dynamic"] - prof.loc[0, "static_match_surplus_no_dynamic"])),
        "base surrenders yr1-3": st.loc["base", "surrenders_yr1_3"],
        "+200 path impact, static": st.loc["parallel +200 bp", "static_match_impact"],
        "+200 path impact, hedged": st.loc["parallel +200 bp", "static_match_plus_payers_impact"],
        "-200 path impact, static": st.loc["parallel -200 bp", "static_match_impact"],
        "liability KRD 3-7y share": krd.loc[[3.0, 5.0, 7.0], "liability_krd"].sum() / krd["liability_krd"].sum(),
        "static assets KRD 20y share": krd.loc[20.0, "static_match_krd"] / krd["static_match_krd"].sum(),
        "sensitivity: ES, no dynamic": sens.loc["none", "static_match_es_995"],
        "sensitivity: ES, Academy x2": sens.loc["Academy x2", "static_match_es_995"],
        "sensitivity: duration, new money crediting": sens.loc["new money rate", "effective_duration"],
        "paths with a negative 3m rate within 10y": neg.share_negative_within_10y,
        "paths with a negative 3m rate within 30y": neg.share_negative_within_30y,
        "effective match worst 5 with negative rates within 10y": neg.effective_match_worst5_negative_within_10y,
        "effective match worst 5, AV at minimum rate in year 15, min": neg.effective_match_worst5_av_at_floor_y15_min,
        "effective match worst 5, AV at minimum rate in year 15, max": neg.effective_match_worst5_av_at_floor_y15_max,
        "30y discount factor sampling error, 1,000 raw paths (bp of price)": mt.set_index("maturity").loc[30, "stderr_bp_of_price"],
        "hedge alternatives: ES reduction per premium, 2y": load("05_hedge_alternatives.csv").iloc[0].es_reduction_per_premium,
        "hedge alternatives: ES reduction per premium, 5y": load("05_hedge_alternatives.csv").iloc[3].es_reduction_per_premium,
    }
    out = pd.DataFrame({"metric": list(rows), "value": list(rows.values())})
    save(out, "00_headline_numbers.csv")
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
