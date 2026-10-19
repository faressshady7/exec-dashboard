"""Robustness check for the Step 5 thresholds (thesis appendix).

The thresholds in settings.yaml `method` are the researcher's choices. This re-runs the
sensitivity model with nearby values, one change at a time, and records whether any result
changes. Output: gold.robustness (one row per variant x company x factor) and, per company x
factor, `robust_pct` = share of variants that give the same result (same method, impact within 5%).

Run:  python -m dashboard.robustness        (compute also runs it and stores the table)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.config import Settings, load_settings

# one change at a time, around the chosen values
VARIANTS = [
    ("fuel_min_r2", 0.4), ("fuel_min_r2", 0.6),
    ("rate_min_r2", 0.2), ("rate_min_r2", 0.4),
    ("exclude_covid", False),
]
SIGMAS = (1.5, 2.5)          # 'unusual' flag alternatives (counted separately: no $ impact)
TOLERANCE = 0.05             # impact within 5% of the base = "same result"


def _key(df: pd.DataFrame) -> pd.DataFrame:
    unit = np.where(df["factor"] == "jet_fuel", 10, 100)          # +10% fuel, +100bp rates
    return df.assign(impact=df["impact_per_unit_q"] * unit)[["ticker", "factor", "method", "impact"]]


def _same(a: pd.Series, b: pd.Series, ma: pd.Series, mb: pd.Series) -> pd.Series:
    both_nan = a.isna() & b.isna()
    close = (a - b).abs() <= TOLERANCE * b.abs()
    same_method = ma.str.split(" ").str[0] == mb.str.split(" ").str[0]   # regression / proportional / not
    return (both_nan | close.fillna(False)) & same_method


def robustness(fin: pd.DataFrame, mkt: pd.DataFrame, s: Settings,
               base: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from dashboard.compute import compute_kpis, sensitivities
    base = _key(base if base is not None else sensitivities(fin, mkt, s))
    rows = []
    for param, value in VARIANTS:
        alt_s = s.model_copy(update={"method": s.method.model_copy(update={param: value})})
        alt = _key(sensitivities(fin, mkt, alt_s))
        m = base.merge(alt, on=["ticker", "factor"], suffixes=("_base", "_alt"))
        m["same_result"] = _same(m["impact_alt"], m["impact_base"], m["method_alt"], m["method_base"])
        m.insert(0, "parameter", param)
        m.insert(1, "chosen", str(getattr(s.method, param)))
        m.insert(2, "alternative", str(value))
        rows.append(m)
    rob = pd.concat(rows, ignore_index=True)

    # 'unusual' flags in each company's latest quarter under other sigma values
    def latest_flags(meth):
        k = compute_kpis(fin, meth)
        last = k[k["period_end"] == k.groupby("ticker")["period_end"].transform("max")]
        return last.groupby("ticker")["unusual"].sum()
    flags_base = latest_flags(s.method)
    for sigma in SIGMAS:
        alt = latest_flags(s.method.model_copy(update={"unusual_sigma": sigma}))
        for t in flags_base.index:
            rob = pd.concat([rob, pd.DataFrame([{
                "parameter": "unusual_sigma", "chosen": str(s.method.unusual_sigma), "alternative": str(sigma),
                "ticker": t, "factor": "unusual_flags_latest_quarter",
                "method_base": "count", "impact_base": float(flags_base[t]),
                "method_alt": "count", "impact_alt": float(alt.get(t, 0)),
                "same_result": bool(flags_base[t] == alt.get(t, 0))}])], ignore_index=True)

    money = rob[rob["factor"].isin(["jet_fuel", "fed_funds"])]
    share = (money.groupby(["ticker", "factor"])["same_result"].mean() * 100).rename("robust_pct").reset_index()
    return rob, share


def main() -> None:
    from dashboard.compute import load_financials, load_market
    from dashboard.db import engine
    s = load_settings()
    eng = engine()
    rob, share = robustness(load_financials(eng), load_market(eng), s)
    changed = rob[~rob["same_result"]]
    print(f"ROBUSTNESS: {len(rob)} checks, {len(changed)} changed a result "
          f"({100 * (1 - len(changed) / len(rob)):.0f}% unchanged)")
    for r in share.itertuples():
        print(f"  {r.ticker:<5}{r.factor:<10} same result in {r.robust_pct:.0f}% of alternative settings")
    if len(changed):
        print("\nWHAT CHANGED (report these in the thesis):")
        for r in changed.itertuples():
            print(f"  {r.parameter} {r.chosen} -> {r.alternative}: {r.ticker} {r.factor}: "
                  f"{r.method_base.split(' ')[0]} {r.impact_base:,.0f} -> {r.method_alt.split(' ')[0]} {r.impact_alt:,.0f}")


if __name__ == "__main__":
    main()
