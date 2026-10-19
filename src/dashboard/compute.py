"""Step 5 - Compute layer: every number the dashboard shows and the AI talks about.

"The system computes, the AI explains": nothing here uses an LLM. Every output is a
deterministic function of gold, so the AI layer can only REPORT these numbers, and the
number checker (Step 7) can verify every figure it writes against these tables.

Outputs (schema gold, rebuilt on every run)
  kpi_quarterly    one company x KPI x quarter      (margins, growth, fuel share, leverage, FCF ...)
  peer_gap         one KPI at the focal company's latest quarter vs each peer and the peer average
  market_snapshot  one market variable today: change over 1M / 3M / 1Y and position vs its history
  sensitivity      one company x risk factor: estimated $ impact per quarter of a 1-unit move
  scenario         one company x factor x shock: impact on quarterly operating income & margin

Method notes (thesis decisions)
  - YoY growth compares a quarter with the same quarter one year earlier (removes seasonality).
  - COVID quarters (Q2 2020 - Q4 2021) are excluded when ESTIMATING sensitivities and "unusual"
    thresholds: airline capacity collapsed, so those quarters describe a different business.
  - Fuel sensitivity: quarterly fuel expense regressed on the quarter's average jet fuel price.
    The slope is the implied gallons burned per quarter, so +10% fuel = slope x price x 10%.
    If the fit is weak (R^2 < fuel_min_r2, default 0.5) the simple proportional rule (fuel expense x shock) is used.
  - Rate sensitivity: quarterly interest expense regressed on the average fed funds rate. Airline
    debt is mostly fixed-rate, so a weak fit (R^2 < 0.3) is reported as "not modelled" instead of
    inventing a number (the 10-K Item 7A disclosure is the proper source - planned manual input).
  - Hedging and refining are not modelled separately: the regression uses REPORTED fuel expense,
    which already includes their effects. None of the three US carriers hedged fuel in 2026.
    Company `caveats` from settings.yaml (e.g. Delta's refinery) travel with every fuel row.

Run:  python -m dashboard.compute
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboard.config import Settings, load_settings

COVID = (pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31"))   # default; settings.yaml `method` overrides


def _outside_covid(dates: pd.Series, m) -> pd.Series:
    """True for dates that may be used for ESTIMATION (all dates if COVID is not excluded)."""
    if m is None or m.covid is None:
        return pd.Series(True, index=dates.index) if m is not None else ~dates.between(*COVID)
    return ~dates.between(pd.Timestamp(m.covid[0]), pd.Timestamp(m.covid[1]))
FUEL_SHOCKS = [-30, -20, -10, 0, 10, 20, 30]          # % change in jet fuel price
RATE_SHOCKS = [-100, -50, 0, 50, 100]                 # basis points change in fed funds
COMBINED = [(10, 100), (20, 100), (-10, -100), (-20, -100)]   # (fuel %, rates bp) moved together

# Shown under the simulator. Same framing as the airlines' own 10-K Item 7A sensitivities.
ASSUMPTIONS = [
    "Linear: the effect grows in proportion to the shock (twice the shock, twice the effect).",
    "No reaction: fares, capacity and hedging are assumed unchanged; in reality airlines respond, "
    "so large shocks are likely overstated.",
    "Combined shocks are added together; interactions between them are ignored.",
    "Rate changes affect interest cost, not operating income.",
]

# KPI catalogue: label, unit, which direction is better (for ranking and colour)
KPIS = {
    "revenue":              ("Revenue",                  "USD",   "higher"),
    "revenue_yoy_pct":      ("Revenue growth (YoY)",     "%",     "higher"),
    "operating_margin_pct": ("Operating margin",         "%",     "higher"),
    "net_margin_pct":       ("Net margin",               "%",     "higher"),
    "fuel_share_opex_pct":  ("Fuel share of costs",      "%",     "lower"),
    "fuel_cost_yoy_pct":    ("Fuel cost growth (YoY)",   "%",     "lower"),
    "interest_coverage_x":  ("Interest coverage",        "x",     "higher"),
    "free_cash_flow":       ("Free cash flow",           "USD",   "higher"),
    "net_debt":             ("Net debt",                 "USD",   "lower"),
    "debt_to_assets_pct":   ("Debt to assets",           "%",     "lower"),
    "cash":                 ("Cash",                     "USD",   "higher"),
}


# =========================================================================== load gold
def load_financials(eng: Engine) -> pd.DataFrame:
    q = text("""
        SELECT c.ticker, c.role, v.code, d.full_date AS period_end, d.quarter_label, f.value
        FROM gold.fact_financials f
        JOIN gold.dim_company c USING (company_key)
        JOIN gold.dim_variable v USING (variable_key)
        JOIN gold.dim_date d USING (date_key)
        WHERE f.period_type IN ('Q', 'I')""")
    with eng.connect() as con:
        df = pd.DataFrame(con.execute(q).mappings().all())
    df["period_end"] = pd.to_datetime(df["period_end"])
    wide = df.pivot_table(index=["ticker", "role", "period_end", "quarter_label"],
                          columns="code", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def load_market(eng: Engine) -> pd.DataFrame:
    q = text("""
        SELECT v.code, v.label, v.unit, v.category, d.full_date AS obs_date, m.value
        FROM gold.fact_market m
        JOIN gold.dim_variable v USING (variable_key)
        JOIN gold.dim_date d USING (date_key)""")
    with eng.connect() as con:
        df = pd.DataFrame(con.execute(q).mappings().all())
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df.sort_values(["code", "obs_date"]).reset_index(drop=True)


def quarterly_average(mkt: pd.DataFrame, code: str) -> pd.Series:
    """Average of a market variable per calendar quarter, indexed by quarter-end date."""
    s = mkt[mkt["code"] == code].set_index("obs_date")["value"]
    return s.resample("QE").mean()


# =========================================================================== KPIs
def _safe_div(a, b):
    return np.where((b != 0) & pd.notna(b) & pd.notna(a), a / b, np.nan)


def compute_kpis(fin: pd.DataFrame, m=None) -> pd.DataFrame:
    df = fin.copy()
    get = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)
    df["operating_margin_pct"] = _safe_div(get("operating_income"), get("revenue")) * 100
    df["net_margin_pct"] = _safe_div(get("net_income"), get("revenue")) * 100
    df["fuel_share_opex_pct"] = _safe_div(get("fuel_expense"), get("operating_expenses")) * 100
    interest = get("interest_expense").where(get("interest_expense") > 0)
    df["interest_coverage_x"] = _safe_div(get("operating_income"), interest)
    df["free_cash_flow"] = get("operating_cash_flow") - get("capex")
    df["net_debt"] = get("long_term_debt") - get("cash")
    df["debt_to_assets_pct"] = _safe_div(get("long_term_debt"), get("total_assets")) * 100

    # YoY: same quarter one year earlier (exact date match on the prior-year quarter end)
    prior = df[["ticker", "period_end", "revenue", "fuel_expense"]].copy()
    prior["period_end"] = prior["period_end"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={"revenue": "revenue_py", "fuel_expense": "fuel_py"})
    df = df.merge(prior, on=["ticker", "period_end"], how="left")
    df["revenue_yoy_pct"] = (_safe_div(df["revenue"], df["revenue_py"]) - 1) * 100
    df["fuel_cost_yoy_pct"] = (_safe_div(df["fuel_expense"], df["fuel_py"]) - 1) * 100

    long = df.melt(id_vars=["ticker", "role", "period_end", "quarter_label"],
                   value_vars=list(KPIS), var_name="kpi", value_name="value").dropna(subset=["value"])
    long["label"] = long["kpi"].map(lambda k: KPIS[k][0])
    long["unit"] = long["kpi"].map(lambda k: KPIS[k][1])
    long["better"] = long["kpi"].map(lambda k: KPIS[k][2])
    return flag_unusual(long, m)


def flag_unusual(long: pd.DataFrame, m=None) -> pd.DataFrame:
    """A quarter is 'unusual' when its change from the previous quarter is more than `unusual_sigma`
    (default 2) standard deviations of that company's normal quarter-to-quarter changes (COVID excluded)."""
    sigma = m.unusual_sigma if m is not None else 2.0
    long = long.sort_values(["ticker", "kpi", "period_end"]).copy()
    long["qoq_change"] = long.groupby(["ticker", "kpi"])["value"].diff()
    normal = long[_outside_covid(long["period_end"], m)]
    sd = normal.groupby(["ticker", "kpi"])["qoq_change"].std().rename("qoq_sd")
    long = long.join(sd, on=["ticker", "kpi"])
    long["unusual"] = (long["qoq_change"].abs() > sigma * long["qoq_sd"]) & long["qoq_sd"].gt(0)
    return long.drop(columns="qoq_sd")


# =========================================================================== peers
def peer_gaps(kpis: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """For each KPI: the focal company's latest quarter vs each peer in the SAME quarter."""
    rows = []
    focal = kpis[kpis["ticker"] == s.focal_company]
    for kpi, f in focal.groupby("kpi"):
        latest = f["period_end"].max()
        same_q = kpis[(kpis["kpi"] == kpi) & (kpis["period_end"] == latest)]
        if len(same_q) < 2:
            continue
        better = KPIS[kpi][2]
        ranked = same_q.sort_values("value", ascending=(better == "lower")).reset_index(drop=True)
        focal_val = same_q.loc[same_q["ticker"] == s.focal_company, "value"].iloc[0]
        peer_vals = same_q.loc[same_q["ticker"] != s.focal_company, "value"]
        for i, r in ranked.iterrows():
            rows.append({"kpi": kpi, "label": KPIS[kpi][0], "unit": KPIS[kpi][1], "better": better,
                         "period_end": latest, "quarter_label": r["quarter_label"], "ticker": r["ticker"],
                         "value": r["value"], "rank": i + 1, "n_companies": len(ranked),
                         "focal_value": focal_val, "peer_avg": peer_vals.mean(),
                         "gap_vs_peer_avg": focal_val - peer_vals.mean()})
    return pd.DataFrame(rows)


# =========================================================================== market
def _value_before(series: pd.Series, when: pd.Timestamp):
    s = series[series.index <= when]
    return s.iloc[-1] if len(s) else np.nan


# Levels that always grow (prices, the economy) are judged on their YoY growth rate instead:
# "CPI is at a 10-year high" is always true and says nothing; "inflation is 3.7%" does.
GROWTH_INSTEAD_OF_LEVEL = {"inflation": "CPI inflation (YoY)", "real_gdp": "Real GDP growth (YoY)"}


def _as_growth(g: pd.DataFrame, label: str) -> pd.DataFrame:
    s = g.set_index("obs_date")["value"]
    prior = s.copy()
    prior.index = prior.index + pd.DateOffset(years=1)
    yoy = (s / prior.reindex(s.index) - 1).dropna() * 100
    out = pd.DataFrame({"obs_date": yoy.index, "value": yoy.to_numpy()})
    out["code"], out["label"], out["unit"], out["category"] = g["code"].iloc[0], label, "percent", g["category"].iloc[0]
    return out


def market_snapshot(mkt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in mkt.groupby("code"):
        if code in GROWTH_INSTEAD_OF_LEVEL:
            g = _as_growth(g, GROWTH_INSTEAD_OF_LEVEL[code])
        s = g.set_index("obs_date")["value"]
        latest_date, latest = s.index[-1], s.iloc[-1]
        level_unit = g["unit"].iloc[0] in ("percent", "percentage points")   # change in points, not %
        row = {"code": code, "label": g["label"].iloc[0], "unit": g["unit"].iloc[0],
               "category": g["category"].iloc[0], "latest_date": latest_date, "latest": latest,
               "change_type": "points" if level_unit else "percent"}
        for name, days in (("1m", 30), ("3m", 91), ("1y", 365)):
            past = _value_before(s, latest_date - pd.Timedelta(days=days))
            row[f"chg_{name}"] = (latest - past) if level_unit else ((latest / past - 1) * 100 if past else np.nan)
        hist = s[s.index >= latest_date - pd.DateOffset(years=10)]
        row["hist_mean_10y"] = hist.mean()
        row["z_10y"] = (latest - hist.mean()) / hist.std() if hist.std() else np.nan
        row["percentile_10y"] = (hist < latest).mean() * 100
        year = s[s.index >= latest_date - pd.DateOffset(years=1)]
        row["low_52w"], row["high_52w"] = year.min(), year.max()
        p = row["percentile_10y"]
        row["position"] = "very high vs 10y" if p >= 90 else "very low vs 10y" if p <= 10 else \
                          "high vs 10y" if p >= 75 else "low vs 10y" if p <= 25 else "normal range"
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== sensitivity
@dataclass
class Fit:
    slope: float
    intercept: float
    r2: float
    n: int


def ols(x: pd.Series, y: pd.Series) -> Fit | None:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 8:
        return None
    xv, yv = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    if np.std(xv) == 0:                    # the factor never moved: no relationship to estimate
        return None
    slope, intercept = np.polyfit(xv, yv, 1)
    pred = slope * xv + intercept
    ss_res, ss_tot = ((yv - pred) ** 2).sum(), ((yv - yv.mean()) ** 2).sum()
    return Fit(slope, intercept, 1 - ss_res / ss_tot if ss_tot else 0.0, len(d))


def sensitivities(fin: pd.DataFrame, mkt: pd.DataFrame, s: Settings) -> pd.DataFrame:
    fuel_px = quarterly_average(mkt, "jet_fuel")
    ffr = quarterly_average(mkt, "fed_funds")
    rows = []
    for c in s.in_scope:
        f = fin[(fin["ticker"] == c.ticker)].set_index("period_end").sort_index()
        f = f[f.index.is_quarter_end]
        est = f[_outside_covid(f.index.to_series(), s.method).to_numpy()]
        latest = f.dropna(subset=["fuel_expense"]).iloc[-1]
        latest_q = latest.name
        px_now = fuel_px.get(latest_q, np.nan)

        # ---- fuel
        fit = ols(fuel_px.reindex(est.index), est["fuel_expense"])
        if fit and fit.r2 >= s.method.fuel_min_r2 and fit.slope > 0:
            per_pct = fit.slope * px_now / 100          # $ per quarter for a 1% fuel price move
            method = "regression (implied gallons x price)"
        else:
            per_pct = latest["fuel_expense"] / 100
            method = "proportional (fuel expense x shock)"
        rows.append({"ticker": c.ticker, "factor": "jet_fuel", "shock_unit": "% price change",
                     "impact_per_unit_q": -per_pct, "method": method,
                     "r2": fit.r2 if fit else np.nan, "n_quarters": fit.n if fit else 0,
                     "implied_gallons_q": fit.slope if fit and fit.slope > 0 else np.nan,
                     "base_quarter": latest_q, "base_price": px_now,
                     "base_operating_income_q": latest.get("operating_income", np.nan),
                     "base_revenue_q": latest.get("revenue", np.nan),
                     "caveat": "; ".join(c.caveats)})

        # ---- interest rates
        fit = ols(ffr.reindex(est.index), est.get("interest_expense"))
        modelled = bool(fit and fit.r2 >= s.method.rate_min_r2 and fit.slope > 0)
        rows.append({"ticker": c.ticker, "factor": "fed_funds", "shock_unit": "bp rate change",
                     "impact_per_unit_q": -(fit.slope / 100) if modelled else np.nan,
                     "method": "regression (interest expense on fed funds)" if modelled
                               else "not modelled: weak link (mostly fixed-rate debt) - use 10-K Item 7A",
                     "r2": fit.r2 if fit else np.nan, "n_quarters": fit.n if fit else 0,
                     "implied_gallons_q": np.nan, "base_quarter": latest_q,
                     "base_price": ffr.get(latest_q, np.nan),
                     "base_operating_income_q": latest.get("operating_income", np.nan),
                     "base_revenue_q": latest.get("revenue", np.nan)})
    return pd.DataFrame(rows)


def past_ranges(mkt: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Largest falls and rises actually seen over any 4-quarter window in the stored history:
    jet fuel in % (quarterly average price), fed funds in basis points. A scenario beyond these
    is 'outside past experience' - the linear estimate is then an extrapolation."""
    out = {}
    fuel = quarterly_average(mkt, "jet_fuel").dropna()
    if len(fuel) > 4:
        chg = (fuel / fuel.shift(4) - 1).dropna() * 100
        out["jet_fuel"] = (float(chg.min()), float(chg.max()))
    ffr = quarterly_average(mkt, "fed_funds").dropna()
    if len(ffr) > 4:
        chg = (ffr - ffr.shift(4)).dropna() * 100
        out["fed_funds"] = (float(chg.min()), float(chg.max()))
    return out


def within_past(fuel_pct: float, rate_bp: float, ranges: dict | None) -> bool | None:
    if not ranges:
        return None
    ok = True
    for value, key in ((fuel_pct, "jet_fuel"), (rate_bp, "fed_funds")):
        if value and key in ranges:
            lo, hi = ranges[key]
            ok &= lo <= value <= hi
    return bool(ok)


def what_if(sens: pd.DataFrame, ticker: str, fuel_pct: float = 0.0, rate_bp: float = 0.0,
            ranges: dict | None = None) -> dict:
    """Live simulator used by the dashboard sliders. Same maths as the scenario table.
    With `ranges` (from past_ranges) it also says whether the shock was ever seen in the data."""
    s = sens[sens["ticker"] == ticker].set_index("factor")
    fuel = s.loc["jet_fuel"]
    impact = fuel["impact_per_unit_q"] * fuel_pct
    rate_row = s.loc["fed_funds"]
    rate_impact = 0.0 if pd.isna(rate_row["impact_per_unit_q"]) else rate_row["impact_per_unit_q"] * rate_bp
    op = fuel["base_operating_income_q"] + impact
    total = impact + rate_impact          # rate impact hits pre-tax income, not operating income
    return {"ticker": ticker, "fuel_pct": fuel_pct, "rate_bp": rate_bp,
            "fuel_impact_q": impact, "rate_impact_q": rate_impact, "total_impact_q": total,
            "operating_income_q": op,
            "operating_margin_pct": op / fuel["base_revenue_q"] * 100,
            "base_operating_margin_pct": fuel["base_operating_income_q"] / fuel["base_revenue_q"] * 100,
            "rate_modelled": not pd.isna(rate_row["impact_per_unit_q"]),
            "within_past_range": within_past(fuel_pct, rate_bp, ranges),
            "note": None if within_past(fuel_pct, rate_bp, ranges) in (True, None)
                    else "Outside moves seen in the data: an extrapolation, treat as rough"}


def historical_moves(mkt: pd.DataFrame, s: Settings) -> list[dict]:
    """Measure each configured episode from the data: % change in jet fuel and bp change in
    fed funds between the average of the start month and the average of the end month."""
    out = []
    last = mkt["obs_date"].max() if len(mkt) else None
    for sc in s.scenarios:
        end = pd.Timestamp(sc.end) if sc.end else last
        start = pd.Timestamp(sc.start) if sc.start else (end - pd.DateOffset(years=1) if end is not None else None)
        if end is None or start is None:
            continue
        def month_avg(code, when):
            v = mkt[(mkt["code"] == code) & (mkt["obs_date"].dt.to_period("M") == when.to_period("M"))]["value"]
            return v.mean() if len(v) else np.nan
        f0, f1 = month_avg("jet_fuel", start), month_avg("jet_fuel", end)
        r0, r1 = month_avg("fed_funds", start), month_avg("fed_funds", end)
        if pd.isna(f0) or pd.isna(f1):
            continue
        out.append({"scenario": sc.name, "start": start.date(), "end": end.date(), "note": sc.note,
                    "fuel_pct": (f1 / f0 - 1) * 100, "rate_bp": 0.0 if pd.isna(r0) or pd.isna(r1) else (r1 - r0) * 100})
    return out


def scenario_grid(sens: pd.DataFrame, ranges: dict | None = None, history: list[dict] | None = None) -> pd.DataFrame:
    """All scenarios for every company. scenario_type: sensitivity (one factor), combined
    (fuel + rates added), historical (a past episode replayed on today's company)."""
    rows = []

    def row(t, stype, name, factor, shock, unit, fuel_pct, rate_bp, extra=None):
        r = what_if(sens, t, fuel_pct=fuel_pct, rate_bp=rate_bp, ranges=ranges)
        rate_only = factor == "fed_funds"
        impact = (r["rate_impact_q"] if r["rate_modelled"] else np.nan) if rate_only else \
                 r["fuel_impact_q"] if factor == "jet_fuel" else r["total_impact_q"]
        rows.append({"ticker": t, "scenario_type": stype, "scenario": name, "factor": factor,
                     "shock": shock, "shock_unit": unit, "fuel_pct": fuel_pct, "rate_bp": rate_bp,
                     "impact_q": impact, "fuel_impact_q": r["fuel_impact_q"],
                     "rate_impact_q": r["rate_impact_q"] if r["rate_modelled"] else np.nan,
                     "operating_income_q": np.nan if rate_only else r["operating_income_q"],
                     "operating_margin_pct": np.nan if rate_only else r["operating_margin_pct"],
                     "within_past_range": r["within_past_range"], **(extra or {})})

    for t in sens["ticker"].unique():
        for shock in FUEL_SHOCKS:
            row(t, "sensitivity", f"jet fuel {shock:+d}%", "jet_fuel", shock, "%", shock, 0)
        for shock in RATE_SHOCKS:
            row(t, "sensitivity", f"rates {shock:+d}bp", "fed_funds", shock, "bp", 0, shock)
        for f, r in COMBINED:
            row(t, "combined", f"jet fuel {f:+d}% and rates {r:+d}bp (added, no interaction)", "combined", None, None, f, r)
        for h in history or []:
            row(t, "historical", h["scenario"], "historical", None, None, round(h["fuel_pct"], 1), round(h["rate_bp"]),
                {"period": f"{h['start']} to {h['end']}", "note": h["note"]})
    return pd.DataFrame(rows)


# =========================================================================== run
def run(s: Settings, eng: Engine) -> dict[str, pd.DataFrame]:
    fin, mkt = load_financials(eng), load_market(eng)
    from dashboard.robustness import robustness
    kpis = compute_kpis(fin, s.method)
    sens = sensitivities(fin, mkt, s)
    rob, share = robustness(fin, mkt, s, base=sens)
    sens = sens.merge(share, on=["ticker", "factor"], how="left")       # robust_pct travels to the AI as a fact
    ranges = past_ranges(mkt)
    from dashboard.valuation import load_stock, stock_index, valuation
    try:
        prices, shares = load_stock(eng)
    except Exception:                      # gold not rebuilt since the stock tables were added
        prices, shares = pd.DataFrame(columns=["ticker", "trade_date", "close"]), pd.DataFrame()
    out = {"kpi_quarterly": kpis, "peer_gap": peer_gaps(kpis, s), "market_snapshot": market_snapshot(mkt),
           "sensitivity": sens, "scenario": scenario_grid(sens, ranges, historical_moves(mkt, s)),
           "robustness": rob,
           "past_range": pd.DataFrame([{"code": k, "low": lo, "high": hi} for k, (lo, hi) in ranges.items()],
                                      columns=["code", "low", "high"]),
           "stock_index": stock_index(prices, s.stocks.index_years),
           "valuation": valuation(prices, shares, fin, s)}
    from dashboard.bi import create_views, drop_views
    with eng.begin() as con:
        drop_views(con)                    # Power BI views sit on these tables; rebuilt right after
        for name, df in out.items():
            df.assign(computed_at=pd.Timestamp.now(tz="UTC")).to_sql(
                name, con, schema="gold", if_exists="replace", index=False)
    create_views(eng)
    return out


def _show(method: str, value: float) -> str:
    """'regression -$379M' / 'not modelled' / '3 flags' for the robustness printout."""
    if method == "count":
        return f"{value:.0f} flags"
    if method.startswith("not") or value != value:
        return "not modelled"
    return f"{method.split(' ')[0]} {'-' if value < 0 else ''}${abs(value) / 1e6:,.0f}M"


def _fmt(v: float, unit: str) -> str:
    if pd.isna(v):
        return "-"
    if unit == "USD":
        return f"${v / 1e9:,.2f}B" if abs(v) >= 1e9 else f"${v / 1e6:,.0f}M"
    if unit == "x":
        return f"{v:.1f}x"
    return f"{v:.1f}%"


def main() -> None:
    from dashboard.db import database_url, engine, masked
    s = load_settings()
    print(f"Computing in {masked(database_url())}")
    out = run(s, engine())
    for name, df in out.items():
        print(f"  gold.{name:<16}{len(df):>6} rows")

    pg = out["peer_gap"]
    focal = pg[pg["ticker"] == s.focal_company].sort_values("kpi")
    print(f"\n{s.focal_company} vs PEERS ({focal['quarter_label'].iloc[0] if len(focal) else '-'})")
    print(f"  {'KPI':<24}{s.focal_company:>11}{'PEER AVG':>11}{'RANK':>7}")
    for r in focal.itertuples():
        print(f"  {r.label:<24}{_fmt(r.value, r.unit):>11}{_fmt(r.peer_avg, r.unit):>11}{f'{r.rank}/{r.n_companies}':>7}")

    ms = out["market_snapshot"].sort_values("percentile_10y", ascending=False)
    print("\nMARKET TODAY")
    print(f"  {'VARIABLE':<34}{'LATEST':>10}{'1M':>9}{'1Y':>9}  POSITION")
    for r in ms.itertuples():
        suffix = "pp" if r.change_type == "points" else "%"
        print(f"  {r.label[:33]:<34}{r.latest:>10.2f}{r.chg_1m:>+8.1f}{suffix[0]}{r.chg_1y:>+8.1f}{suffix[0]}  {r.position}")

    print("\nSENSITIVITY (impact on quarterly results)")
    for r in out["sensitivity"].itertuples():
        unit = 10 if r.factor == "jet_fuel" else 100
        lbl = "+10% jet fuel" if r.factor == "jet_fuel" else "+100bp rates "
        val = "-" if pd.isna(r.impact_per_unit_q) else _fmt(r.impact_per_unit_q * unit, "USD")
        print(f"  {r.ticker:<5}{lbl}  {val:>9}  R2={r.r2:.2f}  {r.method}")

    sc = out["scenario"]
    hist = sc[(sc["scenario_type"] == "historical") & (sc["ticker"] == s.focal_company)]
    if len(hist):
        print(f"\nHISTORICAL SCENARIOS replayed on {s.focal_company} today (quarterly impact)")
        for r in hist.itertuples():
            flag = "" if r.within_past_range in (True, None) else "  [outside past range]"
            print(f"  {r.scenario:<32} fuel {r.fuel_pct:+6.1f}%  rates {r.rate_bp:+5.0f}bp  ->  "
                  f"{_fmt(r.impact_q, 'USD')}{flag}")

    rob = out["robustness"]
    changed = rob[~rob["same_result"]]
    print(f"\nROBUSTNESS: {len(rob)} checks with alternative thresholds, {len(changed)} changed a result")
    for r in changed.itertuples():
        print(f"  {r.parameter} {r.chosen} -> {r.alternative}: {r.ticker} {r.factor}: "
              f"{_show(r.method_base, r.impact_base)} -> {_show(r.method_alt, r.impact_alt)}")


if __name__ == "__main__":
    main()
