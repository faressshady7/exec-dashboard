"""Step 5 tests: the maths behind every number the AI will report."""
import numpy as np
import pandas as pd
import pytest

from dashboard.compute import (compute_kpis, market_snapshot, ols, peer_gaps, scenario_grid,
                               sensitivities, what_if)
from dashboard.config import load_settings


def quarters(n, start="2016-03-31"):
    return pd.date_range(start, periods=n, freq="QE")


def fin_frame(ticker="UAL", role="focal", n=12, fuel_px=None):
    q = quarters(n)
    px = fuel_px if fuel_px is not None else np.linspace(1.5, 3.0, n)
    gallons = 1.0e9
    fuel = gallons * px
    return pd.DataFrame({
        "ticker": ticker, "role": role, "period_end": q, "quarter_label": [f"Q{x.quarter} {x.year}" for x in q],
        "revenue": np.linspace(10e9, 12e9, n), "operating_expenses": 8e9 + fuel, "fuel_expense": fuel,
        "operating_income": np.linspace(10e9, 12e9, n) - (8e9 + fuel), "net_income": 0.5e9,
        "interest_expense": 0.2e9, "operating_cash_flow": 1.5e9, "capex": 1.0e9,
        "cash": 5e9, "long_term_debt": 18e9, "debt_current": 2e9, "short_term_investments": 1e9,
        "total_assets": 80e9, "equity": 10e9})


def market_frame(px_by_quarter: pd.Series) -> pd.DataFrame:
    rows = []
    for qend, px in px_by_quarter.items():
        for d in pd.date_range(qend - pd.offsets.QuarterBegin(startingMonth=1), qend, freq="D"):
            rows.append(("jet_fuel", "Jet fuel", "USD per gallon", "cost", d, px))
            rows.append(("fed_funds", "Fed funds", "percent", "rates", d, 2.0))
    return pd.DataFrame(rows, columns=["code", "label", "unit", "category", "obs_date", "value"])


def test_kpis_margins_and_yoy():
    k = compute_kpis(fin_frame()).set_index(["period_end", "kpi"])["value"]
    first, fifth = quarters(12)[0], quarters(12)[4]
    rev = np.linspace(10e9, 12e9, 12)
    assert k[(first, "debt_to_assets_pct")] == pytest.approx(25.0)
    assert k[(first, "free_cash_flow")] == pytest.approx(0.5e9)
    assert k[(first, "net_debt")] == pytest.approx(14e9)                 # 18 + 2 debt - (5 cash + 1 investments)
    assert (first, "revenue_yoy_pct") not in k.index                      # no prior year yet
    assert k[(fifth, "revenue_yoy_pct")] == pytest.approx((rev[4] / rev[0] - 1) * 100)


def test_net_debt_is_like_for_like():
    f = fin_frame(n=4)
    f.loc[1, "short_term_investments"] = np.nan                            # untagged in one 10-Q -> unknown, not 0
    f.loc[2, "debt_current"] = np.nan                                      # missing debt part -> unknown, not smaller
    k = compute_kpis(f)
    nd = k[k.kpi == "net_debt"].set_index("period_end")["value"]
    q = quarters(4)
    assert q[1] not in nd.index and q[2] not in nd.index and nd[q[3]] == pytest.approx(14e9)


def test_investments_line_dropped_after_zero_counts_as_zero():
    f = fin_frame(n=4)
    f.loc[1, "short_term_investments"] = 0.0                               # reported $0 (Delta, Dec 2024) ...
    f.loc[[2, 3], "short_term_investments"] = np.nan                       # ... then the line disappears
    nd = compute_kpis(f).query("kpi == 'net_debt'").set_index("period_end")["value"]
    q = quarters(4)
    assert nd[q[2]] == pytest.approx(15e9) and nd[q[3]] == pytest.approx(15e9)   # 18 + 2 - (5 + 0)


def test_unusual_flag_catches_a_jump():
    f = fin_frame(n=12)
    f.loc[10, "cash"] = 50e9                                               # sudden jump
    k = compute_kpis(f)
    cash = k[k.kpi == "liquidity"].set_index("period_end")
    assert cash.loc[quarters(12)[10], "unusual"] and not cash.loc[quarters(12)[5], "unusual"]


def test_ols_recovers_slope():
    x = pd.Series(np.arange(10.0))
    fit = ols(x, 3 * x + 1)
    assert fit.slope == pytest.approx(3) and fit.r2 == pytest.approx(1)
    assert ols(x[:5], x[:5]) is None                                       # too few points


def test_fuel_sensitivity_recovers_gallons_and_what_if():
    s = load_settings()
    q = quarters(12)
    px = pd.Series(np.linspace(1.5, 3.0, 12), index=q)
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    sens = sensitivities(fin, market_frame(px), s)
    fuel = sens[(sens.ticker == "UAL") & (sens.factor == "jet_fuel")].iloc[0]
    assert fuel["implied_gallons_q"] == pytest.approx(1e9, rel=1e-6)
    assert fuel["method"].startswith("regression")
    caveat = sens[(sens.ticker == "DAL") & (sens.factor == "jet_fuel")].iloc[0]["caveat"]
    assert "refinery" in caveat and fuel["caveat"] == ""                  # Delta flagged, United not
    assert not any(ch.isdigit() for ch in caveat)                         # number-free: AI number check unaffected
    # +10% fuel at the latest price 3.0 -> 1e9 gal x 3.0 x 10% = $300M lower operating income
    r = what_if(sens, "UAL", fuel_pct=10)
    assert r["fuel_impact_q"] == pytest.approx(-300e6, rel=1e-6)
    assert r["operating_income_q"] == pytest.approx(fuel["base_operating_income_q"] - 300e6, rel=1e-6)
    rate = sens[(sens.ticker == "UAL") & (sens.factor == "fed_funds")].iloc[0]
    assert rate["method"].startswith("not modelled")                       # flat rates -> no fit
    grid = scenario_grid(sens)
    assert set(grid.factor) == {"jet_fuel", "fed_funds", "combined"} and grid[grid.factor == "fed_funds"].impact_q.isna().all()


def test_peer_gap_ranks_by_direction():
    s = load_settings()
    fin = pd.concat([fin_frame("UAL", "focal"), fin_frame("DAL", "peer"), fin_frame("AAL", "peer")])
    fin.loc[fin.ticker == "DAL", "long_term_debt"] = 8e9                   # DAL least indebted
    k = compute_kpis(fin.reset_index(drop=True))
    pg = peer_gaps(k, s)
    d2a = pg[pg.kpi == "debt_to_assets_pct"].set_index("ticker")
    assert d2a.loc["DAL", "rank"] == 1                                     # lower is better
    assert d2a.loc["UAL", "gap_vs_peer_avg"] == pytest.approx(25.0 - (12.5 + 25.0) / 2)


def test_peer_gap_waits_for_the_quarter_every_airline_has_reported():
    s = load_settings()
    fin = pd.concat([fin_frame("UAL", "focal"), fin_frame("DAL", "peer"), fin_frame("AAL", "peer")])
    k = compute_kpis(fin.reset_index(drop=True))
    newest = k["period_end"].max()
    k = k[~((k.ticker == "AAL") & (k.period_end == newest))]          # AAL has not filed the newest quarter yet
    pg = peer_gaps(k, s)
    assert (pg["period_end"] < newest).all()                            # compared in the older, common quarter
    assert (pg.groupby("kpi")["ticker"].nunique() == 3).all()           # all three airlines in every comparison


def test_market_snapshot_uses_growth_for_cpi():
    d = pd.date_range("2015-01-01", "2026-10-01", freq="MS")
    cpi = pd.DataFrame({"code": "inflation", "label": "CPI", "unit": "index", "category": "macro",
                        "obs_date": d, "value": 100 * 1.03 ** (np.arange(len(d)) / 12)})
    snap = market_snapshot(cpi).iloc[0]
    assert snap["label"] == "CPI inflation (YoY)"
    assert snap["latest"] == pytest.approx(3.0, abs=0.01)                 # 3% a year, not a "record level"


def test_refinery_caveat_reaches_the_ai_text():
    from dashboard.insights import exposure_insights, peer_insights
    s = load_settings()
    q = quarters(12)
    px = pd.Series(np.linspace(1.5, 3.0, 12), index=q)
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    ins = exposure_insights(sensitivities(fin, market_frame(px), s), "UAL")
    fuel = next(i for i in ins if i.id == "exposure:jet_fuel")
    assert "Caveat: Delta owns an oil refinery" in fuel.summary
    pg = pd.DataFrame([{"kpi": k, "ticker": t, "unit": "%", "gap_vs_peer_avg": 1.0, "peer_avg": 25.0,
                        "better": "lower", "value": 26.0, "label": "Fuel share of costs", "rank": 2,
                        "n_companies": 3, "quarter_label": "Q2 2026"}
                       for k in ("fuel_share_opex_pct", "operating_margin_pct") for t in ("UAL", "DAL", "AAL")])
    caveats = {c.ticker: "; ".join(c.caveats) for c in s.companies}
    out = {i.id: i.summary for i in peer_insights(pg, "UAL", caveats)}
    assert "refinery" in out["peer:fuel_share_opex_pct"] and "refinery" not in out["peer:operating_margin_pct"]


def test_thresholds_come_from_settings():
    s = load_settings()
    q = quarters(12)
    px = pd.Series(np.linspace(1.5, 3.0, 12), index=q)
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    strict = s.model_copy(update={"method": s.method.model_copy(update={"fuel_min_r2": 1.01})})  # impossible bar
    fuel = sensitivities(fin, market_frame(px), strict).query("ticker == 'UAL' and factor == 'jet_fuel'").iloc[0]
    assert fuel["method"].startswith("proportional")


def test_robustness_table_and_share():
    from dashboard.robustness import robustness
    s = load_settings()
    q = quarters(12)
    px = pd.Series(np.linspace(1.5, 3.0, 12), index=q)
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    rob, share = robustness(fin, market_frame(px), s)
    assert {"fuel_min_r2", "rate_min_r2", "exclude_covid", "unusual_sigma"} <= set(rob["parameter"])
    ual_fuel = share.query("ticker == 'UAL' and factor == 'jet_fuel'")["robust_pct"].iloc[0]
    assert ual_fuel == 100                     # perfect synthetic fit: no threshold changes the result


def test_sensitivity_text_is_framed_as_an_estimate():
    from dashboard.insights import exposure_insights
    s = load_settings()
    q = quarters(12)
    px = pd.Series(np.linspace(1.5, 3.0, 12), index=q)
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    fuel = next(i for i in exposure_insights(sensitivities(fin, market_frame(px), s), "UAL") if i.id == "exposure:jet_fuel")
    assert fuel.summary.startswith("Based on past quarters") and "estimated" in fuel.summary and " would " not in fuel.summary


def test_past_range_flag_combined_and_historical_scenarios():
    from datetime import date
    from dashboard.compute import historical_moves, past_ranges, scenario_grid
    from dashboard.config import HistoricalScenario
    s = load_settings()
    q = quarters(12)
    px = pd.Series(np.linspace(1.5, 3.0, 12), index=q)           # max 4-quarter rise ~ +29%
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    mkt = market_frame(px)
    sens = sensitivities(fin, mkt, s)
    ranges = past_ranges(mkt)
    lo, hi = ranges["jet_fuel"]
    assert 0 < lo < hi < 40                                        # only rises in this synthetic history
    assert what_if(sens, "UAL", fuel_pct=hi - 1, ranges=ranges)["within_past_range"] is True
    far = what_if(sens, "UAL", fuel_pct=100, ranges=ranges)
    assert far["within_past_range"] is False and "extrapolation" in far["note"]
    # combined = fuel + rates added (rates not modelled here -> fuel part only)
    s2 = s.model_copy(update={"scenarios": [HistoricalScenario(name="ramp", start=date(2016, 1, 15), end=date(2018, 12, 15))]})
    hist = historical_moves(mkt, s2)
    assert hist[0]["fuel_pct"] == pytest.approx((3.0 / 1.5 - 1) * 100)   # first vs last quarter price
    grid = scenario_grid(sens, ranges, hist)
    h = grid[(grid.scenario_type == "historical") & (grid.ticker == "UAL")].iloc[0]
    assert h["impact_q"] == pytest.approx(-30e6 * 100, rel=1e-6)          # -$30M per 1% x +100%
    assert h["within_past_range"] is False or h["within_past_range"] == False
    assert (grid.scenario_type == "combined").sum() == 3 * 4
