"""Step 6a - Candidate insights (deterministic, adapted from LEVA's insight functions).

LEVA: the LLM picks insight functions, code computes them, each insight is scored
    score = 0.5 * significance + 0.3 * relevance + 0.2 * impact
Here the code computes EVERY candidate from the Step 5 tables (cheap and reproducible).
Significance and impact are computed by code; RELEVANCE (to an executive, today) is the
part the LLM judges in round 1. So the LLM ranks; it never produces a number.

Each insight carries its facts: the exact numbers (with id, value, unit) it may cite.
The number checker later accepts a figure in the brief only if it matches one of them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboard.config import Settings

# How much each market category matters to an airline's results (impact component)
MARKET_IMPACT = {"cost": 1.0, "demand": 0.7, "rates": 0.6, "uncertainty": 0.6, "fx": 0.5, "market": 0.5, "macro": 0.4}
KPI_IMPACT = {"operating_margin_pct": 1.0, "fuel_share_opex_pct": 0.9, "net_margin_pct": 0.8,
              "free_cash_flow": 0.8, "interest_coverage_x": 0.7, "revenue_yoy_pct": 0.7,
              "fuel_cost_yoy_pct": 0.7, "debt_to_assets_pct": 0.6, "net_debt": 0.6, "liquidity": 0.5,
              "revenue": 0.4}


@dataclass
class Fact:
    id: str
    label: str
    value: float
    unit: str            # USD | % | pp | x | USD/gal | index | count ...

    def display(self) -> str:
        return fmt(self.value, self.unit)


@dataclass
class Insight:
    id: str
    type: str            # market_position | peer_gap | trend | exposure | scenario | change_point | news_event
    title: str
    summary: str         # one plain sentence, numbers already formatted
    facts: list[Fact]
    significance: float  # 0..1, computed
    impact: float        # 0..1, computed
    relevance: float = 0.0   # 0..1, from the LLM in round 1
    score: float = 0.0
    group: str = ""          # theme: at most one insight per group in the final brief
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["facts"] = [asdict(f) | {"display": f.display()} for f in self.facts]
        return d


def fmt(v: float, unit: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if unit == "USD":
        sign, a = ("-" if v < 0 else ""), abs(v)
        return f"{sign}${a / 1e9:,.2f}B" if a >= 1e9 else f"{sign}${a / 1e6:,.0f}M"
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "pp":
        return f"{v:+.1f} pp"
    if unit == "x":
        return f"{v:.1f}x"
    if unit == "USD/gal":
        return f"${v:.2f}/gal"
    if unit == "USD/share":
        return f"${v:.2f}"
    if unit == "USD/bbl":
        return f"${v:.2f}/bbl"
    if unit == "count":
        return f"{int(v)}"
    return f"{v:,.2f}"


def _lc(label: str) -> str:
    """'Fuel cost growth (YoY)' -> 'fuel cost growth (YoY)': lower-case the first letter only."""
    return label[:1].lower() + label[1:]


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, 0.0 if pd.isna(x) else x)))


def _read(eng: Engine, table: str) -> pd.DataFrame:
    with eng.connect() as con:
        return pd.DataFrame(con.execute(text(f"SELECT * FROM gold.{table}")).mappings().all())


# =========================================================================== generators
def market_insights(ms: pd.DataFrame) -> list[Insight]:
    out = []
    unit_of = {"USD per gallon": "USD/gal", "USD per barrel": "USD/bbl", "percent": "%",
               "percentage points": "pp"}
    for r in ms.itertuples():
        u = unit_of.get(r.unit, "index")
        chg_u = "pp" if r.change_type == "points" else "%"
        facts = [Fact(f"{r.code}.latest", f"{r.label} latest", r.latest, u),
                 Fact(f"{r.code}.chg_1y", f"{r.label} change over 1 year", r.chg_1y, chg_u),
                 Fact(f"{r.code}.chg_1m", f"{r.label} change over 1 month", r.chg_1m, chg_u),
                 Fact(f"{r.code}.pctl", f"{r.label} percentile vs last 10 years", r.percentile_10y, "%")]
        sig = _clip(abs(r.z_10y) / 3) if pd.notna(r.z_10y) else 0.0
        out.append(Insight(
            id=f"market:{r.code}", type="market_position", title=f"{r.label}: {r.position}",
            summary=(f"{r.label} is {facts[0].display()}, {'up' if r.chg_1y >= 0 else 'down'} "
                     f"{fmt(abs(r.chg_1y), chg_u).lstrip('+')} over a year, "
                     + ("the highest in 10 years." if r.percentile_10y >= 99.5 else
                        "the lowest in 10 years." if r.percentile_10y <= 0.5 else
                        f"at the {r.percentile_10y:.0f}th percentile of the last 10 years ({r.position}).")),
            facts=facts, significance=sig, impact=MARKET_IMPACT.get(r.category, 0.4),
            group=f"market:{r.category}",                 # jet fuel and Brent = one theme (cost)
            sources=[f"FRED via gold.market_snapshot ({pd.Timestamp(r.latest_date).date()})"]))
    return out


FUEL_KPIS = {"fuel_share_opex_pct", "fuel_cost_yoy_pct"}


def _caveat_text(caveats: dict[str, str], tickers) -> str:
    notes = [caveats[t] for t in tickers if caveats.get(t)]
    return (" Caveat: " + "; ".join(notes) + ".") if notes else ""


def peer_insights(pg: pd.DataFrame, focal: str, caveats: dict[str, str] | None = None) -> list[Insight]:
    caveats = caveats or {}
    out = []
    for kpi, g in pg.groupby("kpi"):
        if (g["ticker"] == focal).sum() == 0:
            continue
        f = g[g["ticker"] == focal].iloc[0].copy()
        # peer average = the OTHER companies in this table (works for any company as the focal one;
        # gold.peer_gap's own peer_avg column is relative to the configured focal company only)
        others = g[g["ticker"] != focal]["value"]
        f["peer_avg"] = others.mean()
        f["gap_vs_peer_avg"] = f["value"] - f["peer_avg"]
        unit = f["unit"]
        gap_unit = "pp" if unit == "%" else unit
        rel = abs(f["gap_vs_peer_avg"]) / abs(f["peer_avg"]) if f["peer_avg"] else 0
        if unit == "%":
            rel = abs(f["gap_vs_peer_avg"]) / 10           # 10 percentage points = maximal
        good = (f["gap_vs_peer_avg"] > 0) == (f["better"] == "higher")
        q = f["quarter_label"]
        facts = [Fact(f"peer.{kpi}.focal", f"{focal} {f['label']} ({q})", f["value"], unit),
                 Fact(f"peer.{kpi}.peer_avg", f"Peer average {f['label']} ({q})", f["peer_avg"], unit),
                 Fact(f"peer.{kpi}.gap", f"{focal} minus peer average", f["gap_vs_peer_avg"], gap_unit),
                 Fact(f"peer.{kpi}.rank", f"{focal} rank among {int(f['n_companies'])}", f["rank"], "count")]
        for _, p in g[g["ticker"] != focal].iterrows():
            facts.append(Fact(f"peer.{kpi}.{p['ticker']}", f"{p['ticker']} {f['label']} ({q})", p["value"], unit))
        out.append(Insight(
            id=f"peer:{kpi}", type="peer_gap",
            title=f"{f['label']}: {focal} ranks {int(f['rank'])} of {int(f['n_companies'])}",
            summary=(f"{focal} {_lc(f['label'])} is {facts[0].display()} vs a peer average of "
                     f"{facts[1].display()} in {f['quarter_label']} "
                     f"({'better' if good else 'worse'} than peers; rank {int(f['rank'])}/{int(f['n_companies'])})."
                     + (_caveat_text(caveats, g["ticker"]) if kpi in FUEL_KPIS else "")),
            facts=facts, significance=_clip(rel), impact=KPI_IMPACT.get(kpi, 0.5),
            sources=[f"SEC 10-Q/10-K via gold.peer_gap ({f['quarter_label']})"]))
    return out


def trend_insights(kq: pd.DataFrame, focal: str) -> list[Insight]:
    out = []
    f = kq[kq["ticker"] == focal]
    for kpi in ("revenue_yoy_pct", "fuel_cost_yoy_pct", "operating_margin_pct"):
        g = f[f["kpi"] == kpi].sort_values("period_end")
        if len(g) < 5:
            continue
        last, year_ago = g.iloc[-1], g.iloc[-5]
        change = last["value"] - year_ago["value"]
        hist = g["value"].iloc[:-1]
        z = (last["value"] - hist.mean()) / hist.std() if hist.std() else 0
        facts = [Fact(f"trend.{kpi}.latest", f"{focal} {last['label']} {last['quarter_label']}", last["value"], "%"),
                 Fact(f"trend.{kpi}.year_ago", f"{focal} {last['label']} {year_ago['quarter_label']}", year_ago["value"], "%"),
                 Fact(f"trend.{kpi}.change", f"Change vs a year earlier", change, "pp")]
        out.append(Insight(
            id=f"trend:{kpi}", type="trend", title=f"{last['label']} trend",
            summary=(f"{focal} {_lc(last['label'])} was {facts[0].display()} in {last['quarter_label']} "
                     f"vs {facts[1].display()} in {year_ago['quarter_label']} ({fmt(change, 'pp')})."),
            facts=facts, significance=_clip(abs(z) / 3), impact=KPI_IMPACT.get(kpi, 0.5),
            sources=[f"SEC 10-Q/10-K via gold.kpi_quarterly ({last['quarter_label']})"]))
    # change points: KPIs flagged unusual in the latest quarter
    latest_q = f["period_end"].max()
    for r in f[(f["period_end"] == latest_q) & (f["unusual"] == True)].itertuples():  # noqa: E712
        unit = r.unit
        out.append(Insight(
            id=f"change:{r.kpi}", type="change_point", title=f"Unusual move in {r.label}",
            summary=f"{focal} {_lc(r.label)} moved {fmt(r.qoq_change, 'pp' if unit == '%' else unit)} "
                    f"in {r.quarter_label}, more than twice its normal quarterly change.",
            facts=[Fact(f"change.{r.kpi}.value", f"{focal} {r.label} {r.quarter_label}", r.value, unit),
                   Fact(f"change.{r.kpi}.qoq", "Change vs previous quarter", r.qoq_change, "pp" if unit == "%" else unit)],
            significance=0.8, impact=KPI_IMPACT.get(r.kpi, 0.5),
            sources=[f"gold.kpi_quarterly ({r.quarter_label})"]))
    return out


def exposure_insights(sens: pd.DataFrame, focal: str) -> list[Insight]:
    out = []
    fuel = sens[sens["factor"] == "jet_fuel"].set_index("ticker")
    if focal in fuel.index:
        f = fuel.loc[focal]
        imp10 = f["impact_per_unit_q"] * 10
        share = abs(imp10) / abs(f["base_operating_income_q"]) if f["base_operating_income_q"] else 1
        facts = [Fact("exposure.fuel.impact_10pct", f"{focal} quarterly operating income change for +10% jet fuel", imp10, "USD"),
                 Fact("exposure.fuel.base_oi", f"{focal} quarterly operating income (base)", f["base_operating_income_q"], "USD"),
                 Fact("exposure.fuel.r2", "Fit of fuel cost vs jet fuel price (R-squared)", f["r2"], "index")]
        if pd.notna(f.get("robust_pct", np.nan)):
            facts.append(Fact("exposure.fuel.robust_pct", "Share of alternative model settings giving the same fuel result",
                              f["robust_pct"], "%"))
        for t, p in fuel.drop(index=focal).iterrows():
            facts.append(Fact(f"exposure.fuel.{t}", f"{t} change for +10% jet fuel", p["impact_per_unit_q"] * 10, "USD"))
        out.append(Insight(
            id="exposure:jet_fuel", type="exposure", title="Exposure to jet fuel",
            summary=(f"Based on past quarters, a 10% rise in jet fuel is estimated to reduce {focal} quarterly operating income by about "
                     f"{fmt(abs(imp10), 'USD')}, against a base of {facts[1].display()}."
                     + _caveat_text(fuel.get("caveat", pd.Series(dtype=str)).fillna("").to_dict(), fuel.index)),
            facts=facts, significance=_clip(share * 2), impact=1.0,      # a 10% shock removing >=50% of profit = maximal
            sources=[f"Regression on {int(f['n_quarters'])} quarters, gold.sensitivity ({f['method']})"]))
    rate = sens[(sens["factor"] == "fed_funds") & (sens["ticker"] == focal)]
    if len(rate) and pd.notna(rate.iloc[0]["impact_per_unit_q"]):
        r = rate.iloc[0]
        imp = r["impact_per_unit_q"] * 100
        out.append(Insight(
            id="exposure:fed_funds", type="exposure", title="Exposure to interest rates",
            summary=f"Based on past quarters, a 1-point (100bp) rise in rates is linked to about {fmt(abs(imp), 'USD')} more {focal} interest cost a quarter (indicative; R-squared {r['r2']:.2f}).",
            facts=[Fact("exposure.rates.impact_100bp", f"{focal} quarterly cost of +100bp rates", imp, "USD"),
                   Fact("exposure.rates.r2", "Fit of interest expense vs fed funds (R-squared)", r["r2"], "index")]
                  + ([Fact("exposure.rates.robust_pct", "Share of alternative model settings giving the same rate result",
                           r["robust_pct"], "%")] if pd.notna(r.get("robust_pct", np.nan)) else []),
            significance=_clip(abs(imp) / abs(r["base_operating_income_q"]) * 3) if r["base_operating_income_q"] else 0.3,
            impact=0.6, sources=["gold.sensitivity (indicative: debt level also changed)"]))
    return out


def news_insights(eng: Engine, focal: str, days: int = 30, max_tier: int = 2) -> list[Insight]:
    q = text("""
        SELECT f.form, f.is_earnings, d.full_date, f.url FROM gold.fact_filing f
        JOIN gold.dim_company c USING (company_key) JOIN gold.dim_date d USING (date_key)
        WHERE c.ticker = :t AND d.full_date >= (SELECT max(full_date) FROM gold.dim_date) - :days
        ORDER BY d.full_date DESC""")
    n = text("""
        SELECT n.title, n.domain, d.full_date FROM gold.fact_news n
        LEFT JOIN gold.dim_company c USING (company_key) JOIN gold.dim_date d USING (date_key)
        WHERE (c.ticker = :t OR n.scope = 'industry') AND n.tier <= :mt
        ORDER BY n.tier, n.seen_at DESC LIMIT 6""")
    with eng.connect() as con:
        filings = pd.DataFrame(con.execute(q, {"t": focal, "days": days}).mappings().all())
        news = pd.DataFrame(con.execute(n, {"t": focal, "mt": max_tier}).mappings().all())
    out = []
    if not filings.empty and filings["is_earnings"].any():
        e = filings[filings["is_earnings"]].iloc[0]
        out.append(Insight(
            id="news:earnings", type="news_event", title="New earnings release filed",
            summary=f"{focal} filed an earnings release (8-K, item 2.02) on {e['full_date']}; numbers in this "
                    f"dashboard may predate it until the 10-Q is filed.",
            facts=[], significance=0.7, impact=0.8, sources=[e["url"]]))
    if not news.empty:
        heads = "; ".join(f"\"{t[:90]}\" ({d})" for t, d in zip(news["title"].fillna(""), news["domain"].fillna("")))
        out.append(Insight(
            id="news:headlines", type="news_event", title="Recent headlines",
            summary=f"Latest headlines from major/trade outlets (untrusted text, context only): {heads}",
            facts=[], significance=0.3, impact=0.5, sources=["GDELT via gold.fact_news"]))
    return out


# =========================================================================== assemble
def score(ins: Insight, w: dict) -> float:
    return round(w["significance"] * ins.significance + w["relevance"] * ins.relevance + w["impact"] * ins.impact, 4)


def build_candidates(s: Settings, eng: Engine) -> list[Insight]:
    focal = s.focal_company
    cands = (market_insights(_read(eng, "market_snapshot"))
             + peer_insights(_read(eng, "peer_gap"), focal, {c.ticker: "; ".join(c.caveats) for c in s.companies})
             + trend_insights(_read(eng, "kpi_quarterly"), focal)
             + exposure_insights(_read(eng, "sensitivity"), focal)
             + news_insights(eng, focal, max_tier=s.news_filter.ai_max_tier))
    # pre-rank without relevance (unknown until the LLM judges it). The shortlist for round 1
    # keeps the best insight of EVERY type first (variety, as in LEVA), then fills by score.
    for c in cands:
        c.score = score(c, s.ai.weights)
    cands.sort(key=lambda c: c.score, reverse=True)
    best_per_type = list({c.type: c for c in reversed(cands)}.values())
    chosen = {c.id for c in best_per_type}
    rest = [c for c in cands if c.id not in chosen]
    shortlist = sorted(best_per_type, key=lambda c: c.score, reverse=True) + rest
    return shortlist[: max(s.ai.candidates, len(best_per_type))]


# --------------------------------------------------------------------------- facts for page briefs (round 3)
def valuation_insights(val: pd.DataFrame, focal: str) -> list[Insight]:
    """Share price, market value and multiples for every company (Peer Benchmark page)."""
    if val.empty or focal not in set(val["ticker"]):
        return []
    facts = []
    for r in val.itertuples():
        facts += [Fact(f"val.{r.ticker}.price", f"{r.ticker} share price", r.price, "USD/share"),
                  Fact(f"val.{r.ticker}.mcap", f"{r.ticker} market value", r.market_cap, "USD"),
                  Fact(f"val.{r.ticker}.ret1y", f"{r.ticker} share price change over 1 year", r.return_1y_pct, "%")]
        if pd.notna(r.pe):
            facts.append(Fact(f"val.{r.ticker}.pe", f"{r.ticker} price / earnings (last 4 quarters)", r.pe, "x"))
        if pd.notna(r.net_debt):
            facts.append(Fact(f"val.{r.ticker}.net_debt", f"{r.ticker} net debt", r.net_debt, "USD"))
    f = val[val["ticker"] == focal].iloc[0]
    return [Insight(id="valuation", type="valuation", title=f"Market view of {focal} vs peers",
                    summary=f"{focal} shares are {fmt(f['return_1y_pct'], '%')} over a year; market value {fmt(f['market_cap'], 'USD')}.",
                    facts=facts, significance=0.5, impact=0.5, sources=["gold.valuation (FMP prices, SEC shares)"])]


def scenario_insights(sc: pd.DataFrame, focal: str) -> list[Insight]:
    """Past episodes replayed on the company (What-if page)."""
    h = sc[(sc["scenario_type"] == "historical") & (sc["ticker"] == focal)] if not sc.empty else sc
    if h.empty:
        return []
    facts = []
    for i, r in enumerate(h.itertuples()):
        facts.append(Fact(f"scen.{i}.impact", f"{r.scenario}: estimated effect on {focal}'s quarterly operating income", r.impact_q, "USD"))
        if pd.notna(r.fuel_pct):
            facts.append(Fact(f"scen.{i}.fuel", f"{r.scenario}: jet fuel move", r.fuel_pct, "%"))
    names = ", ".join(h["scenario"].astype(str))
    return [Insight(id="scenarios", type="scenario", title="Past episodes replayed on today's quarter",
                    summary=f"Historical replays for {focal}: {names}.", facts=facts, significance=0.5, impact=0.6,
                    sources=["gold.scenario (historical)"])]


def driver_insights(drv: pd.DataFrame, focal: str) -> list[Insight]:
    """The calculated 'why' (gold.driver): one insight per KPI, its summary written by code.
    `group` carries the strength (mostly / partly / mixed / unchanged) and the main driver, so the
    checker can allow 'mostly because <driver>' only where the arithmetic supports it."""
    from dashboard.compute import driver_sentence
    if drv.empty or focal not in set(drv["ticker"]):
        return []
    out = []
    for kpi, g in drv[drv["ticker"] == focal].groupby("kpi", sort=False):
        g = g.sort_values("rank")
        r, unit = g.iloc[0], g.iloc[0]["unit"]
        kpi_unit = "%" if unit == "pp" else "USD"
        facts = [Fact(f"why.{kpi}.now", f"{focal} {r['label']} {r['quarter_label']}", r["value_now"], kpi_unit),
                 Fact(f"why.{kpi}.before", f"{focal} {r['label']} {r['compared_with']}", r["value_before"], kpi_unit),
                 Fact(f"why.{kpi}.change", f"{r['label']} change vs a year earlier", r["change"], unit)]
        facts += [Fact(f"why.{kpi}.{x.driver}", f"Part of the change from {x.driver_label}", x.contribution, unit)
                  for x in g[~g["noise"]].itertuples()]
        main = g[g["is_main"]]
        out.append(Insight(id=f"driver:{kpi}", type="driver", title=f"Why {r['label'].lower()} changed",
                           summary=driver_sentence(g), facts=facts, significance=0.8, impact=0.9,
                           group=f"{r['strength']}:{main.iloc[0]['driver'] if len(main) else ''}",
                           sources=["gold.driver (exact split of the change, same quarter a year earlier)"]))
    return out


def news_items(eng: Engine, focal: str, days: int = 30, limit: int = 8) -> list[dict]:
    """Headlines that PASSED the news test (silver), for this airline or the industry, newest first.
    The AI may only pick one by id; the dashboard line ('In the news: ...') is written by code."""
    q = text("""
        SELECT n.title, n.domain, d.full_date, n.topic, n.url FROM gold.fact_news n
        LEFT JOIN gold.dim_company c USING (company_key) JOIN gold.dim_date d USING (date_key)
        WHERE n.usable AND (c.ticker = :t OR n.scope = 'industry')
          AND d.full_date >= (SELECT max(full_date) FROM gold.dim_date) - :days
        ORDER BY d.full_date DESC, n.tier LIMIT 40""")
    try:
        with eng.connect() as con:
            rows = con.execute(q, {"t": focal, "days": days}).mappings().all()
    except Exception:                                    # gold built before the news test existed
        return []
    seen, out = set(), []
    for r in rows:
        key = (r["title"] or "").lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"n{len(out) + 1}", "title": r["title"], "outlet": outlet_name(r["domain"]),
                    "date": r["full_date"], "topic": r["topic"], "url": r["url"]})
        if len(out) >= limit:
            break
    return out


def outlet_name(domain: str) -> str:
    """reuters.com -> Reuters, finance.yahoo.com -> Yahoo Finance-style readable name."""
    known = {"wsj.com": "WSJ", "ft.com": "FT", "cnbc.com": "CNBC", "apnews.com": "AP", "bbc.com": "BBC",
             "cnn.com": "CNN", "npr.org": "NPR", "nytimes.com": "New York Times", "finance.yahoo.com": "Yahoo Finance",
             "businessinsider.com": "Business Insider", "aviationweek.com": "Aviation Week", "ainonline.com": "AIN"}
    d = (domain or "").lower().removeprefix("www.")
    return known.get(d, d.split(".")[0].replace("-", " ").title() if d else "News")


def news_line(item: dict) -> str:
    d = pd.Timestamp(item["date"])
    return f"In the news: {item['outlet']}, {d.day} {d.strftime('%b')}: \u201c{item['title']}\u201d"


def narrative_facts(s: Settings, eng: Engine, cands: list[Insight]) -> list[Insight]:
    """Everything the page briefs and recommendations may cite: the ranked candidates + market view + scenarios."""
    extra = valuation_insights(_read(eng, "valuation"), s.focal_company) + scenario_insights(_read(eng, "scenario"), s.focal_company)
    have = {c.id for c in cands}
    peers = [p for p in peer_insights(_read(eng, "peer_gap"), s.focal_company,
                                      {c.ticker: "; ".join(c.caveats) for c in s.companies})
             if p.id not in have and p.id in ("peer:operating_margin_pct", "peer:fuel_share_opex_pct")]
    try:
        drivers = driver_insights(_read(eng, "driver"), s.focal_company)
    except Exception:                      # compute not rerun since drivers were added
        drivers = []
    return cands + peers + extra + drivers   # the peer charts always have their own KPI as evidence
