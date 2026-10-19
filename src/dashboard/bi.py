"""Power BI layer: clean, ready-to-use views in schema `bi`.

Power BI only READS these views (Get Data -> PostgreSQL -> schema bi). All logic stays in
Python: the views just join and rename, so the dashboard needs almost no DAX.

compute.run() drops the views first (it rebuilds its tables) and recreates them at the end;
ai.store() recreates them too, so the brief view is always current.

Run alone:  python -m dashboard.bi
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

YEARS = 3       # rolling window shown in the dashboard (matches project.display_years)

VIEWS: dict[str, str] = {
    "companies": """
        SELECT ticker, name AS company, role, sort_order, industry FROM gold.dim_company""",

    # KPI cards + peer comparison (latest quarter)
    "kpi_latest": """
        SELECT c.name AS company, c.role, c.sort_order, pg.*
        FROM gold.peer_gap pg JOIN gold.dim_company c USING (ticker)""",

    # KPI trend lines (last N years), unusual quarters marked
    "kpi_trend": f"""
        SELECT k.ticker, c.name AS company, c.role, k.period_end::date AS quarter_end, k.quarter_label,
               k.kpi, k.label, k.unit, k.better, k.value, k.unusual
        FROM gold.kpi_quarterly k JOIN gold.dim_company c USING (ticker)
        WHERE k.period_end >= current_date - interval '{YEARS} years'""",

    # Market panel: today's position (General / Industry slicer uses `scope`)
    "market_today": """
        SELECT m.*, v.scope FROM gold.market_snapshot m LEFT JOIN gold.dim_variable v USING (code)""",

    "market_history": f"""
        SELECT v.code, v.label, v.unit, v.category, v.scope, d.full_date AS obs_date, f.value
        FROM gold.fact_market f
        JOIN gold.dim_variable v USING (variable_key)
        JOIN gold.dim_date d USING (date_key)
        WHERE d.full_date >= current_date - interval '{YEARS} years'""",

    # What-if: one row per company with everything the sliders need (DAX: base + slope x slider)
    "whatif": """
        SELECT c.ticker, c.name AS company, c.role, c.sort_order,
               f.impact_per_unit_q          AS fuel_impact_per_pct,
               r.impact_per_unit_q          AS rate_impact_per_bp,
               f.base_operating_income_q, f.base_revenue_q, f.base_quarter::date AS base_quarter,
               f.method AS fuel_method, f.r2 AS fuel_r2, r.method AS rate_method, r.r2 AS rate_r2,
               f.robust_pct AS fuel_robust_pct, r.robust_pct AS rate_robust_pct, f.caveat,
               pf.low AS fuel_past_low_pct, pf.high AS fuel_past_high_pct,
               pr.low AS rate_past_low_bp,  pr.high AS rate_past_high_bp
        FROM gold.dim_company c
        JOIN gold.sensitivity f ON f.ticker = c.ticker AND f.factor = 'jet_fuel'
        LEFT JOIN gold.sensitivity r ON r.ticker = c.ticker AND r.factor = 'fed_funds'
        LEFT JOIN gold.past_range pf ON pf.code = 'jet_fuel'
        LEFT JOIN gold.past_range pr ON pr.code = 'fed_funds'""",

    # Precomputed scenarios: sensitivity grid, combined, historical replays
    "scenarios": """
        SELECT c.name AS company, c.sort_order, s.* FROM gold.scenario s JOIN gold.dim_company c USING (ticker)""",

    "robustness": """SELECT * FROM gold.robustness""",

    "assumptions": None,      # filled from compute.ASSUMPTIONS at creation time

    # News list with tier labels (last 30 days)
    "news": """
        SELECT f.news_id, COALESCE(c.ticker, 'INDUSTRY') AS ticker, COALESCE(c.name, 'Airline industry') AS company,
               f.scope, d.full_date AS published, f.title, f.domain, f.url, f.tier,
               CASE f.tier WHEN 1 THEN 'Major outlet' WHEN 2 THEN 'Trade press' ELSE 'Other source' END AS tier_label
        FROM gold.fact_news f
        LEFT JOIN gold.dim_company c USING (company_key)
        JOIN gold.dim_date d USING (date_key)
        WHERE d.full_date >= current_date - interval '30 days'""",

    # AI brief card: the frozen study brief if one exists, otherwise the newest
    "ai_brief": """
        SELECT brief_id, created_at, company, model, prompt_version, headline, brief,
               check_passed, fallback_used, wording_flags::text AS wording_flags, frozen
        FROM ai.brief ORDER BY frozen DESC, created_at DESC LIMIT 1""",

    # Peer Benchmark: share prices rebased to 100 at the start of the window (empty until FMP_API_KEY is set)
    "stock_history": """
        SELECT c.ticker, c.name AS company, c.role, c.sort_order, i.trade_date::date AS trade_date,
               i.close, i.index_100
        FROM gold.stock_index i JOIN gold.dim_company c USING (ticker)""",

    # Peer Benchmark: market cap, EV and multiples, one row per company (+ the comparability caveat)
    "valuation": """
        SELECT c.name AS company, c.role, c.sort_order, v.*, w.caveat
        FROM gold.valuation v JOIN gold.dim_company c USING (ticker)
        LEFT JOIN (SELECT ticker, caveat FROM gold.sensitivity WHERE factor = 'jet_fuel') w USING (ticker)""",

    # Header: "Data as of ..."
    "freshness": """
        SELECT v.label, v.scope, v.frequency, max(d.full_date) AS latest_date
        FROM gold.fact_market f JOIN gold.dim_variable v USING (variable_key) JOIN gold.dim_date d USING (date_key)
        GROUP BY v.label, v.scope, v.frequency""",
}


def drop_views(con) -> None:
    con.execute(text("DROP SCHEMA IF EXISTS bi CASCADE"))


def _assumptions_sql() -> str:
    from dashboard.compute import ASSUMPTIONS
    rows = ", ".join("({}, '{}')".format(i + 1, a.replace("'", "''")) for i, a in enumerate(ASSUMPTIONS))
    return f"SELECT * FROM (VALUES {rows}) AS t(n, assumption)"


def create_views(eng: Engine) -> tuple[list[str], dict[str, str]]:
    """Create every view whose source tables exist. Returns (created, skipped->reason)."""
    created, skipped = [], {}
    with eng.begin() as con:
        con.execute(text("CREATE SCHEMA IF NOT EXISTS bi"))
    for name, sql in VIEWS.items():
        sql = sql or _assumptions_sql()
        try:
            with eng.begin() as con:                 # one transaction per view: a missing table skips only that view
                con.execute(text(f"CREATE OR REPLACE VIEW bi.{name} AS {sql}"))
            created.append(name)
        except Exception as e:                       # e.g. ai.brief not created yet on a fresh database
            skipped[name] = str(e).splitlines()[0][:120]
    return created, skipped


def main() -> None:
    from dashboard.db import engine
    created, skipped = create_views(engine())
    print(f"Power BI views ready in schema bi: {', '.join(created)}")
    for k, v in skipped.items():
        print(f"  skipped bi.{k}: {v}")


if __name__ == "__main__":
    main()
