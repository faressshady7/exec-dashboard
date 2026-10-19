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
    # Each company vs the AVERAGE OF THE OTHER companies in the same quarter (own_gap), with a status pill.
    # "In line" = within 2% of that average. Colours are the dashboard theme (text, background).
    "kpi_latest": """
        WITH x AS (
            SELECT c.name AS company, c.role, c.sort_order, pg.*,
                   (sum(pg.value) OVER w - pg.value) / NULLIF(count(*) OVER w - 1, 0) AS own_peer_avg
            FROM gold.peer_gap pg JOIN gold.dim_company c USING (ticker)
            WINDOW w AS (PARTITION BY pg.kpi))
        SELECT x.*, x.value - x.own_peer_avg AS own_gap,
               bi.fmt(x.value, x.unit) AS value_text, bi.fmt(x.own_peer_avg, x.unit) AS own_peer_avg_text,
               bi.fmt(x.peer_avg, x.unit) AS peer_avg_text,
               x.rank || ' of ' || x.n_companies AS rank_text,
               s.status, s.status_color, s.status_bg
        FROM x CROSS JOIN LATERAL (SELECT CASE
                 WHEN x.own_peer_avg IS NULL THEN 'NO COMPARISON'
                 WHEN abs(x.value - x.own_peer_avg) <= 0.02 * abs(x.own_peer_avg) THEN 'IN LINE'
                 WHEN (x.value > x.own_peer_avg) = (x.better = 'higher') THEN 'BETTER THAN THE OTHER TWO'
                 ELSE 'WORSE THAN THE OTHER TWO' END AS status) st
        CROSS JOIN LATERAL (SELECT st.status,
                 CASE st.status WHEN 'BETTER THAN THE OTHER TWO' THEN '#2E6B4F' WHEN 'WORSE THAN THE OTHER TWO' THEN '#A33A3A' ELSE '#5F6670' END AS status_color,
                 CASE st.status WHEN 'BETTER THAN THE OTHER TWO' THEN '#DCEBDF' WHEN 'WORSE THAN THE OTHER TWO' THEN '#F3D9D9' ELSE '#E6E2DC' END AS status_bg) s""",

    # KPI trend lines (last N years), unusual quarters marked
    "kpi_trend": f"""
        SELECT k.ticker, c.name AS company, c.role, k.period_end::date AS quarter_end, k.quarter_label,
               k.kpi, k.label, k.unit, k.better, k.value, k.unusual
        FROM gold.kpi_quarterly k JOIN gold.dim_company c USING (ticker)
        WHERE k.period_end >= current_date - interval '{YEARS} years'""",

    # Market panel: today's position (General / Industry slicer uses `scope`)
    "market_today": """
        SELECT m.*, v.scope,
               to_char(m.latest, 'FM999,990.00') AS latest_text,
               to_char(m.chg_1y, 'SGFM9990.0') || CASE m.change_type WHEN 'points' THEN ' points' ELSE '%' END || ' in 1 year' AS chg_1y_text,
               to_char(m.chg_1m, 'SGFM9990.0') || CASE m.change_type WHEN 'points' THEN ' points' ELSE '%' END || ' in 1 month' AS chg_1m_text,
               CASE WHEN m.percentile_10y >= 90 THEN 'VERY HIGH FOR 10 YEARS' WHEN m.percentile_10y >= 75 THEN 'HIGH FOR 10 YEARS' WHEN m.percentile_10y <= 10 THEN 'VERY LOW FOR 10 YEARS' WHEN m.percentile_10y <= 25 THEN 'LOW FOR 10 YEARS' ELSE 'NORMAL RANGE' END AS position_label,
               'Above ' || to_char(m.percentile_10y, 'FM990') || '% of the past 10 years' AS percentile_text,
               CASE WHEN m.percentile_10y >= 90 OR m.percentile_10y <= 10 THEN '#8A5A00'
                    WHEN m.percentile_10y >= 75 OR m.percentile_10y <= 25 THEN '#5F4B8B' ELSE '#5F6670' END AS position_color,
               CASE WHEN m.percentile_10y >= 90 OR m.percentile_10y <= 10 THEN '#FBEFD3'
                    WHEN m.percentile_10y >= 75 OR m.percentile_10y <= 25 THEN '#E9E3F3' ELSE '#E6E2DC' END AS position_bg
        FROM gold.market_snapshot m LEFT JOIN gold.dim_variable v USING (code)""",

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
               pr.low AS rate_past_low_bp,  pr.high AS rate_past_high_bp,
               f.impact_per_unit_q * 10   AS fuel_impact_10pct,
               r.impact_per_unit_q * 100  AS rate_impact_100bp,
               bi.fmt(f.impact_per_unit_q * 10, 'USD') AS fuel_impact_10pct_text,
               bi.fmt(r.impact_per_unit_q * 100, 'USD') AS rate_impact_100bp_text,
               to_char(f.r2, 'FM0.00') AS fuel_r2_text, to_char(r.r2, 'FM0.00') AS rate_r2_text,
               to_char(f.robust_pct, 'FM990') || '%' AS fuel_robust_text, to_char(r.robust_pct, 'FM990') || '%' AS rate_robust_text,
               bi.reliability(f.method, f.robust_pct) AS fuel_reliability, bi.reliability(r.method, r.robust_pct) AS rate_reliability,
               bi.pill_color(bi.reliability(f.method, f.robust_pct)) AS fuel_rel_color,
               bi.pill_bg(bi.reliability(f.method, f.robust_pct))    AS fuel_rel_bg,
               bi.pill_color(bi.reliability(r.method, r.robust_pct)) AS rate_rel_color,
               bi.pill_bg(bi.reliability(r.method, r.robust_pct))    AS rate_rel_bg
        FROM gold.dim_company c
        JOIN gold.sensitivity f ON f.ticker = c.ticker AND f.factor = 'jet_fuel'
        LEFT JOIN gold.sensitivity r ON r.ticker = c.ticker AND r.factor = 'fed_funds'
        LEFT JOIN gold.past_range pf ON pf.code = 'jet_fuel'
        LEFT JOIN gold.past_range pr ON pr.code = 'fed_funds'""",

    # Precomputed scenarios: sensitivity grid, combined, historical replays
    "scenarios": """
        SELECT c.name AS company, c.sort_order, s.*, bi.fmt(s.impact_q, 'USD') AS impact_text,
               CASE WHEN s.within_past_range THEN 'SEEN BEFORE' ELSE 'BIGGER THAN ANY PAST MOVE' END AS range_label,
               CASE WHEN s.within_past_range THEN '#2E6B4F' ELSE '#8A5A00' END AS range_color,
               CASE WHEN s.within_past_range THEN '#DCEBDF' ELSE '#FBEFD3' END AS range_bg
        FROM gold.scenario s JOIN gold.dim_company c USING (ticker)""",

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
    # AI brief per company (`company` = ticker): the frozen study brief if one exists, otherwise the newest
    "ai_brief": """
        SELECT DISTINCT ON (company) brief_id, created_at, company, company AS ticker, model, prompt_version, headline, brief,
               check_passed, fallback_used, wording_flags::text AS wording_flags, frozen,
               pages->>'page_briefing' AS page_briefing, pages->>'page_peers' AS page_peers,
               pages->>'page_risks' AS page_risks, pages->>'page_whatif' AS page_whatif,
               pages->>'page_detail' AS page_detail, narrative_passed, narrative_fallback
        FROM ai.brief ORDER BY company, frozen DESC, created_at DESC""",

    # AI options to consider, one row each (from the same brief as above)
    "ai_recommendations": """
        WITH b AS (SELECT DISTINCT ON (company) * FROM ai.brief ORDER BY company, frozen DESC, created_at DESC)
        SELECT b.company AS ticker, r.n::int AS rank, r.x->>'title' AS title, r.x->>'detail' AS detail,
               (r.x->>'page')::int AS page, b.narrative_passed, b.narrative_fallback
        FROM b CROSS JOIN LATERAL jsonb_array_elements(COALESCE(b.recommendations, '[]'::jsonb)) WITH ORDINALITY AS r(x, n)""",

    # Peer Benchmark: share prices rebased to 100 at the start of the window (empty until FMP_API_KEY is set)
    "stock_history": """
        SELECT c.ticker, c.name AS company, c.role, c.sort_order, i.trade_date::date AS trade_date,
               i.close, i.index_100
        FROM gold.stock_index i JOIN gold.dim_company c USING (ticker)""",

    # Peer Benchmark: market cap, EV and multiples, one row per company (+ the comparability caveat)
    "valuation": """
        SELECT c.name AS company, c.role, c.sort_order, v.*, w.caveat,
               '$' || to_char(v.price, 'FM9,990.00') AS price_text,
               bi.fmt(v.market_cap, 'USD') AS market_cap_text, bi.fmt(v.enterprise_value, 'USD') AS ev_text,
               bi.fmt(v.net_debt, 'USD') AS net_debt_text,
               COALESCE(to_char(v.pe, 'FM990.0') || 'x', 'not meaningful (loss)') AS pe_text,
               COALESCE(to_char(v.ev_revenue, 'FM990.00') || 'x', 'not available') AS ev_revenue_text,
               COALESCE(to_char(v.pb, 'FM990.0') || 'x', 'not meaningful (negative equity)') AS pb_text,
               to_char(v.return_1y_pct, 'SGFM9990.0') || '% in 1 year' AS return_1y_text,
               to_char(v.return_index_pct, 'SGFM9990.0') || '%' AS return_index_text,
               CASE WHEN v.return_1y_pct >= 0 THEN '#2E6B4F' ELSE '#A33A3A' END AS return_color,
               CASE WHEN v.return_1y_pct >= 0 THEN '#DCEBDF' ELSE '#F3D9D9' END AS return_bg,
               'LAST 4 QUARTERS TO ' || upper(to_char(v.ttm_end::date, 'DD Mon YY')) AS ttm_label,
               'BALANCE SHEET ' || upper(to_char(v.balance_date::date, 'DD Mon YY')) AS balance_label
        FROM gold.valuation v JOIN gold.dim_company c USING (ticker)
        LEFT JOIN (SELECT ticker, caveat FROM gold.sensitivity WHERE factor = 'jet_fuel') w USING (ticker)""",

    # Attention boxes (rule-based, see dashboard.attention), with the theme colours per severity
    "attention": """
        SELECT a.*, CASE a.severity WHEN 'HIGH' THEN 'HIGH ALERT' ELSE a.severity END AS badge,
               CASE a.severity WHEN 'HIGH' THEN '#F3D5D8' WHEN 'MODERATE' THEN '#FBEFD3' ELSE '#ECE8E2' END AS box_bg,
               CASE a.severity WHEN 'HIGH' THEN '#9E2A2B' WHEN 'MODERATE' THEN '#8A5A00' ELSE '#3A3F47' END AS title_color,
               CASE a.severity WHEN 'HIGH' THEN '#9E2A2B' WHEN 'MODERATE' THEN '#8A6200' ELSE '#5F6670' END AS badge_bg
        FROM gold.attention a""",

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


# Display helpers (text for cards and tables, so Power BI needs no formatting logic)
FUNCTIONS = [
    """CREATE OR REPLACE FUNCTION bi.fmt(v double precision, unit text) RETURNS text LANGUAGE sql IMMUTABLE AS $$
        SELECT CASE WHEN v IS NULL OR v = 'NaN'::float8 THEN '-'
            WHEN unit = 'USD' AND abs(v) >= 1e9 THEN CASE WHEN v < 0 THEN '-' ELSE '' END || '$' || to_char(abs(v) / 1e9, 'FM999,990.0') || 'B'
            WHEN unit = 'USD' THEN CASE WHEN v < 0 THEN '-' ELSE '' END || '$' || to_char(abs(v) / 1e6, 'FM999,990') || 'M'
            WHEN unit = 'x' THEN to_char(v, 'FM9990.0') || 'x'
            WHEN unit = '%' THEN to_char(v, 'FM9990.0') || '%'
            ELSE to_char(v, 'FM999,990.00') END $$""",
    """CREATE OR REPLACE FUNCTION bi.reliability(method text, robust double precision) RETURNS text LANGUAGE sql IMMUTABLE AS $$
        SELECT CASE WHEN method IS NULL OR method LIKE 'not%' THEN 'NO CLEAR LINK'
            WHEN method LIKE 'regression%' AND robust >= 80 THEN 'RELIABLE'
            ELSE 'ROUGH ESTIMATE' END $$""",
    """CREATE OR REPLACE FUNCTION bi.pill_color(label text) RETURNS text LANGUAGE sql IMMUTABLE AS $$
        SELECT CASE label WHEN 'RELIABLE' THEN '#2E6B4F' WHEN 'ROUGH ESTIMATE' THEN '#8A5A00' ELSE '#5F6670' END $$""",
    """CREATE OR REPLACE FUNCTION bi.pill_bg(label text) RETURNS text LANGUAGE sql IMMUTABLE AS $$
        SELECT CASE label WHEN 'RELIABLE' THEN '#DCEBDF' WHEN 'ROUGH ESTIMATE' THEN '#FBEFD3' ELSE '#E6E2DC' END $$""",
]


def create_views(eng: Engine) -> tuple[list[str], dict[str, str]]:
    """Create every view whose source tables exist. Returns (created, skipped->reason)."""
    created, skipped = [], {}
    with eng.begin() as con:
        con.execute(text("CREATE SCHEMA IF NOT EXISTS bi"))
        for f in FUNCTIONS:
            con.execute(text(f))
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
