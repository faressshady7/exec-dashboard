"""Step 4 - Gold layer: a Kimball star schema (fact constellation) built from silver.

Ng & Nagalingham (2023) built a star schema but never stated its grain. Every table
here states its grain - what ONE row means - because that is what makes the numbers
safe to add up, compare and chart.

Dimensions (shared by every fact table, built from settings.yaml = company-agnostic)
  dim_company   grain: one company in scope              (focal + peers)
  dim_date      grain: one calendar day                  (start_date -> today)
  dim_variable  grain: one measurable thing              (financial metric or market factor)

Facts
  fact_financials  grain: one company x one metric x one period (Q / FY / I) ending on a date
  fact_market      grain: one market variable x one observation date (incl. derived term spread)
  fact_filing      grain: one SEC filing (10-K / 10-Q / 8-K)
  fact_news        grain: one news article about one company (or the industry)
  fact_stock       grain: one company x one trading day (closing price, volume)
  fact_shares      grain: one company x one cover-page date (shares outstanding)

Gold is rebuilt from silver in one transaction (small tables, < 1 minute), so it always
matches silver exactly and a failed build leaves the old gold untouched.

Run:  python -m dashboard.gold
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboard.config import Settings, load_settings

GRAIN = {
    "dim_company": "one company in scope",
    "dim_date": "one calendar day",
    "dim_variable": "one financial metric or market variable",
    "fact_financials": "one company x metric x period type x period end",
    "fact_market": "one market variable x observation date",
    "fact_filing": "one SEC filing",
    "fact_news": "one kept article x company (or INDUSTRY), with source tier",
    "fact_stock": "one company x trading day (close, volume)",
    "fact_shares": "one company x cover-page date (shares outstanding)",
}

DDL = """
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.dim_company (
    company_key  SMALLINT PRIMARY KEY,
    ticker       TEXT UNIQUE NOT NULL,
    cik          INTEGER NOT NULL,
    name         TEXT NOT NULL,
    industry     TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('focal', 'peer')),
    sort_order   SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS gold.dim_date (
    date_key        INTEGER PRIMARY KEY,          -- yyyymmdd
    full_date       DATE UNIQUE NOT NULL,
    year            SMALLINT NOT NULL,
    quarter         SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    quarter_label   TEXT NOT NULL,                -- e.g. Q3 2026
    month_label     TEXT NOT NULL,                -- e.g. Oct 2026
    is_quarter_end  BOOLEAN NOT NULL,
    is_weekday      BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS gold.dim_variable (
    variable_key  SMALLINT PRIMARY KEY,
    code          TEXT UNIQUE NOT NULL,            -- e.g. revenue, jet_fuel, term_spread
    label         TEXT NOT NULL,
    domain        TEXT NOT NULL CHECK (domain IN ('financial', 'market')),
    category      TEXT NOT NULL,                   -- income / cash_flow / balance / cost / fx / rates ...
    unit          TEXT NOT NULL,
    frequency     TEXT NOT NULL,                   -- quarterly / daily / monthly
    source        TEXT NOT NULL,                   -- e.g. SEC XBRL, FRED DJFUELUSGULF, derived
    evidence      TEXT,
    scope         TEXT                             -- company (financials) | general (any firm) | industry
);
ALTER TABLE gold.dim_variable ADD COLUMN IF NOT EXISTS scope TEXT;

CREATE TABLE IF NOT EXISTS gold.fact_financials (
    company_key   SMALLINT NOT NULL REFERENCES gold.dim_company,
    variable_key  SMALLINT NOT NULL REFERENCES gold.dim_variable,
    date_key      INTEGER  NOT NULL REFERENCES gold.dim_date,      -- period end
    period_type   TEXT     NOT NULL CHECK (period_type IN ('Q', 'FY', 'I')),
    fiscal_label  TEXT,                                            -- e.g. Q3 2026, FY 2025
    value         DOUBLE PRECISION NOT NULL,                        -- USD
    derived       BOOLEAN  NOT NULL,                                -- TRUE = computed from YTD totals
    xbrl_tag      TEXT     NOT NULL,
    filed_date    DATE,
    PRIMARY KEY (company_key, variable_key, period_type, date_key)
);

CREATE TABLE IF NOT EXISTS gold.fact_market (
    variable_key  SMALLINT NOT NULL REFERENCES gold.dim_variable,
    date_key      INTEGER  NOT NULL REFERENCES gold.dim_date,
    value         DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (variable_key, date_key)
);

CREATE TABLE IF NOT EXISTS gold.fact_filing (
    accn             TEXT PRIMARY KEY,
    company_key      SMALLINT NOT NULL REFERENCES gold.dim_company,
    date_key         INTEGER  NOT NULL REFERENCES gold.dim_date,   -- filing date
    report_date_key  INTEGER  REFERENCES gold.dim_date,
    form             TEXT NOT NULL,
    is_earnings      BOOLEAN NOT NULL,
    items            TEXT,
    url              TEXT
);

CREATE TABLE IF NOT EXISTS gold.fact_news (
    news_id       BIGSERIAL PRIMARY KEY,
    company_key   SMALLINT REFERENCES gold.dim_company,             -- NULL = industry news
    date_key      INTEGER NOT NULL REFERENCES gold.dim_date,
    url           TEXT NOT NULL,
    title         TEXT,
    domain        TEXT,
    seen_at       TIMESTAMPTZ,
    scope         TEXT NOT NULL CHECK (scope IN ('company', 'industry')),
    tier          SMALLINT                                          -- 1 major outlet, 2 trade press, 3 other
);
ALTER TABLE gold.fact_news ADD COLUMN IF NOT EXISTS tier SMALLINT;
ALTER TABLE gold.fact_news ADD COLUMN IF NOT EXISTS topic TEXT;
ALTER TABLE gold.fact_news ADD COLUMN IF NOT EXISTS usable BOOLEAN;

CREATE TABLE IF NOT EXISTS gold.fact_stock (
    company_key  SMALLINT NOT NULL REFERENCES gold.dim_company,
    date_key     INTEGER  NOT NULL REFERENCES gold.dim_date,
    close        DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION,
    PRIMARY KEY (company_key, date_key)
);

CREATE TABLE IF NOT EXISTS gold.fact_shares (
    company_key  SMALLINT NOT NULL REFERENCES gold.dim_company,
    date_key     INTEGER  NOT NULL REFERENCES gold.dim_date,
    shares       DOUBLE PRECISION NOT NULL,
    filed        DATE,
    PRIMARY KEY (company_key, date_key)
);
"""

FINANCIAL_CATEGORY = {
    "revenue": "income", "operating_expenses": "income", "fuel_expense": "income",
    "operating_income": "income", "interest_expense": "income", "net_income": "income",
    "operating_cash_flow": "cash_flow", "capex": "cash_flow",
    "cash": "balance", "total_assets": "balance", "long_term_debt": "balance", "equity": "balance",
    "debt_current": "balance", "short_term_investments": "balance",
}


def ensure_tables(eng: Engine) -> None:
    with eng.begin() as con:
        for stmt in [x for x in DDL.split(";") if x.strip()]:
            con.execute(text(stmt))


# --------------------------------------------------------------------------- dimensions
def company_rows(s: Settings) -> list[dict]:
    return [{"company_key": i + 1, "ticker": c.ticker, "cik": c.cik, "name": c.name,
             "industry": c.industry, "role": "focal" if c.ticker == s.focal_company else "peer",
             "sort_order": i} for i, c in enumerate(s.in_scope)]


def variable_rows(s: Settings) -> list[dict]:
    rows = []
    for code, m in s.financial_metrics.items():
        rows.append({"code": code, "label": m.label, "domain": "financial",
                     "category": FINANCIAL_CATEGORY.get(code, "other"), "unit": "USD",
                     "frequency": "quarterly", "source": "SEC XBRL (10-K/10-Q)",
                     "evidence": "Company financial statements", "scope": "company"})
    for f in s.risk_factors:
        rows.append({"code": f.id, "label": f.label, "domain": "market", "category": f.category,
                     "unit": f.unit, "frequency": f.frequency, "source": f"FRED {f.series}",
                     "evidence": f.evidence, "scope": f.scope})
    for d in s.derived_factors:
        rows.append({"code": d.id, "label": d.label, "domain": "market", "category": "rates",
                     "unit": d.unit, "frequency": "daily", "source": f"derived: {d.formula}",
                     "evidence": d.evidence, "scope": d.scope})
    for i, r in enumerate(rows):
        r["variable_key"] = i + 1
    return rows


def date_rows(start: date, end: date) -> pd.DataFrame:
    d = pd.DataFrame({"full_date": pd.date_range(start, end, freq="D")})
    out = pd.DataFrame({
        "date_key": d.full_date.dt.strftime("%Y%m%d").astype(int),
        "full_date": d.full_date.dt.date,
        "year": d.full_date.dt.year, "quarter": d.full_date.dt.quarter, "month": d.full_date.dt.month,
        "quarter_label": "Q" + d.full_date.dt.quarter.astype(str) + " " + d.full_date.dt.year.astype(str),
        "month_label": d.full_date.dt.strftime("%b %Y"),
        "is_quarter_end": d.full_date.dt.is_quarter_end,
        "is_weekday": d.full_date.dt.dayofweek < 5,
    })
    return out


# --------------------------------------------------------------------------- facts (SQL)
FACT_SQL = {
    "fact_financials": """
        INSERT INTO gold.fact_financials
        SELECT c.company_key, v.variable_key, to_char(f.period_end, 'YYYYMMDD')::int, f.period_type,
               CASE f.period_type
                    WHEN 'Q'  THEN 'Q' || extract(quarter FROM f.period_end) || ' ' || extract(year FROM f.period_end)
                    WHEN 'FY' THEN 'FY ' || extract(year FROM f.period_end)
                    ELSE to_char(f.period_end, 'DD Mon YYYY') END,
               f.value, f.derived, f.xbrl_tag, f.filed
        FROM silver.sec_fact f
        JOIN gold.dim_company  c ON c.ticker = f.ticker
        JOIN gold.dim_variable v ON v.code = f.metric AND v.domain = 'financial'
        WHERE f.period_end BETWEEN :start AND :end""",
    "fact_market": """
        INSERT INTO gold.fact_market
        SELECT v.variable_key, to_char(o.obs_date, 'YYYYMMDD')::int, o.value
        FROM silver.fred_observation o
        JOIN gold.dim_variable v ON v.source = 'FRED ' || o.series_id
        WHERE o.obs_date BETWEEN :start AND :end""",
    "fact_filing": """
        INSERT INTO gold.fact_filing
        SELECT f.accn, c.company_key, to_char(f.filing_date, 'YYYYMMDD')::int,
               CASE WHEN f.report_date BETWEEN :start AND :end
                    THEN to_char(f.report_date, 'YYYYMMDD')::int END,
               f.form, f.is_earnings, f.items, f.url
        FROM silver.filing f JOIN gold.dim_company c ON c.ticker = f.ticker
        WHERE f.filing_date BETWEEN :start AND :end""",
    "fact_news": """
        INSERT INTO gold.fact_news (company_key, date_key, url, title, domain, seen_at, scope, tier, topic, usable)
        SELECT c.company_key, to_char((n.seen_at AT TIME ZONE 'UTC')::date, 'YYYYMMDD')::int,
               n.url, n.title, n.domain, n.seen_at,
               CASE WHEN n.company_key = 'INDUSTRY' THEN 'industry' ELSE 'company' END AS scope,
               n.tier, COALESCE(n.topic, 'other'), COALESCE(n.usable, FALSE)
        FROM silver.news_article n
        LEFT JOIN gold.dim_company c ON c.ticker = n.company_key
        WHERE n.seen_at IS NOT NULL AND (n.seen_at AT TIME ZONE 'UTC')::date BETWEEN :start AND :end
          AND n.kept IS TRUE                                   -- only filtered articles reach gold
          AND (c.company_key IS NOT NULL OR n.company_key = 'INDUSTRY')""",
    "fact_stock": """
        INSERT INTO gold.fact_stock
        SELECT c.company_key, to_char(p.trade_date, 'YYYYMMDD')::int, p.close, p.volume
        FROM silver.stock_price p JOIN gold.dim_company c ON c.ticker = p.ticker
        WHERE p.trade_date BETWEEN :start AND :end""",
    "fact_shares": """
        INSERT INTO gold.fact_shares
        SELECT c.company_key, to_char(x.as_of, 'YYYYMMDD')::int, x.shares, x.filed
        FROM silver.share_count x JOIN gold.dim_company c ON c.ticker = x.ticker
        WHERE x.as_of BETWEEN :start AND :end""",
}


def derived_market_sql(s: Settings) -> list[str]:
    """Derived factors, e.g. term_spread = yield_10y - yield_1y, on dates where both exist."""
    out = []
    for d in s.derived_factors:
        a, op, b = d.formula.split()
        out.append(f"""
            INSERT INTO gold.fact_market
            SELECT dv.variable_key, x.date_key, x.value {op} y.value
            FROM gold.fact_market x
            JOIN gold.dim_variable vx ON vx.variable_key = x.variable_key AND vx.code = '{a}'
            JOIN gold.fact_market y  ON y.date_key = x.date_key
            JOIN gold.dim_variable vy ON vy.variable_key = y.variable_key AND vy.code = '{b}'
            CROSS JOIN gold.dim_variable dv WHERE dv.code = '{d.id}'""")
    return out


# --------------------------------------------------------------------------- build
def build(s: Settings, eng: Engine, today: date | None = None) -> dict[str, int]:
    """Rebuild gold from silver in ONE transaction: all or nothing."""
    ensure_tables(eng)
    start, end = s.project.start_date, (today or s.project.end)
    params = {"start": start, "end": end}
    dates = date_rows(start, end)
    with eng.begin() as con:
        con.execute(text("TRUNCATE gold.fact_financials, gold.fact_market, gold.fact_filing, gold.fact_news, "
                         "gold.fact_stock, gold.fact_shares, "
                         "gold.dim_company, gold.dim_variable, gold.dim_date RESTART IDENTITY"))
        con.execute(text("INSERT INTO gold.dim_company VALUES (:company_key, :ticker, :cik, :name, :industry, :role, :sort_order)"),
                    company_rows(s))
        con.execute(text("INSERT INTO gold.dim_variable (variable_key, code, label, domain, category, unit, frequency, source, evidence, scope) "
                         "VALUES (:variable_key, :code, :label, :domain, :category, :unit, :frequency, :source, :evidence, :scope)"),
                    variable_rows(s))
        con.execute(text("INSERT INTO gold.dim_date VALUES (:date_key, :full_date, :year, :quarter, :month, "
                         ":quarter_label, :month_label, :is_quarter_end, :is_weekday)"),
                    dates.to_dict("records"))
        for sql in FACT_SQL.values():
            con.execute(text(sql), params)
        for sql in derived_market_sql(s):
            con.execute(text(sql))
        counts = {t: con.execute(text(f"SELECT count(*) FROM gold.{t}")).scalar() for t in GRAIN}
    return counts


FY_MISMATCH_SQL = """
    SELECT c.ticker, v.code AS metric, d.year,
           max(f.value) FILTER (WHERE period_type='FY')   AS fy,
           sum(f.value) FILTER (WHERE period_type='Q')    AS q_sum,
           count(DISTINCT f.xbrl_tag) FILTER (WHERE period_type='Q') AS q_tags
    FROM gold.fact_financials f
    JOIN gold.dim_date d USING (date_key)
    JOIN gold.dim_variable v USING (variable_key)
    JOIN gold.dim_company c USING (company_key)
    WHERE v.category IN ('income', 'cash_flow')
    GROUP BY 1, 2, 3
    HAVING count(*) FILTER (WHERE period_type='Q') = 4
       AND abs(max(f.value) FILTER (WHERE period_type='FY')) > 1e6
       AND abs(sum(f.value) FILTER (WHERE period_type='Q') - max(f.value) FILTER (WHERE period_type='FY'))
           > 0.01 * abs(max(f.value) FILTER (WHERE period_type='FY'))
    ORDER BY 1, 2, 3"""


def fy_mismatches(eng: Engine) -> pd.DataFrame:
    """Years where Q1+Q2+Q3+Q4 differs from the annual figure by more than 1%.
    1-5%: expected when a company RESTATES a year (e.g. the 2018 ASC 606 / ASU 2017-07
    adoption restated 2016-17) or switches tags mid-year - quarters stay as filed.
    >5%: treated as an error."""
    with eng.connect() as con:
        df = pd.DataFrame(con.execute(text(FY_MISMATCH_SQL)).mappings().all())
    if df.empty:
        return df
    df["diff_pct"] = (df["q_sum"] - df["fy"]) / df["fy"].abs() * 100
    df["reason"] = ["tag change mid-year" if t > 1 else "restated year (kept as filed)" for t in df["q_tags"]]
    df["level"] = ["INFO" if abs(p) <= 5 else "FAIL" for p in df["diff_pct"]]
    return df


def checks(eng: Engine) -> list[tuple[str, int]]:
    """Quality checks on gold. Each returns the number of PROBLEM rows (0 = pass)."""
    q = {
        "financial rows with no matching date": "SELECT count(*) FROM gold.fact_financials f LEFT JOIN gold.dim_date d USING (date_key) WHERE d.date_key IS NULL",
        "negative revenue quarters": "SELECT count(*) FROM gold.fact_financials f JOIN gold.dim_variable v USING (variable_key) WHERE v.code='revenue' AND f.value < 0",
        "market variables with no data": "SELECT count(*) FROM gold.dim_variable v WHERE v.domain='market' AND NOT EXISTS (SELECT 1 FROM gold.fact_market m WHERE m.variable_key=v.variable_key)",
        "companies with no financials": "SELECT count(*) FROM gold.dim_company c WHERE NOT EXISTS (SELECT 1 FROM gold.fact_financials f WHERE f.company_key=c.company_key)",
    }
    with eng.connect() as con:
        out = [(name, con.execute(text(sql)).scalar()) for name, sql in q.items()]
    mm = fy_mismatches(eng)
    out.insert(2, ("quarters do not add up to FY by >5%", int((mm["level"] == "FAIL").sum()) if not mm.empty else 0))
    return out


def main() -> None:
    from dashboard.db import database_url, engine, masked
    s = load_settings()
    print(f"Building gold in {masked(database_url())}")
    counts = build(s, engine())
    print(f"\n{'TABLE':<18}{'ROWS':>8}   GRAIN (what one row means)")
    for t, n in counts.items():
        print(f"gold.{t:<13}{n:>8}   {GRAIN[t]}")
    print("\nQUALITY CHECKS (0 = pass)")
    bad = 0
    for name, n in checks(engine()):
        bad += n > 0
        print(f"  {'PASS' if n == 0 else 'FAIL'}  {name}: {n}")
    mm = fy_mismatches(engine())
    info = mm[mm["level"] == "INFO"] if not mm.empty else mm
    if not info.empty:
        print(f"\n  INFO  {len(info)} years where Q1-Q4 differ from FY by 1-5% (kept exactly as filed):")
        for r in info.itertuples():
            print(f"        {r.ticker} {r.metric:<17} {r.year}  FY {r.fy/1e6:>8,.0f}M  sum of Q {r.q_sum/1e6:>8,.0f}M  "
                  f"({r.diff_pct:+.1f}%)  {r.reason}")
    print(f"\n{'All checks passed' if bad == 0 else f'{bad} check(s) failed - tell Claude'}")


if __name__ == "__main__":
    main()
