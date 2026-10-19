"""Step 4 tests. Database tests run only with TEST_DATABASE_URL (never your real Supabase)."""
import json
import os
from datetime import date

import pytest

from dashboard.config import load_settings
from dashboard.gold import company_rows, date_rows, variable_rows


def test_companies_come_from_config_with_focal_first():
    rows = company_rows(load_settings())
    assert [r["ticker"] for r in rows] == ["UAL", "DAL", "AAL"]
    assert [r["role"] for r in rows] == ["focal", "peer", "peer"]


def test_variables_cover_financials_market_and_derived():
    s = load_settings()
    rows = variable_rows(s)
    codes = {r["code"]: r for r in rows}
    assert len(rows) == len(s.financial_metrics) + len(s.risk_factors) + len(s.derived_factors)
    assert codes["revenue"]["domain"] == "financial" and codes["capex"]["category"] == "cash_flow"
    assert codes["jet_fuel"]["source"] == "FRED DJFUELUSGULF"
    assert codes["term_spread"]["source"].startswith("derived")
    assert len({r["variable_key"] for r in rows}) == len(rows)          # unique keys


def test_date_dimension_labels():
    d = date_rows(date(2026, 9, 29), date(2026, 10, 1)).set_index("date_key")
    assert d.loc[20260930, "quarter_label"] == "Q3 2026" and d.loc[20260930, "is_quarter_end"]
    assert d.loc[20261001, "month_label"] == "Oct 2026" and not d.loc[20261001, "is_quarter_end"]


DB = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not DB, reason="set TEST_DATABASE_URL to run database tests")
def test_gold_build_from_silver(tmp_path):
    from sqlalchemy import create_engine, text
    from dashboard import silver
    from dashboard.gold import build, checks
    from test_silver import CASHFLOW_YTD

    eng = create_engine(DB.replace("postgresql://", "postgresql+psycopg://"))
    with eng.begin() as con:
        con.execute(text("DROP SCHEMA IF EXISTS silver CASCADE; DROP SCHEMA IF EXISTS gold CASCADE"))
    s = load_settings()
    b = tmp_path / "bronze"
    for series, rows in {"DGS10": "2026-10-15,4.10\n2026-10-16,4.20", "DGS1": "2026-10-15,3.60\n2026-10-16,3.65"}.items():
        (b / f"fred/{series}").mkdir(parents=True)
        (b / f"fred/{series}/2026-10-19.csv").write_text(f"observation_date,{series}\n{rows}\n")
    sec = b / "sec/companyfacts/UAL_0000100517"
    sec.mkdir(parents=True)
    (sec / "2026-10-19.json").write_text(json.dumps(CASHFLOW_YTD))
    for key in ("UAL", "DAL", "INDUSTRY"):
        (b / f"news/gdelt/{key}").mkdir(parents=True)
        (b / f"news/gdelt/{key}/2026-10-19.json").write_text(json.dumps({"articles": [
            {"url": "https://x/shared", "title": "United Airlines and Delta Air Lines face airline fuel squeeze",
             "domain": "reuters.com", "seendate": "20261018T121500Z"}]}))
    silver.run(s, eng, b)

    counts = build(s, eng, today=date(2026, 10, 19))
    assert counts["dim_company"] == 3 and counts["fact_financials"] == 5     # 4 quarters + FY
    assert counts["fact_news"] == 3                                          # same URL kept per company + industry
    with eng.connect() as con:
        spread = con.execute(text("""
            SELECT round(m.value::numeric, 2) FROM gold.fact_market m
            JOIN gold.dim_variable v USING (variable_key) WHERE v.code='term_spread' ORDER BY date_key""")).scalars().all()
        q4 = con.execute(text("""
            SELECT value, derived, fiscal_label FROM gold.fact_financials WHERE date_key=20251231 AND period_type='Q'""")).one()
    assert [float(x) for x in spread] == [0.50, 0.55]                        # 4.10-3.60, 4.20-3.65
    assert q4 == (180.0, True, "Q4 2025")
    problems = {name: n for name, n in checks(eng) if n}
    assert set(problems) <= {"market variables with no data", "companies with no financials"}  # tiny fixture

    assert build(s, eng, today=date(2026, 10, 19)) == counts                 # rebuild = identical
