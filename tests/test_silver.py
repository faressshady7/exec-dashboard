"""Step 3 tests.

Pure-logic tests always run. Database tests run ONLY if TEST_DATABASE_URL is set,
so `pytest` on your laptop can never touch your real Supabase data.
"""
import json
import os
from datetime import date

import pandas as pd
import pytest

from dashboard.config import FinancialMetric, load_settings
from dashboard.silver import (build_sec_facts, extract_candidates, parse_filings, parse_fred,
                              parse_news, to_periods)


def fact(start, end, val, filed, fy=2025, fp="Q1", form="10-Q", accn="0001"):
    e = {"end": end, "val": val, "filed": filed, "fy": fy, "fp": fp, "form": form, "accn": accn}
    if start:
        e["start"] = start
    return e


def facts_json(tags: dict) -> dict:
    return {"entityName": "Test Air", "facts": {"us-gaap": {
        tag: {"units": {"USD": entries}} for tag, entries in tags.items()}}}


CF = FinancialMetric(label="Operating cash flow", kind="duration",
                     tags=["NetCashProvidedByUsedInOperatingActivities"])

# Cash flow is filed YEAR-TO-DATE only: Q1 (3M), H1 (6M), 9M, FY
CASHFLOW_YTD = facts_json({"NetCashProvidedByUsedInOperatingActivities": [
    fact("2025-01-01", "2025-03-31", 100, "2025-04-20", fp="Q1"),
    fact("2025-01-01", "2025-06-30", 250, "2025-07-20", fp="Q2"),
    fact("2025-01-01", "2025-09-30", 420, "2025-10-20", fp="Q3"),
    fact("2025-01-01", "2025-12-31", 600, "2026-02-15", fp="FY", form="10-K"),
]})


# ---------------------------------------------------------------- FRED
def test_fred_drops_missing_and_old_rows():
    raw = b"observation_date,VIXCLS\n2014-12-31,20\n2015-01-02,.\n2015-01-05,17.5\n"
    df = parse_fred(raw, "VIXCLS", date(2015, 1, 1))
    assert df.to_dict("records") == [{"series_id": "VIXCLS", "obs_date": date(2015, 1, 5), "value": 17.5}]


# ---------------------------------------------------------------- SEC quarters
def test_year_to_date_becomes_true_quarters():
    df = to_periods(extract_candidates(CASHFLOW_YTD, "operating_cash_flow", CF))
    q = df[df.period_type == "Q"].sort_values("period_end")
    assert list(q["value"]) == [100, 150, 170, 180]           # 250-100, 420-250, 600-420
    assert list(q["derived"]) == [False, True, True, True]
    assert list(q["fiscal_period"]) == ["Q1", "Q2", "Q3", "Q4"]
    assert (df.period_type == "FY").sum() == 1


def test_reported_quarter_is_preferred_over_derived():
    # income statements report the 3-month figure directly -> use it, don't derive
    j = facts_json({"Revenues": [
        fact("2025-01-01", "2025-03-31", 100, "2025-04-20"),
        fact("2025-04-01", "2025-06-30", 160, "2025-07-20", fp="Q2"),
        fact("2025-01-01", "2025-06-30", 999, "2025-07-20", fp="Q2"),   # YTD (inconsistent on purpose)
    ]})
    spec = FinancialMetric(label="Revenue", kind="duration", tags=["Revenues"])
    q = to_periods(extract_candidates(j, "revenue", spec))
    q2 = q[(q.period_type == "Q") & (q.period_end == pd.Timestamp("2025-06-30"))]
    assert len(q2) == 1 and q2.iloc[0]["value"] == 160 and not q2.iloc[0]["derived"]


def test_q4_from_three_quarters_when_no_9m_filed():
    j = facts_json({"Revenues": [
        fact("2025-01-01", "2025-03-31", 100, "2025-04-20"),
        fact("2025-04-01", "2025-06-30", 110, "2025-07-20"),
        fact("2025-07-01", "2025-09-30", 120, "2025-10-20"),
        fact("2025-01-01", "2025-12-31", 500, "2026-02-15", form="10-K", fp="FY"),
    ]})
    spec = FinancialMetric(label="Revenue", kind="duration", tags=["Revenues"])
    q = to_periods(extract_candidates(j, "revenue", spec))
    q4 = q[(q.period_type == "Q") & (q.period_end == pd.Timestamp("2025-12-31"))].iloc[0]
    assert q4["value"] == 170 and q4["derived"]


def test_restated_value_newest_filing_wins():
    j = facts_json({"Revenues": [
        fact("2025-01-01", "2025-03-31", 100, "2025-04-20"),
        fact("2025-01-01", "2025-03-31", 105, "2026-04-20", accn="0009"),   # restated a year later
    ]})
    spec = FinancialMetric(label="Revenue", kind="duration", tags=["Revenues"])
    df = extract_candidates(j, "revenue", spec)
    assert len(df) == 1 and df.iloc[0]["value"] == 105


def test_tag_switch_over_time_is_combined_and_first_tag_wins_on_overlap():
    j = facts_json({
        "Revenues": [fact("2016-01-01", "2016-03-31", 90, "2016-04-20")],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            fact("2016-01-01", "2016-03-31", 91, "2016-04-21"),            # same period, 2nd tag
            fact("2019-01-01", "2019-03-31", 120, "2019-04-20")],          # only the new tag
    })
    spec = FinancialMetric(label="Revenue", kind="duration",
                           tags=["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    df = extract_candidates(j, "revenue", spec).sort_values("period_end")
    assert list(df["value"]) == [90, 120]


def test_build_sec_facts_filters_before_start_and_keeps_instants():
    s = load_settings()
    j = facts_json({
        "Assets": [fact(None, "2014-12-31", 1, "2015-02-01", form="10-K"),
                   fact(None, "2025-12-31", 2, "2026-02-15", form="10-K")],
        "NetCashProvidedByUsedInOperatingActivities": CASHFLOW_YTD["facts"]["us-gaap"]
            ["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"],
    })
    out = build_sec_facts(j, "UAL", s)
    assets = out[out.metric == "total_assets"]
    assert list(assets["period_end"]) == [date(2025, 12, 31)] and assets.iloc[0]["period_type"] == "I"
    assert out[(out.metric == "operating_cash_flow") & (out.period_type == "Q")].shape[0] == 4


# ---------------------------------------------------------------- filings & news
def test_8k_earnings_release_is_flagged():
    raw = json.dumps({"filings": {"recent": {
        "form": ["8-K", "10-Q", "4"], "filingDate": ["2026-10-15", "2026-07-16", "2026-07-01"],
        "reportDate": ["2026-10-15", "2026-06-30", ""], "accessionNumber": ["a-1", "a-2", "a-3"],
        "items": ["2.02,9.01", "", ""], "primaryDocument": ["ex99.htm", "q2.htm", "f4.xml"]}}}).encode()
    df = parse_filings(raw, "UAL", 100517, date(2015, 1, 1))
    assert list(df["form"]) == ["8-K", "10-Q"]                     # Form 4 ignored
    assert df.iloc[0]["is_earnings"] and not df.iloc[1]["is_earnings"]
    assert df.iloc[0]["url"].endswith("/100517/a1/ex99.htm")


def test_news_dedupes_by_url():
    a = {"url": "https://x/1", "title": "T", "seendate": "20261018T121500Z", "domain": "x"}
    raw = json.dumps({"articles": [a, a]}).encode()
    df = parse_news(raw, "UAL")
    assert len(df) == 1 and str(df.iloc[0]["seen_at"]).startswith("2026-10-18 12:15")


# ---------------------------------------------------------------- database (opt-in)
DB = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not DB, reason="set TEST_DATABASE_URL to run database tests")
def test_full_load_is_idempotent(tmp_path):
    from sqlalchemy import create_engine, text
    from dashboard.silver import run

    eng = create_engine(DB.replace("postgresql://", "postgresql+psycopg://"))
    with eng.begin() as con:
        con.execute(text("DROP SCHEMA IF EXISTS silver CASCADE"))
    s = load_settings()
    b = tmp_path / "bronze"
    (b / "fred/VIXCLS").mkdir(parents=True)
    (b / "fred/VIXCLS/2026-10-19.csv").write_bytes(b"observation_date,VIXCLS\n2026-10-15,17.5\n2026-10-16,18\n")
    sec = b / "sec/companyfacts/UAL_0000100517"
    sec.mkdir(parents=True)
    (sec / "2026-10-19.json").write_text(json.dumps(CASHFLOW_YTD))
    news = b / "news/gdelt/UAL"
    news.mkdir(parents=True)
    (news / "2026-10-19.json").write_text(json.dumps({"articles": [
        {"url": "https://x/1", "title": "T", "seendate": "20261018T121500Z"}]}))

    first = run(s, eng, b)
    second = run(s, eng, b)                                     # re-run: no duplicates
    with eng.connect() as con:
        n_fred = con.execute(text("select count(*) from silver.fred_observation")).scalar()
        n_q = con.execute(text("select count(*) from silver.sec_fact where period_type='Q'")).scalar()
        n_news = con.execute(text("select count(*) from silver.news_article")).scalar()
    assert first["fred_observation"] == 2 and (n_fred, n_q, n_news) == (2, 4, 1)
    assert second["sec_fact"] == first["sec_fact"]


def test_minus_tag_flips_sign_and_is_only_a_fallback():
    j = facts_json({
        "InterestExpense": [fact("2017-01-01", "2017-03-31", 100, "2017-04-20")],
        "InterestIncomeExpenseNonoperatingNet": [
            fact("2017-01-01", "2017-03-31", -95, "2017-04-20"),     # same period: gross tag wins
            fact("2024-01-01", "2024-03-31", -150, "2024-04-20")],   # only net exists -> flipped
    })
    spec = FinancialMetric(label="Interest", kind="duration",
                           tags=["InterestExpense", "-InterestIncomeExpenseNonoperatingNet"])
    df = extract_candidates(j, "interest_expense", spec).sort_values("period_end")
    assert list(df["value"]) == [100, 150]
    assert list(df["xbrl_tag"]) == ["InterestExpense", "-InterestIncomeExpenseNonoperatingNet"]


def test_config_parses_minus_tags():
    tags = load_settings().financial_metrics["interest_expense"].tags
    assert "-InterestIncomeExpenseNet" in tags and tags[0] == "InterestExpenseNonoperating"


def test_plus_tag_sums_parts_only_when_all_present():
    j = facts_json({
        "PaymentsForFlightEquipment": [fact("2026-01-01", "2026-06-30", 2000, "2026-07-20"),
                                       fact("2026-01-01", "2026-09-30", 3000, "2026-10-20")],
        "PaymentsToAcquireOtherProductiveAssets": [fact("2026-01-01", "2026-06-30", 658, "2026-07-20")],
    })
    spec = FinancialMetric(label="Capex", kind="duration",
                           tags=["PaymentsForFlightEquipment+PaymentsToAcquireOtherProductiveAssets"])
    df = extract_candidates(j, "capex", spec)
    assert list(df["value"]) == [2658]          # 9M dropped: "other" part missing


def test_news_filter_tags_relevance_tier_block_and_duplicates():
    import pandas as pd
    from dashboard.silver import classify_news
    s = load_settings()
    ts = pd.Timestamp("2026-10-18", tz="UTC")
    df = pd.DataFrame([
        {"company_key": "UAL", "url": "u1", "title": "United Airlines raises fares - Reuters", "domain": "reuters.com", "seen_at": ts},
        {"company_key": "UAL", "url": "u2", "title": "United Airlines raises fares | KTVU", "domain": "ktvu.com", "seen_at": ts},
        {"company_key": "UAL", "url": "u3", "title": "Texas elections preview", "domain": "cnbc.com", "seen_at": ts},
        {"company_key": "UAL", "url": "u4", "title": "United Airlines (UAL) consensus target price", "domain": "tickerreport.com", "seen_at": ts},
        {"company_key": "UAL", "url": "u5", "title": "UAL stock moves", "domain": "skift.com", "seen_at": ts},
        {"company_key": "INDUSTRY", "url": "i1", "title": "Airlines scrap routes as jet fuel soars", "domain": "perthnow.com.au", "seen_at": ts},
    ])
    out = classify_news(df, s).set_index("url")
    assert out.loc["u1", "kept"] and out.loc["u1", "tier"] == 1
    assert out.loc["u2", "duplicate"] and not out.loc["u2", "kept"]          # same story, worse tier
    assert not out.loc["u3", "relevant"]                                      # name not in headline
    assert out.loc["u4", "blocked"] and not out.loc["u4", "kept"]            # stock-spam site
    assert out.loc["u5", "kept"] and out.loc["u5", "tier"] == 2              # ticker as a whole word
    assert out.loc["i1", "kept"] and out.loc["i1", "tier"] == 3


def test_news_audit_scores_precision_per_tier(tmp_path):
    import pandas as pd
    from dashboard.news_audit import score
    p = tmp_path / "audit.csv"
    pd.DataFrame({"relevant": ["y", "n", "Y", "yes", ""], "tier": [2, 2, 3, 3, 3],
                  "title": list("abcde")}).to_csv(p, index=False)
    res = score(p)
    assert res.loc[2, "precision"] == 0.5 and res.loc[3, "precision"] == 1.0
    assert res.loc["all", "rated"] == 4                 # blank answers are ignored
