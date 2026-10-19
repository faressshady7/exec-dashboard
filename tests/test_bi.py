"""Power BI views. Database test runs only with TEST_DATABASE_URL (never your real Supabase)."""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from dashboard.bi import VIEWS

DB = os.environ.get("TEST_DATABASE_URL")


def test_every_view_is_defined():
    assert {"kpi_latest", "kpi_trend", "market_today", "market_history", "whatif", "scenarios",
            "robustness", "assumptions", "news", "ai_brief", "freshness", "companies",
            "stock_history", "valuation", "attention"} == set(VIEWS)


@pytest.mark.skipif(not DB, reason="set TEST_DATABASE_URL to run database tests")
def test_views_build_on_gold_tables_and_survive_recompute():
    from sqlalchemy import create_engine, text
    from dashboard.ai import MockLLM, Round1, generate, store
    from dashboard.bi import create_views, drop_views
    from dashboard.compute import (compute_kpis, historical_moves, market_snapshot, past_ranges, peer_gaps,
                                   scenario_grid, sensitivities)
    from dashboard.config import load_settings
    from dashboard.gold import DDL, company_rows
    from dashboard.insights import Fact, Insight
    from test_compute import fin_frame, market_frame, quarters

    eng = create_engine(DB.replace("postgresql://", "postgresql+psycopg://"))
    s = load_settings()
    with eng.begin() as con:
        con.execute(text("DROP SCHEMA IF EXISTS bi CASCADE; DROP SCHEMA IF EXISTS gold CASCADE; DROP SCHEMA IF EXISTS ai CASCADE"))
        con.execute(text("CREATE SCHEMA gold"))
        for stmt in [x for x in DDL.split(";") if x.strip()]:
            con.execute(text(stmt))
        for r in company_rows(s):
            con.execute(text("INSERT INTO gold.dim_company VALUES (:company_key, :ticker, :cik, :name, :industry, :role, :sort_order)"), r)

    px = pd.Series(np.linspace(1.5, 3.0, 12), index=quarters(12))
    fin = pd.concat([fin_frame(t, "focal" if t == "UAL" else "peer", fuel_px=px.to_numpy())
                     for t in ("UAL", "DAL", "AAL")], ignore_index=True)
    mkt = market_frame(px)
    kpis = compute_kpis(fin, s.method)
    sens = sensitivities(fin, mkt, s).assign(robust_pct=100.0)
    ranges = past_ranges(mkt)
    tables = {"kpi_quarterly": kpis, "peer_gap": peer_gaps(kpis, s),
              "market_snapshot": pd.DataFrame({"code": ["jet_fuel"], "label": ["Jet fuel"], "latest": [3.0]}),
              "sensitivity": sens, "scenario": scenario_grid(sens, ranges, historical_moves(mkt, s)),
              "robustness": pd.DataFrame({"parameter": ["fuel_min_r2"], "same_result": [True]}),
              "past_range": pd.DataFrame([{"code": k, "low": lo, "high": hi} for k, (lo, hi) in ranges.items()])}
    from dashboard.valuation import stock_index, valuation
    from test_valuation import fin as vfin, prices as vprices, shares as vshares
    tables["stock_index"] = stock_index(vprices(), 3)
    tables["valuation"] = valuation(vprices(), vshares(), vfin(), s)
    from dashboard.attention import attention
    tables["attention"] = attention(kpis, tables["peer_gap"], market_snapshot(mkt), sens, tables["valuation"], s)
    tables["market_snapshot"] = market_snapshot(mkt)

    def write_all():
        with eng.begin() as con:
            drop_views(con)
            for name, df in tables.items():
                df.to_sql(name, con, schema="gold", if_exists="replace", index=False)
        return create_views(eng)

    created, skipped = write_all()
    assert set(skipped) == {"ai_brief"}                       # ai.brief does not exist yet
    created, skipped = write_all()                            # recompute: replace tables under the views
    assert set(skipped) == {"ai_brief"}

    cands = [Insight(id=f"t:{i}", type=f"type{i}", title=f"Insight {i}", summary="S",
                     facts=[Fact(f"f{i}", "fact", 10.0 + i, "%")], significance=0.5, impact=0.5) for i in range(3)]
    reply = lambda p, schema: ({"ranking": [{"id": c.id, "relevance": 0.5, "why": "x"} for c in cands]}
                               if schema is Round1 else {"headline": "H", "brief": "It is 11.0%.", "fact_ids_used": ["f1"]})
    store(eng, s, generate(s, cands, MockLLM(reply)))          # store() recreates the views

    with eng.connect() as con:
        views = set(con.execute(text("SELECT table_name FROM information_schema.views WHERE table_schema='bi'")).scalars())
        w = con.execute(text("SELECT ticker, fuel_impact_per_pct, fuel_past_high_pct, caveat FROM bi.whatif ORDER BY sort_order")).all()
        brief = con.execute(text("SELECT headline, check_passed FROM bi.ai_brief")).one()
        n_assump = con.execute(text("SELECT count(*) FROM bi.assumptions")).scalar()
        hist = con.execute(text("SELECT count(*) FROM bi.scenarios WHERE scenario_type = 'historical'")).scalar()
        val = con.execute(text("SELECT ticker, pe, caveat, pe_text, market_cap_text FROM bi.valuation ORDER BY sort_order")).all()
        kl = con.execute(text("SELECT ticker, status, value_text FROM bi.kpi_latest WHERE kpi = 'net_debt' ORDER BY sort_order")).all()
        att = con.execute(text("SELECT page, rank, severity, badge, box_bg FROM bi.attention ORDER BY page, rank")).all()
        wi2 = con.execute(text("SELECT fuel_reliability, fuel_impact_10pct_text FROM bi.whatif ORDER BY sort_order")).first()
        mt = con.execute(text("SELECT latest_text, position_label FROM bi.market_today")).first()
        idx0 = con.execute(text("SELECT min(index_100), count(DISTINCT ticker) FROM bi.stock_history "
                                "WHERE trade_date = (SELECT min(trade_date) FROM bi.stock_history)")).one()
    assert views == set(VIEWS)
    assert [r.ticker for r in w] == ["UAL", "DAL", "AAL"] and w[0].fuel_impact_per_pct == pytest.approx(-30e6)
    assert w[0].fuel_past_high_pct > 0 and "refinery" in w[1].caveat
    assert [r.ticker for r in val] == ["UAL", "DAL", "AAL"] and val[2].pe is None and "refinery" in val[1].caveat
    assert idx0 == (pytest.approx(100), 3)
    assert val[2].pe_text == "n/m" and val[0].market_cap_text.startswith("$") and val[0].market_cap_text.endswith("B")
    assert {r.status for r in kl} <= {"BETTER THAN PEERS", "WORSE THAN PEERS", "IN LINE"} and kl[0].value_text.endswith("B")
    assert {r.page for r in att} == {"brief", "peer", "risk"} and att[0].rank == 1
    assert any(r.severity == "MODERATE" and r.page == "peer" for r in att)       # Delta refinery caveat
    assert wi2.fuel_impact_10pct_text.startswith("-$") and wi2.fuel_reliability in ("RELIABLE", "INDICATIVE")
    assert mt.latest_text and mt.position_label
    assert brief == ("H", True) and n_assump == 4 and hist == 3   # only "Last 12 months, again" has data here: 1 per company
