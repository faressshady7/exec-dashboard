"""Share-price index and valuation multiples (no database)."""
import numpy as np
import pandas as pd
import pytest

from dashboard.config import load_settings
from dashboard.valuation import stock_index, valuation


def prices():
    days = pd.bdate_range("2022-10-03", "2026-10-16")
    rows = []
    for t, start, end in [("UAL", 40, 80), ("DAL", 30, 45), ("AAL", 15, 12)]:
        for d, v in zip(days, np.linspace(start, end, len(days))):
            rows.append({"ticker": t, "trade_date": d, "close": v})
    return pd.DataFrame(rows)


def fin():
    q = pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"])
    rows = []
    for t, ni, eq in [("UAL", 1e9, 12e9), ("DAL", 1.5e9, 15e9), ("AAL", -0.1e9, -4e9)]:
        for d in q:
            rows.append({"ticker": t, "period_end": d, "net_income": ni, "revenue": 15e9,
                         "long_term_debt": 25e9, "cash": 5e9, "equity": eq})
    return pd.DataFrame(rows)


def shares():
    return pd.DataFrame({"ticker": ["UAL", "UAL", "DAL", "AAL"],
                         "as_of": pd.to_datetime(["2026-04-16", "2026-11-01", "2026-07-09", "2026-07-20"]),
                         "shares": [300e6, 999e6, 650e6, 660e6]})


def test_index_starts_at_100_on_the_same_day():
    idx = stock_index(prices(), 3)
    first = idx.groupby("ticker").apply(lambda g: g.sort_values("trade_date").iloc[0])
    assert first["index_100"].tolist() == pytest.approx([100, 100, 100])
    assert first["trade_date"].nunique() == 1
    last = idx.sort_values("trade_date").groupby("ticker")["index_100"].last()
    assert last["UAL"] > 100 > last["AAL"]


def test_multiples_and_not_meaningful_cases():
    v = valuation(prices(), shares(), fin(), load_settings()).set_index("ticker")
    ual = v.loc["UAL"]
    assert ual["shares"] == 300e6                                 # the Nov share count is AFTER the price day
    assert ual["market_cap"] == pytest.approx(80 * 300e6)
    assert ual["enterprise_value"] == pytest.approx(80 * 300e6 + 20e9)
    assert ual["pe"] == pytest.approx(80 * 300e6 / 4e9)
    assert ual["ev_revenue"] == pytest.approx((24e9 + 20e9) / 60e9)
    assert np.isnan(v.loc["AAL", "pe"]) and np.isnan(v.loc["AAL", "pb"]) and "not meaningful" in v.loc["AAL", "note"]
    assert v.loc["AAL", "return_1y_pct"] < 0


def test_no_prices_gives_empty_tables():
    empty = pd.DataFrame(columns=["ticker", "trade_date", "close"])
    assert stock_index(empty, 3).empty and valuation(empty, shares(), fin(), load_settings()).empty
