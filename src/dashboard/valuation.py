"""Market view of each company (Peer Benchmark page): share-price performance and valuation.

Inputs: gold.fact_stock (FMP daily closes), gold.fact_shares (SEC cover-page shares outstanding),
and the quarterly financials already used for the KPIs.

Definitions (standard, stated so an examiner can recompute them):
  index_100     close / close on the first common trading day of the window x 100 (window = stocks.index_years)
  market_cap    latest close x latest shares outstanding reported on or before that day
  net_debt      total debt (long-term + current, incl. finance leases; operating leases excluded)
                - (cash + short-term investments), latest balance sheet - same definition as the KPI card
  EV            market_cap + net_debt
  P/E           market_cap / net income of the last 4 quarters   (blank when that income is a loss)
  EV/Revenue    EV / revenue of the last 4 quarters
  P/B           market_cap / shareholders' equity                 (blank when equity is negative)
The last 4 quarters can end on different dates (one company has already reported, another not yet):
`ttm_end` and `balance_date` say which quarter each row uses, and the dashboard shows them.
Limitation: raw closes, not dividend-adjusted (Delta pays a small dividend; UAL and AAL do not).

Run alone:  python -m dashboard.valuation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboard.config import Settings, load_settings

INDEX_COLS = ["ticker", "trade_date", "close", "index_100"]
VALUATION_COLS = ["ticker", "price_date", "price", "shares", "shares_as_of", "market_cap", "total_debt", "liquidity",
                  "net_debt", "balance_date", "enterprise_value", "ttm_net_income", "ttm_revenue", "ttm_quarters",
                  "ttm_end", "equity",
                  "pe", "ev_revenue", "pb", "return_1m_pct", "return_1y_pct", "return_index_pct", "note"]


def load_stock(eng: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    with eng.connect() as con:
        prices = pd.DataFrame(con.execute(text("""
            SELECT c.ticker, d.full_date AS trade_date, f.close
            FROM gold.fact_stock f JOIN gold.dim_company c USING (company_key) JOIN gold.dim_date d USING (date_key)
        """)).mappings().all(), columns=["ticker", "trade_date", "close"])
        shares = pd.DataFrame(con.execute(text("""
            SELECT c.ticker, d.full_date AS as_of, f.shares
            FROM gold.fact_shares f JOIN gold.dim_company c USING (company_key) JOIN gold.dim_date d USING (date_key)
        """)).mappings().all(), columns=["ticker", "as_of", "shares"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    shares["as_of"] = pd.to_datetime(shares["as_of"])
    return prices, shares


def stock_index(prices: pd.DataFrame, years: int) -> pd.DataFrame:
    """Every company rebased to 100 on the SAME first trading day of the window."""
    if prices.empty:
        return pd.DataFrame(columns=INDEX_COLS)
    p = prices.sort_values(["ticker", "trade_date"])
    end = p["trade_date"].max()
    start = end - pd.DateOffset(years=years)
    w = p[p["trade_date"] >= start]
    first_common = w.groupby("ticker")["trade_date"].min().max()      # a day every company has traded since
    w = w[w["trade_date"] >= first_common].copy()
    base = w[w["trade_date"] == w.groupby("ticker")["trade_date"].transform("min")].set_index("ticker")["close"]
    w["index_100"] = w["close"] / w["ticker"].map(base) * 100
    return w[INDEX_COLS].reset_index(drop=True)


def _ret(series: pd.Series, end: pd.Timestamp, offset: pd.DateOffset) -> float:
    """% change from the last close on or before (end - offset) to the last close."""
    past = series[series.index <= end - offset]
    if past.empty:
        return np.nan
    return (series.iloc[-1] / past.iloc[-1] - 1) * 100


def _ttm(f: pd.DataFrame, col: str) -> tuple[float, int, object]:
    """Sum of the last 4 quarters that have a value, only if they are 4 consecutive quarters.
    Returns (sum, quarters found, end date of the last quarter)."""
    q = f.dropna(subset=[col]).sort_values("period_end").tail(4)
    end = q["period_end"].iloc[-1].date() if len(q) else None
    if len(q) < 4 or (q["period_end"].iloc[-1] - q["period_end"].iloc[0]).days > 300:
        return np.nan, len(q), end
    return float(q[col].sum()), 4, end


def _balance(f: pd.DataFrame) -> tuple[float, float, object]:
    """(total debt, liquidity, date) from the latest quarter where both are known."""
    from dashboard.compute import liquidity, total_debt
    if f.empty:
        return np.nan, np.nan, None
    b = f.assign(_debt=total_debt(f), _liq=liquidity(f)).dropna(subset=["_debt", "_liq"]).sort_values("period_end")
    if b.empty:
        return np.nan, np.nan, None
    r = b.iloc[-1]
    return float(r["_debt"]), float(r["_liq"]), r["period_end"].date()


def _latest(f: pd.DataFrame, col: str) -> float:
    v = f.dropna(subset=[col]).sort_values("period_end")
    return float(v[col].iloc[-1]) if len(v) else np.nan


def valuation(prices: pd.DataFrame, shares: pd.DataFrame, fin: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=VALUATION_COLS)
    rows = []
    for c in s.in_scope:
        px = prices[prices["ticker"] == c.ticker].sort_values("trade_date").set_index("trade_date")["close"]
        if px.empty:
            continue
        day, price = px.index[-1], float(px.iloc[-1])
        sh = shares[(shares["ticker"] == c.ticker) & (shares["as_of"] <= day)].sort_values("as_of")
        n_shares = float(sh["shares"].iloc[-1]) if len(sh) else np.nan
        f = fin[fin["ticker"] == c.ticker] if "ticker" in fin else fin.iloc[0:0]
        get = lambda col: _latest(f, col) if col in f else np.nan
        debt, liq, bal_date = _balance(f)
        net_debt = debt - liq
        ni, nq, ttm_end = _ttm(f, "net_income") if "net_income" in f else (np.nan, 0, None)
        rev, _, _ = _ttm(f, "revenue") if "revenue" in f else (np.nan, 0, None)
        equity = get("equity")
        mcap = price * n_shares
        ev = mcap + net_debt
        notes = []
        if not (ni > 0):
            notes.append("P/E not meaningful (loss or missing income over the last 4 quarters)")
        if not (equity > 0):
            notes.append("P/B not meaningful (negative or missing equity)")
        rows.append({
            "ticker": c.ticker, "price_date": day.date(), "price": price,
            "shares": n_shares, "shares_as_of": sh["as_of"].iloc[-1].date() if len(sh) else None,
            "market_cap": mcap, "total_debt": debt, "liquidity": liq, "net_debt": net_debt,
            "balance_date": bal_date, "enterprise_value": ev,
            "ttm_net_income": ni, "ttm_revenue": rev, "ttm_quarters": nq, "ttm_end": ttm_end, "equity": equity,
            "pe": mcap / ni if ni > 0 else np.nan,
            "ev_revenue": ev / rev if rev > 0 else np.nan,
            "pb": mcap / equity if equity > 0 else np.nan,
            "return_1m_pct": _ret(px, day, pd.DateOffset(months=1)),
            "return_1y_pct": _ret(px, day, pd.DateOffset(years=1)),
            "return_index_pct": _ret(px, day, pd.DateOffset(years=s.stocks.index_years)),
            "note": "; ".join(notes),
        })
    return pd.DataFrame(rows, columns=VALUATION_COLS)


def main() -> None:
    from dashboard.compute import load_financials
    from dashboard.db import engine
    s = load_settings()
    eng = engine()
    prices, shares = load_stock(eng)
    if prices.empty:
        raise SystemExit("No share prices in gold yet: set FMP_API_KEY, then run bronze, silver and gold.")
    v = valuation(prices, shares, load_financials(eng), s)
    print(f"VALUATION ({v['price_date'].max()})")
    print(f"  {'TICKER':<7}{'PRICE':>9}{'MKT CAP':>10}{'DEBT':>9}{'LIQUID':>9}{'NET DEBT':>10}{'EV':>9}"
          f"{'P/E':>7}{'EV/REV':>8}{'P/B':>6}{'1Y':>8}  LAST 4Q TO")
    for r in v.itertuples():
        f = lambda x, d=1: "-" if pd.isna(x) else f"{x:.{d}f}"
        b = lambda x: "-" if pd.isna(x) else f"{x / 1e9:.1f}B"
        print(f"  {r.ticker:<7}{r.price:>9.2f}{b(r.market_cap):>10}{b(r.total_debt):>9}{b(r.liquidity):>9}"
              f"{b(r.net_debt):>10}{b(r.enterprise_value):>9}{f(r.pe):>7}{f(r.ev_revenue, 2):>8}{f(r.pb):>6}"
              f"{f(r.return_1y_pct):>7}%  {r.ttm_end}")


if __name__ == "__main__":
    main()
