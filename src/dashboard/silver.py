"""Step 3 - Silver layer: turn raw bronze files into clean, typed, de-duplicated tables.

Reads the NEWEST bronze file per source, writes to the `silver` schema in Supabase.
Loads are incremental (upsert): re-running never duplicates rows.

Cleaning rules (each one is a thesis decision you can defend):
  FRED   - missing values ('.') are dropped, never filled; one value per series per day;
           only the last 45 days are re-sent on later runs (FRED revises recent values).
  SEC    - only 10-K / 10-Q (and amendments) numbers are used.
         - a period is classed by length: 3 / 6 / 9 / 12 months, or an instant (balance sheet).
         - the same period appears in many filings (later reports repeat or restate it):
           the most recently FILED value wins.
         - companies switch XBRL tags over time: tags are tried in config order.
         - quarterly values: a reported 3-month figure is used when it exists; otherwise
           it is DERIVED from year-to-date totals (Q2 = 6M - Q1, Q3 = 9M - 6M, Q4 = FY - 9M).
           Cash-flow statements are only filed year-to-date, so they need this.
           Derived rows are flagged derived = TRUE.
  Filings- 10-K, 10-Q, 8-K (+ amendments) with their 8-K item codes
           (2.02 = results of operations, i.e. an earnings release).
  News   - one row per (company, article URL); the same article fetched on two days is kept once.
  Prices - one closing price per company per trading day (FMP; raw close, not dividend-adjusted).
  Shares - shares outstanding from each 10-K/10-Q cover page (dei:EntityCommonStockSharesOutstanding),
           dated at the cover date, newest filing wins. Used only for market cap.
           A cover date later than the filing date (a typo in the filing) is replaced by the filing date.

Run:  python -m dashboard.silver
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from dashboard.config import FinancialMetric, Settings, load_settings

FILING_FORMS = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"}
NUMBER_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
FRED_RESEND_DAYS = 45

DDL = """
CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE IF NOT EXISTS silver.fred_observation (
    series_id  TEXT    NOT NULL,
    obs_date   DATE    NOT NULL,
    value      DOUBLE PRECISION NOT NULL,
    loaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS silver.sec_fact (
    ticker        TEXT    NOT NULL,
    metric        TEXT    NOT NULL,
    period_type   TEXT    NOT NULL,          -- Q (quarter) | FY (year) | I (instant)
    period_end    DATE    NOT NULL,
    period_start  DATE,                      -- NULL for instants
    value         DOUBLE PRECISION NOT NULL,
    unit          TEXT    NOT NULL,
    xbrl_tag      TEXT    NOT NULL,
    fiscal_year   INTEGER,
    fiscal_period TEXT,
    form          TEXT,
    filed         DATE,
    accn          TEXT,
    derived       BOOLEAN NOT NULL DEFAULT FALSE,
    loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, metric, period_type, period_end)
);

CREATE TABLE IF NOT EXISTS silver.filing (
    accn         TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    form         TEXT NOT NULL,
    filing_date  DATE NOT NULL,
    report_date  DATE,
    items        TEXT,                       -- 8-K item codes, e.g. "2.02,9.01"
    is_earnings  BOOLEAN NOT NULL DEFAULT FALSE,
    url          TEXT,
    loaded_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS silver.news_article (
    company_key     TEXT NOT NULL,           -- ticker or INDUSTRY
    url             TEXT NOT NULL,
    title           TEXT,
    domain          TEXT,
    language        TEXT,
    source_country  TEXT,
    seen_at         TIMESTAMPTZ,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (company_key, url)
);

ALTER TABLE silver.news_article ADD COLUMN IF NOT EXISTS norm_title TEXT;
ALTER TABLE silver.news_article ADD COLUMN IF NOT EXISTS tier SMALLINT;          -- 1 major outlet, 2 trade press, 3 other
ALTER TABLE silver.news_article ADD COLUMN IF NOT EXISTS relevant BOOLEAN;       -- company/industry term in the headline
ALTER TABLE silver.news_article ADD COLUMN IF NOT EXISTS blocked BOOLEAN;        -- spam / content farm / state media
ALTER TABLE silver.news_article ADD COLUMN IF NOT EXISTS duplicate BOOLEAN;      -- same headline already kept
ALTER TABLE silver.news_article ADD COLUMN IF NOT EXISTS kept BOOLEAN;           -- relevant AND NOT blocked AND NOT duplicate

CREATE TABLE IF NOT EXISTS silver.stock_price (
    ticker      TEXT NOT NULL,
    trade_date  DATE NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION,
    loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS silver.share_count (
    ticker     TEXT NOT NULL,
    as_of      DATE NOT NULL,                -- cover-page date, usually a few weeks after quarter end
    shares     DOUBLE PRECISION NOT NULL,
    form       TEXT,
    filed      DATE,
    accn       TEXT,
    loaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, as_of)
);

CREATE TABLE IF NOT EXISTS silver.load_log (
    run_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    table_name   TEXT NOT NULL,
    rows_sent    INTEGER NOT NULL,
    source_files TEXT
);
"""


# =========================================================================== bronze files
def latest_file(folder: Path, pattern: str) -> Optional[Path]:
    """Bronze files are named YYYY-MM-DD.*, so the newest sorts last."""
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    return files[-1] if files else None


# =========================================================================== FRED
def parse_fred(raw: bytes, series: str, start: date) -> pd.DataFrame:
    rows = []
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    next(reader, None)
    for r in reader:
        if len(r) < 2 or r[1].strip() in ("", "."):
            continue
        d = date.fromisoformat(r[0].strip())
        if d >= start:
            rows.append((series, d, float(r[1])))
    df = pd.DataFrame(rows, columns=["series_id", "obs_date", "value"])
    return df.drop_duplicates(["series_id", "obs_date"], keep="last")


# =========================================================================== SEC numbers
def _months(start, end) -> Optional[int]:
    if pd.isna(start):
        return 0                                  # instant
    days = (end - start).days
    for m, (lo, hi) in {3: (80, 100), 6: (170, 195), 9: (260, 285), 12: (350, 380)}.items():
        if lo <= days <= hi:
            return m
    return None                                   # odd lengths (e.g. 2 months after a merger)


def _sum_tags(gaap: dict, parts: list[str]) -> list[dict]:
    """Add several tags period by period (newest filing of each). A period is kept only
    when EVERY part reports it, so a missing part never silently shrinks the total."""
    latest: list[dict] = []
    for part in parts:
        best: dict = {}
        for e in gaap.get(part, {}).get("units", {}).get("USD", []):
            if e.get("form") not in NUMBER_FORMS:
                continue
            k = (e.get("start"), e["end"])
            if k not in best or e["filed"] > best[k]["filed"]:
                best[k] = e
        latest.append(best)
    out = []
    for k in set.intersection(*(set(b) for b in latest)) if latest else []:
        rows = [b[k] for b in latest]
        newest = max(rows, key=lambda e: e["filed"])
        out.append({**newest, "val": sum(r["val"] for r in rows)})
    return out


def extract_candidates(facts: dict, metric: str, spec: FinancialMetric) -> pd.DataFrame:
    """All rows for one metric, best tag + newest filing per period."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for rank, raw_tag in enumerate(spec.tags):
        sign = -1 if raw_tag.startswith("-") else 1          # "-Tag" = flip the sign (net-income-style tags)
        tag = raw_tag.lstrip("-")
        if "+" in tag:                                        # "A+B" = sum of two tags for the same period
            entries, unit = _sum_tags(gaap, tag.split("+")), "USD"
        else:
            units = gaap.get(tag, {}).get("units", {})
            unit = "USD" if "USD" in units else next(iter(units), None)
            entries = units.get(unit, []) if unit else []
        for e in entries:
            if e.get("form") not in NUMBER_FORMS:
                continue
            rows.append({"metric": metric, "xbrl_tag": raw_tag, "rank": rank, "unit": unit,
                         "period_start": e.get("start"), "period_end": e["end"], "value": sign * e["val"],
                         "fiscal_year": e.get("fy"), "fiscal_period": e.get("fp"),
                         "form": e.get("form"), "filed": e.get("filed"), "accn": e.get("accn")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("period_start", "period_end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["months"] = [_months(s, e) for s, e in zip(df["period_start"], df["period_end"])]
    df = df[df["months"].notna()]
    df = df[(df["months"] == 0) == (spec.kind == "instant")]
    # best tag, then newest filing -> one row per exact period
    df = df.sort_values(["rank", "filed"], ascending=[True, False])
    df = df.drop_duplicates(["months", "period_start", "period_end"], keep="first")
    df = df.drop_duplicates(["months", "period_end"], keep="first")   # two tags, slightly different starts
    return df.drop(columns="rank")


def to_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Return Q / FY rows for a duration metric: reported quarters + quarters derived from YTD."""
    out = []
    direct_q = df[df["months"] == 3]
    have_q_end = set(direct_q["period_end"])
    for _, r in direct_q.iterrows():
        out.append({**r, "period_type": "Q", "derived": False})
    for _, r in df[df["months"] == 12].iterrows():
        out.append({**r, "period_type": "FY", "derived": False})

    # year-to-date chains share the fiscal-year start date: 3M, 6M, 9M, 12M
    for _, chain in df[df["months"].isin([3, 6, 9, 12])].groupby("period_start"):
        chain = chain.sort_values("months")
        prev = None
        for _, cur in chain.iterrows():
            if prev is not None and cur["months"] - prev["months"] == 3 and cur["period_end"] not in have_q_end:
                out.append({**cur, "period_type": "Q", "derived": True,
                            "period_start": prev["period_end"] + pd.Timedelta(days=1),
                            "value": cur["value"] - prev["value"],
                            "fiscal_period": {6: "Q2", 9: "Q3", 12: "Q4"}[cur["months"]]})
                have_q_end.add(cur["period_end"])
            prev = cur

    # fallback for Q4: FY minus three reported quarters inside it (no 9M filed)
    rows = pd.DataFrame(out) if out else pd.DataFrame()
    if not rows.empty:
        q = rows[rows["period_type"] == "Q"]
        for _, fy in rows[rows["period_type"] == "FY"].iterrows():
            if fy["period_end"] in have_q_end:
                continue
            inside = q[(q["period_start"] >= fy["period_start"]) & (q["period_end"] < fy["period_end"])]
            if len(inside) == 3:
                out.append({**fy, "period_type": "Q", "derived": True, "fiscal_period": "Q4",
                            "period_start": inside["period_end"].max() + pd.Timedelta(days=1),
                            "value": fy["value"] - inside["value"].sum()})
                have_q_end.add(fy["period_end"])
    return pd.DataFrame(out)


def build_sec_facts(facts: dict, ticker: str, s: Settings) -> pd.DataFrame:
    frames = []
    for metric, spec in s.financial_metrics.items():
        df = extract_candidates(facts, metric, spec)
        if df.empty:
            continue
        if spec.kind == "duration":
            df = to_periods(df)
        else:
            df = df.assign(period_type="I", derived=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["period_end"] >= pd.Timestamp(s.project.start_date)]
    out["ticker"] = ticker
    for c in ("period_start", "period_end", "filed"):
        out[c] = pd.to_datetime(out[c]).dt.date
    out["period_start"] = out["period_start"].where(out["period_type"] != "I", None)
    cols = ["ticker", "metric", "period_type", "period_end", "period_start", "value", "unit",
            "xbrl_tag", "fiscal_year", "fiscal_period", "form", "filed", "accn", "derived"]
    out = out[cols].drop_duplicates(["ticker", "metric", "period_type", "period_end"], keep="first")
    out["fiscal_year"] = out["fiscal_year"].astype("Int64")
    return out


# =========================================================================== filings / news
def parse_filings(raw: bytes, ticker: str, cik: int, start: date) -> pd.DataFrame:
    recent = json.loads(raw).get("filings", {}).get("recent", {})
    rows = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in FILING_FORMS:
            continue
        filed = date.fromisoformat(recent["filingDate"][i])
        if filed < start:
            continue
        accn = recent["accessionNumber"][i]
        items = (recent.get("items") or [""] * len(recent["form"]))[i] or None
        doc = (recent.get("primaryDocument") or [""] * len(recent["form"]))[i]
        rows.append({
            "accn": accn, "ticker": ticker, "form": form, "filing_date": filed,
            "report_date": date.fromisoformat(recent["reportDate"][i]) if recent["reportDate"][i] else None,
            "items": items, "is_earnings": bool(items and "2.02" in items.split(",")),
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{doc}" if doc else None,
        })
    return pd.DataFrame(rows)


def parse_news(raw: bytes, key: str) -> pd.DataFrame:
    text_ = raw.decode("utf-8", errors="replace").strip()
    articles = json.loads(text_).get("articles", []) if text_ else []
    rows = []
    for a in articles:
        if not a.get("url"):
            continue
        seen = a.get("seendate")
        rows.append({"company_key": key, "url": a["url"], "title": a.get("title"),
                     "domain": a.get("domain"), "language": a.get("language"),
                     "source_country": a.get("sourcecountry"),
                     "seen_at": datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                     if seen else None})
    return pd.DataFrame(rows).drop_duplicates(["company_key", "url"]) if rows else pd.DataFrame()


# =========================================================================== news filter
def parse_prices(raw: bytes, ticker: str, start: date) -> pd.DataFrame:
    """FMP daily prices -> (ticker, trade_date, close, volume), one row per day."""
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("historical", [])
    rows = [{"ticker": ticker, "trade_date": r["date"], "close": float(r["close"]),
             "volume": float(r["volume"]) if r.get("volume") is not None else None}
            for r in data if r.get("date") and r.get("close") is not None]
    df = pd.DataFrame(rows, columns=["ticker", "trade_date", "close", "volume"])
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[(df["trade_date"] >= start) & (df["close"] > 0)]
    return df.drop_duplicates(["ticker", "trade_date"], keep="last").reset_index(drop=True)


def parse_shares(facts: dict, ticker: str) -> pd.DataFrame:
    """Cover-page shares outstanding from 10-K/10-Q filings; newest filing wins per date."""
    units = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {})
    rows = [{"ticker": ticker, "as_of": e["end"], "shares": float(e["val"]), "form": e.get("form"),
             "filed": e.get("filed"), "accn": e.get("accn")}
            for e in units.get("shares", []) if e.get("form") in NUMBER_FORMS and e.get("val")]
    df = pd.DataFrame(rows, columns=["ticker", "as_of", "shares", "form", "filed", "accn"])
    if df.empty:
        return df
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    df["filed"] = pd.to_datetime(df["filed"]).dt.date
    # a cover date AFTER the filing date is a typo in the filing (found: AAL's Q2 2026 10-Q says
    # 17 Jul 2027 instead of 2026) -> use the filing date, which is never earlier than the true count date
    late = df["filed"].notna() & (df["as_of"] > df["filed"])
    df.loc[late, "as_of"] = df.loc[late, "filed"]
    df = df.sort_values("filed").drop_duplicates(["ticker", "as_of"], keep="last")
    return df.sort_values("as_of").reset_index(drop=True)


def _domain_in(domain: str, lst: list[str]) -> bool:
    d = (domain or "").lower().strip()
    return any(d == x or d.endswith("." + x) for x in lst)


def news_tier(domain: str, f) -> int:
    return 1 if _domain_in(domain, f.tier1) else 2 if _domain_in(domain, f.tier2) else 3


def headline_relevant(title: str, terms: list[str]) -> bool:
    t = (title or "").lower()
    for term in terms:
        if term.isupper() and len(term) <= 5:                       # ticker: whole word, case-sensitive
            if re.search(rf"\b{re.escape(term)}\b", title or ""):
                return True
        elif term.lower() in t:
            return True
    return False


def normalize_title(title: str) -> str:
    """'Delta unveils X - Reuters' and 'Delta unveils X | CNBC' -> 'delta unveils x' (for duplicates)."""
    t = re.sub(r"\s[-|–]\s[^-|–]{2,40}$", "", (title or "").strip())
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).replace("  ", " ").strip()


def classify_news(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Tag every article: tier, relevant, blocked, duplicate, kept. Duplicates keep the best
    tier first, then the earliest seen. Nothing is deleted - the funnel stays measurable."""
    f = s.news_filter
    df = df.copy()
    df["norm_title"] = df["title"].map(normalize_title)
    df["tier"] = df["domain"].map(lambda d: news_tier(d, f))
    df["blocked"] = df["domain"].map(lambda d: _domain_in(d, f.blocklist))
    df["relevant"] = [headline_relevant(t, f.headline_terms.get(k, [])) for t, k in zip(df["title"], df["company_key"])]
    df["duplicate"] = False
    cand = df[df["relevant"] & ~df["blocked"]].sort_values(["tier", "seen_at"])
    dup_idx = cand[cand.duplicated(["company_key", "norm_title"], keep="first")].index
    df.loc[dup_idx, "duplicate"] = True
    df["kept"] = df["relevant"] & ~df["blocked"] & ~df["duplicate"]
    return df


def refilter_news(eng: Engine, s: Settings) -> dict:
    """Re-tag ALL stored articles (so changing the lists in settings.yaml applies to history)."""
    with eng.connect() as con:
        df = pd.DataFrame(con.execute(text(
            "SELECT company_key, url, title, domain, seen_at FROM silver.news_article")).mappings().all())
    if df.empty:
        return {"total": 0}
    df = classify_news(df, s)
    rows = df[["company_key", "url", "norm_title", "tier", "relevant", "blocked", "duplicate", "kept"]]
    with eng.begin() as con:
        con.execute(text("""UPDATE silver.news_article SET norm_title=:norm_title, tier=:tier, relevant=:relevant,
                            blocked=:blocked, duplicate=:duplicate, kept=:kept
                            WHERE company_key=:company_key AND url=:url"""),
                    rows.astype(object).to_dict("records"))
    rel = df[df["relevant"]]
    return {"total": len(df), "relevant": int(df["relevant"].sum()),
            "not_blocked": int((df["relevant"] & ~df["blocked"]).sum()), "kept": int(df["kept"].sum()),
            "tier1": int((df["kept"] & (df["tier"] == 1)).sum()), "tier2": int((df["kept"] & (df["tier"] == 2)).sum()),
            "tier3": int((df["kept"] & (df["tier"] == 3)).sum()), "blocked_relevant": int((rel["blocked"]).sum())}


# =========================================================================== database
def ensure_tables(eng: Engine) -> None:
    with eng.begin() as con:
        for stmt in [x for x in DDL.split(";") if x.strip()]:
            con.execute(text(stmt))


def upsert(eng: Engine, table: str, df: pd.DataFrame, keys: list[str], chunk: int = 2000) -> int:
    """INSERT ... ON CONFLICT (keys) DO UPDATE: re-running a load never creates duplicates."""
    if df.empty:
        return 0
    schema, name = table.split(".")
    tbl = Table(name, MetaData(), schema=schema, autoload_with=eng)
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    with eng.begin() as con:
        for i in range(0, len(records), chunk):
            stmt = pg_insert(tbl).values(records[i:i + chunk])
            update = {c: stmt.excluded[c] for c in df.columns if c not in keys}
            update["loaded_at"] = text("now()")
            con.execute(stmt.on_conflict_do_update(index_elements=keys, set_=update))
    return len(records)


def _log(eng: Engine, table: str, n: int, files: list[str]) -> None:
    with eng.begin() as con:
        con.execute(text("INSERT INTO silver.load_log (table_name, rows_sent, source_files) VALUES (:t, :n, :f)"),
                    {"t": table, "n": n, "f": ", ".join(files)})


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================================================================== run
def run(s: Settings, eng: Engine, bronze: Path) -> dict:
    ensure_tables(eng)
    start = s.project.start_date
    report: dict = {}

    # ---- FRED (incremental: only recent rows after the first load)
    with eng.connect() as con:
        stored = {r[0]: (r[1], r[2]) for r in con.execute(text(
            "SELECT series_id, max(obs_date), count(*) FROM silver.fred_observation GROUP BY series_id")).all()}
    frames, files = [], []
    for rf in s.risk_factors:
        f = latest_file(bronze / "fred" / rf.series, "*.csv")
        if not f:
            continue
        df = parse_fred(f.read_bytes(), rf.series, start)
        # incremental: re-send only recent rows - but only if the stored history looks complete,
        # otherwise (e.g. an interrupted first load) reload the whole series
        if rf.series in stored and stored[rf.series][1] >= 0.98 * len(df):
            df = df[df["obs_date"] >= stored[rf.series][0] - timedelta(days=FRED_RESEND_DAYS)]
        frames.append(df)
        files.append(f.name)
    fred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report["fred_observation"] = upsert(eng, "silver.fred_observation", fred, ["series_id", "obs_date"])
    _log(eng, "fred_observation", report["fred_observation"], sorted(set(files)))

    # ---- SEC numbers + filings
    facts_frames, filing_frames, share_frames, files = [], [], [], []
    for c in s.in_scope:
        folder = f"{c.ticker}_{c.cik10}"
        f = latest_file(bronze / "sec" / "companyfacts" / folder, "*.json")
        if f:
            facts_json = json.loads(f.read_bytes())
            facts_frames.append(build_sec_facts(facts_json, c.ticker, s))
            share_frames.append(parse_shares(facts_json, c.ticker))
            files.append(f"{folder}/{f.name}")
        g = latest_file(bronze / "sec" / "submissions" / folder, "*.json")
        if g:
            filing_frames.append(parse_filings(g.read_bytes(), c.ticker, c.cik, start))
    facts = pd.concat([x for x in facts_frames if not x.empty], ignore_index=True) if facts_frames else pd.DataFrame()
    filings = pd.concat([x for x in filing_frames if not x.empty], ignore_index=True) if filing_frames else pd.DataFrame()
    report["sec_fact"] = upsert(eng, "silver.sec_fact", facts, ["ticker", "metric", "period_type", "period_end"])
    report["filing"] = upsert(eng, "silver.filing", filings, ["accn"])
    _log(eng, "sec_fact", report["sec_fact"], files)
    shares = _concat(share_frames)
    report["share_count"] = upsert(eng, "silver.share_count", shares, ["ticker", "as_of"])

    # ---- Share prices (only if bronze has them: needs FMP_API_KEY)
    frames, files = [], []
    for c in s.in_scope:
        f = latest_file(bronze / "prices" / "fmp" / c.ticker, "*.json")
        if f:
            frames.append(parse_prices(f.read_bytes(), c.ticker, start))
            files.append(f"{c.ticker}/{f.name}")
    prices = _concat(frames)
    report["stock_price"] = upsert(eng, "silver.stock_price", prices, ["ticker", "trade_date"])
    _log(eng, "stock_price", report["stock_price"], files)

    # ---- News
    frames = []
    for key in [c.ticker for c in s.in_scope] + ["INDUSTRY"]:
        f = latest_file(bronze / "news" / "gdelt" / key, "*.json")
        if f:
            frames.append(parse_news(f.read_bytes(), key))
    news = pd.concat([x for x in frames if not x.empty], ignore_index=True) if frames else pd.DataFrame()
    report["news_article"] = upsert(eng, "silver.news_article", news, ["company_key", "url"])
    report["news_funnel"] = refilter_news(eng, s)
    return report


def coverage(eng: Engine, s: Settings) -> pd.DataFrame:
    """Quality check: for every company x metric, how many quarters, from when to when, which tag."""
    q = text("""
        SELECT ticker, metric,
               count(*) FILTER (WHERE period_type IN ('Q','I'))            AS periods,
               count(*) FILTER (WHERE derived)                              AS derived,
               count(*) FILTER (WHERE period_type IN ('Q','I'))
                 - (count(DISTINCT date_trunc('quarter', period_end))
                    FILTER (WHERE period_type IN ('Q','I')))                AS dupes,
               ((extract(year FROM max(period_end) FILTER (WHERE period_type IN ('Q','I')))
                 - extract(year FROM min(period_end) FILTER (WHERE period_type IN ('Q','I')))) * 4
                + extract(quarter FROM max(period_end) FILTER (WHERE period_type IN ('Q','I')))
                - extract(quarter FROM min(period_end) FILTER (WHERE period_type IN ('Q','I'))) + 1)::int
                 - count(DISTINCT date_trunc('quarter', period_end))
                   FILTER (WHERE period_type IN ('Q','I'))                  AS gaps,
               min(period_end)::text AS first, max(period_end)::text AS last,
               string_agg(DISTINCT xbrl_tag, ' + ')                         AS tags
        FROM silver.sec_fact GROUP BY ticker, metric""")
    with eng.connect() as con:
        df = pd.DataFrame(con.execute(q).mappings().all())
    grid = pd.MultiIndex.from_product([[c.ticker for c in s.in_scope], list(s.financial_metrics)],
                                      names=["ticker", "metric"]).to_frame(index=False)
    return grid.merge(df, how="left", on=["ticker", "metric"])


def main() -> None:
    from dashboard.db import engine, masked, database_url
    s = load_settings()
    print(f"Loading silver into {masked(database_url())}")
    rep = run(s, engine(), s.data_path / "bronze")
    funnel = rep.pop("news_funnel", {})
    print("\nROWS SENT (upsert: new or updated)")
    for k, v in rep.items():
        print(f"  silver.{k:<18}{v:>7}")
    if funnel.get("total"):
        print("\nNEWS FILTER FUNNEL (all stored articles)")
        print(f"  downloaded                     {funnel['total']:>5}")
        print(f"  company/industry in headline   {funnel['relevant']:>5}  ({funnel['relevant'] / funnel['total']:.0%})")
        print(f"  not spam / state media         {funnel['not_blocked']:>5}")
        print(f"  unique (duplicates removed)    {funnel['kept']:>5}  -> tier 1: {funnel['tier1']}, "
              f"tier 2: {funnel['tier2']}, tier 3: {funnel['tier3']}")

    cov = coverage(engine(), s)
    print(f"\nFINANCIALS COVERAGE (quarters or balance-sheet dates since {s.project.start_date})")
    print(f"{'TICKER':<7}{'METRIC':<21}{'N':>4}{'DERIVED':>8}{'GAPS':>6}  {'FIRST':<11}{'LAST':<11}TAG")
    for r in cov.itertuples():
        if pd.isna(r.periods):
            print(f"{r.ticker:<7}{r.metric:<21}{'-':>4}{'':>8}{'':>6}  MISSING: no tag matched")
        else:
            flag = "" if r.gaps == 0 else "  <-- check"
            print(f"{r.ticker:<7}{r.metric:<21}{int(r.periods):>4}{int(r.derived):>8}{int(r.gaps):>6}  "
                  f"{r.first:<11}{r.last:<11}{r.tags}{flag}")
    missing = int(cov["periods"].isna().sum())
    gappy = int((cov["gaps"].fillna(0) > 0).sum())
    late = cov[cov["first"].notna() & (cov["first"].astype(str) > f"{s.project.start_date.year}-06-30")]
    print(f"\n{len(cov) - missing}/{len(cov)} pairs found | {gappy} with gaps inside | "
          f"{len(late)} starting after {s.project.start_date.year} Q2")
    print("Find a company's real tag names:  python -m dashboard.tags <TICKER> <word>")


if __name__ == "__main__":
    main()
