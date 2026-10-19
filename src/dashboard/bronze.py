"""Step 2 - Bronze layer: download raw data and save it UNTOUCHED.

Sources (all free, no API key):
  FRED  - one CSV per risk factor in settings.yaml
  SEC   - companyfacts JSON (every XBRL number from 10-K / 10-Q filings)
        - submissions JSON (filing history: latest 10-Q / 10-K, and 8-K announcements)
  GDELT - news articles per company + one industry query (last N days)

Rules of the bronze layer
  1. Never edit raw data. Files are saved exactly as received.
  2. Every download is logged in _manifest.jsonl (what, when, from where, checksum).
  3. One failing source never stops the others; it is logged as FAILED.
  4. A freshness report says, for every source, how old its latest value is.

Run:  python -m dashboard.bronze            (skips files already fetched today)
      python -m dashboard.bronze --force    (re-downloads)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from dashboard.config import Settings, load_settings

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
GDELT_URL = ("https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist"
             "&format=json&maxrecords={n}&sort=DateDesc&timespan={days}d")
FILING_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}


# --------------------------------------------------------------------------- HTTP
class Fetcher:
    """Small HTTP client: polite to SEC (User-Agent, <=10 requests/s) and retries."""

    def __init__(self, user_agent: str, session: Optional[requests.Session] = None,
                 retries: int = 3, backoff: float = 2.0, min_interval: float = 0.15):
        self.session = session or requests.Session()
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.retries, self.backoff, self.min_interval = retries, backoff, min_interval
        self._last = 0.0

    def get(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, headers=self.headers, timeout=60)
                if r.status_code == 200:
                    return r.content
                last_error = RuntimeError(f"HTTP {r.status_code} for {url}")
                if r.status_code not in (429, 500, 502, 503, 504):
                    break                                  # 403/404: retrying won't help
            except requests.RequestException as e:
                last_error = e
            time.sleep(self.backoff * (attempt + 1))
        raise last_error or RuntimeError(f"failed: {url}")


# --------------------------------------------------------------------------- results
@dataclass
class FetchResult:
    source: str            # fred | sec_facts | sec_submissions
    key: str               # series id or ticker
    status: str            # OK | SKIPPED | FAILED
    url: str = ""
    path: str = ""
    bytes: int = 0
    sha256: str = ""
    latest: Optional[str] = None       # date of the newest observation / filing
    frequency: str = ""                # daily | monthly | quarterly | filings
    detail: str = ""                   # rows, last form, or the error message
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


# --------------------------------------------------------------------------- parsers
def parse_fred_csv(raw: bytes, series: str) -> tuple[int, Optional[str]]:
    """Return (number of non-missing rows, latest date). FRED marks missing values with '.'."""
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader, None)
    if not header or len(header) < 2 or header[1].strip() != series:
        raise ValueError(f"unexpected FRED header for {series}: {header}")
    rows, latest = 0, None
    for row in reader:
        if len(row) >= 2 and row[1].strip() not in ("", "."):
            rows, latest = rows + 1, row[0].strip()
    if rows == 0:
        raise ValueError(f"FRED returned no values for {series}")
    return rows, latest


def parse_sec_facts(raw: bytes) -> tuple[int, str]:
    data = json.loads(raw)
    if "facts" not in data:
        raise ValueError("SEC companyfacts payload has no 'facts' key")
    n = sum(len(tags) for tags in data["facts"].values())
    return n, data.get("entityName", "")


def parse_sec_latest_filing(raw: bytes) -> tuple[Optional[str], str]:
    """Return (filing date, 'FORM for period REPORTDATE') of the newest 10-K/10-Q."""
    recent = json.loads(raw).get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    n_8k = sum(f == "8-K" for f in forms[:40])
    for i, form in enumerate(forms):            # SEC lists newest first
        if form in FILING_FORMS:
            return (recent["filingDate"][i],
                    f"{form} for period {recent['reportDate'][i]} · {n_8k} recent 8-Ks")
    return None, "no 10-K/10-Q found"


def gdelt_query(terms: list[str], language: str) -> str:
    """["United Airlines"] -> '"United Airlines" sourcelang:english'; several terms are OR-ed."""
    phrases = [f'"{t}"' for t in terms]
    core = phrases[0] if len(phrases) == 1 else "(" + " OR ".join(phrases) + ")"
    return f"{core} sourcelang:{language}"


def parse_gdelt(raw: bytes) -> tuple[int, Optional[str]]:
    """Return (number of articles, date of the newest one). GDELT answers its
    rate limit with plain text instead of JSON, which raises here."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return 0, None                                   # no articles in the window
    try:
        articles = json.loads(text).get("articles", [])
    except json.JSONDecodeError:
        raise ValueError(f"GDELT did not return JSON: {text[:80]}")
    dates = [a["seendate"][:8] for a in articles if a.get("seendate")]   # 20261018T121500Z
    latest = max(dates) if dates else None
    return len(articles), (f"{latest[:4]}-{latest[4:6]}-{latest[6:]}" if latest else None)


# --------------------------------------------------------------------------- saving
def _save(out_dir: Path, rel: str, raw: bytes) -> tuple[str, str]:
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return str(path.relative_to(out_dir)), hashlib.sha256(raw).hexdigest()


def _already_fetched(out_dir: Path, rel: str, force: bool) -> bool:
    return (out_dir / rel).exists() and not force


# --------------------------------------------------------------------------- steps
def fetch_fred(s: Settings, f: Fetcher, out_dir: Path, today: str, force=False) -> list[FetchResult]:
    results = []
    for rf in s.risk_factors:
        rel = f"fred/{rf.series}/{today}.csv"
        url = FRED_URL.format(series=rf.series, start=s.project.start_date.isoformat())
        res = FetchResult("fred", rf.series, "OK", url=url, frequency=rf.frequency)
        try:
            if _already_fetched(out_dir, rel, force):
                raw, res.status = (out_dir / rel).read_bytes(), "SKIPPED"
            else:
                raw = f.get(url)
            rows, res.latest = parse_fred_csv(raw, rf.series)      # validate BEFORE saving
            if res.status == "OK":
                res.path, res.sha256 = _save(out_dir, rel, raw)
            else:
                res.path = rel
            res.bytes, res.detail = len(raw), f"{rows} values"
        except Exception as e:                                       # one failure != stop
            res.status, res.detail = "FAILED", str(e)
        results.append(res)
    return results


def fetch_sec(s: Settings, f: Fetcher, out_dir: Path, today: str, force=False) -> list[FetchResult]:
    results = []
    for c in s.in_scope:
        folder = f"{c.ticker}_{c.cik10}"
        jobs = [
            ("sec_facts", SEC_FACTS_URL, f"sec/companyfacts/{folder}/{today}.json"),
            ("sec_submissions", SEC_SUBMISSIONS_URL, f"sec/submissions/{folder}/{today}.json"),
        ]
        for source, tmpl, rel in jobs:
            url = tmpl.format(cik10=c.cik10)
            res = FetchResult(source, c.ticker, "OK", url=url)
            try:
                if _already_fetched(out_dir, rel, force):
                    raw, res.status, res.path = (out_dir / rel).read_bytes(), "SKIPPED", rel
                else:
                    raw = f.get(url)
                if source == "sec_facts":
                    n, name = parse_sec_facts(raw)
                    res.detail = f"{n} XBRL tags · {name}"
                else:
                    res.latest, res.detail = parse_sec_latest_filing(raw)
                    res.frequency = "filings"
                if res.status == "OK":
                    res.path, res.sha256 = _save(out_dir, rel, raw)
                res.bytes = len(raw)
            except Exception as e:
                res.status, res.detail = "FAILED", str(e)
            results.append(res)
    return results


def fetch_news(s: Settings, f: Fetcher, out_dir: Path, today: str, force=False) -> list[FetchResult]:
    """One GDELT query per company (+ one for the industry). Raw JSON saved as received."""
    jobs = [(c.ticker, c.search_terms) for c in s.in_scope]
    if s.news.industry_terms:
        jobs.append(("INDUSTRY", s.news.industry_terms))
    results = []
    for key, terms in jobs:
        rel = f"news/gdelt/{key}/{today}.json"
        url = GDELT_URL.format(query=quote(gdelt_query(terms, s.news.language)),
                               n=s.news.max_articles, days=s.news.lookback_days)
        res = FetchResult("news", key, "OK", url=url, frequency="news")
        try:
            if _already_fetched(out_dir, rel, force):
                raw, res.status, res.path = (out_dir / rel).read_bytes(), "SKIPPED", rel
                n, res.latest = parse_gdelt(raw)
            else:
                for attempt in range(2):                 # GDELT: 1 request / 5 s, else plain-text error
                    raw = f.get(url)
                    try:
                        n, res.latest = parse_gdelt(raw)
                        break
                    except ValueError:
                        if attempt == 1:
                            raise
                        time.sleep(f.backoff * 3)
                res.path, res.sha256 = _save(out_dir, rel, raw)
            res.bytes, res.detail = len(raw), f"{n} articles in {s.news.lookback_days} days"
        except Exception as e:
            res.status, res.detail = "FAILED", str(e)
        results.append(res)
    return results


# --------------------------------------------------------------------------- freshness
def freshness(results: list[FetchResult], s: Settings, today: date) -> list[dict]:
    report = []
    for r in results:
        if not r.frequency:                      # companyfacts has no single "latest" date
            continue
        limit = s.project.max_age_days[r.frequency]
        if r.status == "FAILED" or not r.latest:
            age, verdict = None, "FAILED"
        else:
            age = (today - date.fromisoformat(r.latest)).days
            verdict = "OK" if age <= limit else "STALE"
        report.append({"source": r.source, "key": r.key, "latest": r.latest,
                       "age_days": age, "limit_days": limit, "status": verdict})
    return report


# --------------------------------------------------------------------------- run
def run(s: Settings, fetcher: Fetcher, out_dir: Path, today: Optional[date] = None,
        force: bool = False, news_fetcher: Optional[Fetcher] = None) -> tuple[list[FetchResult], list[dict]]:
    today = today or date.today()
    stamp = today.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = (fetch_fred(s, fetcher, out_dir, stamp, force)
               + fetch_sec(s, fetcher, out_dir, stamp, force)
               + fetch_news(s, news_fetcher or fetcher, out_dir, stamp, force))

    with open(out_dir / "_manifest.jsonl", "a", encoding="utf-8") as fh:   # append-only log
        for r in results:
            if r.status != "SKIPPED":
                fh.write(json.dumps(asdict(r)) + "\n")

    report = freshness(results, s, today)
    with open(out_dir / "_freshness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "key", "latest", "age_days", "limit_days", "status"])
        w.writeheader()
        w.writerows(report)
    return results, report


def _print(results: list[FetchResult], report: list[dict]) -> None:
    print(f"\n{'SOURCE':<16}{'KEY':<14}{'STATUS':<9}DETAIL")
    for r in results:
        print(f"{r.source:<16}{r.key:<14}{r.status:<9}{r.detail[:70]}")
    print(f"\n{'FRESHNESS':<16}{'KEY':<14}{'LATEST':<12}{'AGE':>5}{'LIMIT':>7}  STATUS")
    for x in report:
        age = "-" if x["age_days"] is None else x["age_days"]
        print(f"{x['source']:<16}{x['key']:<14}{str(x['latest']):<12}{age:>5}{x['limit_days']:>7}  {x['status']}")
    failed = sum(r.status == "FAILED" for r in results)
    stale = sum(x["status"] == "STALE" for x in report)
    print(f"\n{len(results)} downloads · {failed} failed · {stale} stale")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: download raw data into the bronze layer")
    ap.add_argument("--force", action="store_true", help="re-download files already fetched today")
    args = ap.parse_args()
    s = load_settings()
    if "example.com" in s.project.sec_user_agent:
        raise SystemExit("Put your real email in sec_user_agent (config/settings.yaml) first - SEC requires it.")
    out = s.data_path / "bronze"
    print("Downloading... (news is slow on purpose: GDELT blocks fast requests, about 1 minute)")
    # GDELT throttles bursts with HTTP 429 even at 1 request / 5 s, so go slower and wait longer
    news = Fetcher(s.project.sec_user_agent, min_interval=10.0, backoff=15.0, retries=4)
    results, report = run(s, Fetcher(s.project.sec_user_agent), out, force=args.force, news_fetcher=news)
    _print(results, report)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
