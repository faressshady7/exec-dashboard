"""Step 2 tests. No internet needed: a fake session returns canned responses."""
import json
from datetime import date

import pytest

from dashboard.bronze import (Fetcher, gdelt_query, parse_fred_csv, parse_gdelt,
                              parse_sec_latest_filing, run)
from dashboard.config import load_settings

TODAY = date(2026, 10, 19)


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


class FakeSession:
    """Answers by URL. `script` can hold a list of responses to simulate retries."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        for key, resp in self.routes.items():
            if key in url:
                if isinstance(resp, list):
                    return resp.pop(0) if len(resp) > 1 else resp[0]
                return resp
        return FakeResponse(404)


def fred_csv(series, last="2026-10-16", monthly=False):
    rows = ["observation_date," + series, "2015-01-01,1.0", "2015-01-02,."]
    rows.append(f"{last},2.5")
    return ("\n".join(rows) + "\n").encode()


def submissions(form="10-Q", filed="2026-07-17", period="2026-06-30"):
    return json.dumps({"filings": {"recent": {
        "form": ["8-K", form, "10-K"], "filingDate": ["2026-10-01", filed, "2026-02-20"],
        "reportDate": ["2026-10-01", period, "2025-12-31"]}}}).encode()


def gdelt(seen="20261018T121500Z", n=3):
    return json.dumps({"articles": [{"url": f"https://news.example/{i}", "title": f"Story {i}",
                                     "seendate": seen, "domain": "news.example", "language": "English"}
                                    for i in range(n)]}).encode()


FACTS = json.dumps({"entityName": "Test Air", "facts": {"us-gaap": {"Revenues": {}, "Assets": {}}}}).encode()


@pytest.fixture
def settings():
    return load_settings()


def all_routes(settings, **overrides):
    latest = {"monthly": "2026-08-01", "quarterly": "2026-04-01", "daily": "2026-10-16"}
    routes = {f"id={f.series}&": FakeResponse(200, fred_csv(f.series, latest[f.frequency]))
              for f in settings.risk_factors}
    routes["companyfacts"] = FakeResponse(200, FACTS)
    routes["submissions"] = FakeResponse(200, submissions())
    routes["gdeltproject"] = FakeResponse(200, gdelt())
    routes.update(overrides)
    return routes


def fast_fetcher(session):
    return Fetcher("Fares test@example.com", session=session, backoff=0, min_interval=0)


# ---------------------------------------------------------------- parsers
def test_fred_parser_skips_missing_dot_values():
    rows, latest = parse_fred_csv(fred_csv("VIXCLS"), "VIXCLS")
    assert rows == 2 and latest == "2026-10-16"


def test_fred_parser_rejects_wrong_series():
    with pytest.raises(ValueError, match="unexpected FRED header"):
        parse_fred_csv(fred_csv("VIXCLS"), "DGS10")


def test_latest_filing_ignores_8k():
    filed, detail = parse_sec_latest_filing(submissions())
    assert filed == "2026-07-17" and detail.startswith("10-Q")


# ---------------------------------------------------------------- full run
def test_run_saves_raw_files_manifest_and_freshness(settings, tmp_path):
    session = FakeSession(all_routes(settings))
    results, report = run(settings, fast_fetcher(session), tmp_path, today=TODAY)

    n_expected = len(settings.risk_factors) + 3 * len(settings.in_scope) + 1   # +1 industry news
    assert len(results) == n_expected and all(r.status == "OK" for r in results)
    # raw file is byte-for-byte what the server sent
    assert (tmp_path / "fred/VIXCLS/2026-10-19.csv").read_bytes() == fred_csv("VIXCLS")
    assert (tmp_path / "sec/companyfacts/UAL_0000100517/2026-10-19.json").exists()
    assert len((tmp_path / "_manifest.jsonl").read_text().splitlines()) == n_expected
    assert all(x["status"] == "OK" for x in report)       # monthly/quarterly lags are allowed


def test_sec_requests_carry_user_agent(settings, tmp_path):
    session = FakeSession(all_routes(settings))
    run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    sec_calls = [h for url, h in session.calls if "sec.gov" in url]
    assert sec_calls and all("@" in h["User-Agent"] for h in sec_calls)


def test_one_failure_does_not_stop_the_rest(settings, tmp_path):
    session = FakeSession(all_routes(settings, **{"id=VIXCLS&": FakeResponse(404)}))
    results, report = run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    status = {r.key: r.status for r in results if r.source == "fred"}
    assert status["VIXCLS"] == "FAILED" and status["DGS10"] == "OK"
    assert not (tmp_path / "fred/VIXCLS").exists()          # nothing half-saved
    assert {"key": "VIXCLS", "status": "FAILED"}.items() <= next(x for x in report if x["key"] == "VIXCLS").items()


def test_stale_data_is_flagged(settings, tmp_path):
    old = FakeResponse(200, fred_csv("DJFUELUSGULF", last="2026-09-01"))
    session = FakeSession(all_routes(settings, **{"id=DJFUELUSGULF&": old}))
    _, report = run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    fuel = next(x for x in report if x["key"] == "DJFUELUSGULF")
    assert fuel["status"] == "STALE" and fuel["age_days"] == 48


def test_retries_after_rate_limit(settings, tmp_path):
    flaky = [FakeResponse(429), FakeResponse(200, fred_csv("VIXCLS"))]
    session = FakeSession(all_routes(settings, **{"id=VIXCLS&": flaky}))
    results, _ = run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    assert next(r for r in results if r.key == "VIXCLS").status == "OK"


def test_second_run_same_day_skips_downloads(settings, tmp_path):
    run(settings, fast_fetcher(FakeSession(all_routes(settings))), tmp_path, today=TODAY)
    session2 = FakeSession(all_routes(settings))
    results, _ = run(settings, fast_fetcher(session2), tmp_path, today=TODAY)
    assert session2.calls == [] and all(r.status == "SKIPPED" for r in results)


# ---------------------------------------------------------------- news
def test_gdelt_query_builds_exact_phrases():
    assert gdelt_query(["United Airlines"], "english") == '"United Airlines" sourcelang:english'
    assert gdelt_query(["jet fuel", "airfares"], "english") == '("jet fuel" OR "airfares") sourcelang:english'


def test_gdelt_parser_reads_newest_date_and_handles_empty():
    assert parse_gdelt(gdelt()) == (3, "2026-10-18")
    assert parse_gdelt(b"") == (0, None)


def test_news_saved_per_company_and_industry(settings, tmp_path):
    run(settings, fast_fetcher(FakeSession(all_routes(settings))), tmp_path, today=TODAY)
    saved = sorted(p.parent.name for p in (tmp_path / "news/gdelt").glob("*/2026-10-19.json"))
    assert saved == sorted([c.ticker for c in settings.in_scope] + ["INDUSTRY"])


def test_gdelt_rate_limit_text_is_retried(settings, tmp_path):
    limited = [FakeResponse(200, b"Please limit requests to one every 5 seconds"), FakeResponse(200, gdelt())]
    session = FakeSession(all_routes(settings, gdeltproject=limited))
    results, _ = run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    assert next(r for r in results if r.source == "news").status == "OK"


def test_old_news_is_flagged_stale(settings, tmp_path):
    session = FakeSession(all_routes(settings, gdeltproject=FakeResponse(200, gdelt("20261001T000000Z"))))
    _, report = run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    assert all(x["status"] == "STALE" for x in report if x["source"] == "news")


# ---------------------------------------------------------------- share prices (FMP)
def fmp(symbol, days=("2026-10-16", "2026-10-15")):
    return json.dumps([{"symbol": symbol, "date": d, "open": 1, "high": 1, "low": 1, "close": 50.0,
                        "volume": 1000} for d in days]).encode()


def test_prices_skipped_without_key(settings, tmp_path):
    results, _ = run(settings, fast_fetcher(FakeSession(all_routes(settings))), tmp_path, today=TODAY)
    assert not any(r.source == "prices" for r in results)


def test_prices_saved_and_key_never_logged(settings, tmp_path):
    routes = all_routes(settings, **{f"symbol={c.ticker}&": FakeResponse(200, fmp(c.ticker)) for c in settings.in_scope})
    session = FakeSession(routes)
    results, report = run(settings, fast_fetcher(session), tmp_path, today=TODAY, price_key="SECRET123")
    prices = [r for r in results if r.source == "prices"]
    assert [r.status for r in prices] == ["OK"] * 3 and prices[0].latest == "2026-10-16"
    assert (tmp_path / "prices/fmp/UAL/2026-10-19.json").read_bytes() == fmp("UAL")
    assert "SECRET123" not in (tmp_path / "_manifest.jsonl").read_text()          # masked in the log
    fmp_calls = [h for u, h in session.calls if "financialmodelingprep" in u]
    assert all("@" not in h["User-Agent"] for h in fmp_calls)                      # SEC email not sent to FMP
    assert any(x["source"] == "prices" and x["status"] == "OK" for x in report)


def test_prices_error_is_failed_and_masked(settings, tmp_path):
    bad = FakeResponse(200, json.dumps({"Error Message": "Invalid API KEY"}).encode())
    routes = all_routes(settings, financialmodelingprep=bad)
    results, _ = run(settings, fast_fetcher(FakeSession(routes)), tmp_path, today=TODAY, price_key="SECRET123")
    prices = [r for r in results if r.source == "prices"]
    assert all(r.status == "FAILED" and "Invalid API KEY" in r.detail for r in prices)
    denied = all_routes(settings, financialmodelingprep=FakeResponse(401))
    results, _ = run(settings, fast_fetcher(FakeSession(denied)), tmp_path, today=TODAY, price_key="SECRET123", force=True)
    assert all("SECRET123" not in r.detail and "SECRET123" not in r.url for r in results)


# --------------------------------------------------------------------------- Finnhub news + SEC documents
def finnhub(n=2):
    return json.dumps([{"datetime": 1792900000 + i, "headline": f"United story {i}", "source": "CNBC",
                        "url": f"https://x/{i}", "summary": "A long enough summary about United Airlines and fuel."}
                       for i in range(n)]).encode()


def test_finnhub_saved_with_key_masked_and_skipped_without_key(settings, tmp_path):
    routes = all_routes(settings, **{"finnhub.io": FakeResponse(200, finnhub())})
    results, _ = run(settings, fast_fetcher(FakeSession(routes)), tmp_path, today=TODAY, news_key="FHSECRET99")
    fh = [r for r in results if r.source == "news_finnhub"]
    assert len(fh) == 3 and all(r.status == "OK" for r in fh) and fh[0].detail.startswith("2 articles")
    assert "FHSECRET99" not in (tmp_path / "_manifest.jsonl").read_text()
    assert all("FHSECRET99" not in r.url for r in fh)
    results, _ = run(settings, fast_fetcher(FakeSession(routes)), tmp_path / "b", today=TODAY)
    assert not any(r.source == "news_finnhub" for r in results)


def test_finnhub_error_object_fails_and_is_masked(settings, tmp_path):
    routes = all_routes(settings, **{"finnhub.io": FakeResponse(200, b'{"error": "Invalid API key FHSECRET99"}')})
    results, _ = run(settings, fast_fetcher(FakeSession(routes)), tmp_path, today=TODAY, news_key="FHSECRET99")
    fh = [r for r in results if r.source == "news_finnhub"]
    assert all(r.status == "FAILED" and "FHSECRET99" not in r.detail for r in fh)


def test_latest_10k_and_10q_documents_downloaded_once(settings, tmp_path):
    from dashboard.bronze import latest_text_filings
    sub = json.dumps({"filings": {"recent": {
        "form": ["8-K", "10-Q", "10-Q", "10-K"], "filingDate": ["2026-10-21", "2026-10-20", "2026-07-17", "2026-02-20"],
        "reportDate": ["", "2026-09-30", "2026-06-30", "2025-12-31"], "accessionNumber": ["a-1", "a-2", "a-3", "a-4"],
        "primaryDocument": ["e.htm", "q3.htm", "q2.htm", "k.htm"]}}}).encode()
    assert [(x["form"], x["accn"]) for x in latest_text_filings(sub)] == [("10-Q", "a-2"), ("10-K", "a-4")]
    session = FakeSession(all_routes(settings, submissions=FakeResponse(200, sub),
                                     **{"Archives/edgar": FakeResponse(200, b"<html><p>Item 1A. Risk Factors</p></html>")}))
    results, _ = run(settings, fast_fetcher(session), tmp_path, today=TODAY)
    docs = [r for r in results if r.source == "sec_document"]
    assert len(docs) == 6 and all(r.status == "OK" for r in docs)          # 3 companies x (10-K + 10-Q)
    n = sum("Archives/edgar" in u for u, _ in session.calls)
    results, _ = run(settings, fast_fetcher(session), tmp_path, today=TODAY, force=True)
    assert sum("Archives/edgar" in u for u, _ in session.calls) == n         # a filing never changes: not fetched again
    assert all(r.status == "SKIPPED" for r in results if r.source == "sec_document")
