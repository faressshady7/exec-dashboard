"""Find which XBRL tags a company really uses (to fix gaps in the coverage report).

    python -m dashboard.tags DAL interest
    python -m dashboard.tags AAL revenue

Shows every us-gaap tag whose name contains the word, how many 10-K/10-Q values it has
since the start date, and the first/last period - newest first.
"""
from __future__ import annotations

import json
import sys

from dashboard.config import load_settings
from dashboard.silver import NUMBER_FORMS, latest_file


def find(ticker: str, word: str) -> list[tuple[str, int, str, str]]:
    s = load_settings()
    c = s.company(ticker.upper())
    f = latest_file(s.data_path / "bronze" / "sec" / "companyfacts" / f"{c.ticker}_{c.cik10}", "*.json")
    if not f:
        raise SystemExit(f"No bronze companyfacts file for {c.ticker}. Run: python -m dashboard.bronze")
    gaap = json.loads(f.read_bytes())["facts"].get("us-gaap", {})
    start = s.project.start_date.isoformat()
    out = []
    for tag, body in gaap.items():
        if word.lower() not in tag.lower():
            continue
        entries = [e for unit in body.get("units", {}).values() for e in unit
                   if e.get("form") in NUMBER_FORMS and e["end"] >= start]
        if entries:
            out.append((tag, len(entries), min(e["end"] for e in entries), max(e["end"] for e in entries)))
    return sorted(out, key=lambda r: r[3], reverse=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m dashboard.tags <TICKER> <word>")
    rows = find(sys.argv[1], sys.argv[2])
    print(f"{'TAG':<70}{'N':>5}  FIRST       LAST")
    for tag, n, first, last in rows:
        print(f"{tag[:69]:<70}{n:>5}  {first}  {last}")
    if not rows:
        print("(no tags contain that word)")


if __name__ == "__main__":
    main()
