"""News filter audit (thesis method check).

The source tiers and headline rules are the researcher's judgement. This measures how well
they work, in two steps:

  1. python -m dashboard.news_audit            -> docs/news_audit.csv: a random sample of KEPT
     headlines (up to 50, spread across tiers) with an empty `relevant` column.
  2. Open the CSV, write y or n in `relevant` for each row (is this headline really about the
     company or the airline industry, and useful to an executive?). Save.
  3. python -m dashboard.news_audit --score    -> precision overall and per tier, for the thesis.

The funnel (all -> relevant -> not blocked -> unique) is printed too.
"""
from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text

from dashboard.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "news_audit.csv"
SAMPLE = 50


def funnel(eng) -> dict:
    q = text("""SELECT count(*) AS total, count(*) FILTER (WHERE relevant) AS relevant,
                       count(*) FILTER (WHERE relevant AND NOT blocked) AS not_blocked,
                       count(*) FILTER (WHERE kept) AS kept,
                       count(*) FILTER (WHERE kept AND tier = 1) AS tier1,
                       count(*) FILTER (WHERE kept AND tier = 2) AS tier2,
                       count(*) FILTER (WHERE kept AND tier = 3) AS tier3
                FROM silver.news_article""")
    with eng.connect() as con:
        return dict(con.execute(q).mappings().one())


def sample(eng, n: int = SAMPLE, seed: int = 42) -> pd.DataFrame:
    with eng.connect() as con:
        kept = pd.DataFrame(con.execute(text(
            "SELECT company_key, tier, domain, title, url, seen_at FROM silver.news_article WHERE kept")).mappings().all())
    if kept.empty:
        return kept
    # spread across tiers: equal share per tier, topped up from the rest
    per = max(1, n // kept["tier"].nunique())
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in kept.groupby("tier")]
    out = pd.concat(parts)
    rest = kept.drop(out.index)
    if len(out) < n and len(rest):
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    out = out.sort_values(["tier", "company_key"]).reset_index(drop=True)
    out.insert(0, "relevant", "")
    return out


def score(path=OUT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["relevant"] = df["relevant"].astype(str).str.strip().str.lower().str[:1]
    done = df[df["relevant"].isin(["y", "n"])]
    if done.empty:
        raise SystemExit(f"No y/n answers found in {path} - fill the 'relevant' column first.")
    res = (done.assign(ok=done["relevant"] == "y").groupby("tier")["ok"].agg(["count", "mean"])
           .rename(columns={"count": "rated", "mean": "precision"}))
    res.loc["all"] = [len(done), (done["relevant"] == "y").mean()]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the news filter")
    ap.add_argument("--score", action="store_true", help="score the filled-in docs/news_audit.csv")
    args = ap.parse_args()
    if args.score:
        res = score()
        print("NEWS FILTER PRECISION (share of kept headlines you rated relevant)")
        for tier, r in res.iterrows():
            label = "all tiers" if tier == "all" else f"tier {int(tier)}"
            print(f"  {label:<10} {r['precision']:>6.0%}  ({int(r['rated'])} rated)")
        return
    from dashboard.db import engine
    eng = engine()
    f = funnel(eng)
    print("NEWS FUNNEL (all stored articles)")
    print(f"  downloaded {f['total']} -> company/industry in headline {f['relevant']} -> not blocked "
          f"{f['not_blocked']} -> unique {f['kept']}  (tier 1: {f['tier1']}, tier 2: {f['tier2']}, tier 3: {f['tier3']})")
    df = sample(eng)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")       # utf-8-sig: opens cleanly in Excel
    print(f"\nWrote {len(df)} headlines to {OUT}")
    print("Next: open it, put y or n in the 'relevant' column, save, then run:  scripts\\run_news_audit.bat score")


if __name__ == "__main__":
    main()
