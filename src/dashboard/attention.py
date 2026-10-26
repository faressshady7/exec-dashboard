"""'Attention' boxes for the dashboard: rule-based, from computed results only (no AI, no invented data).

One row = one box: company (whose page it appears on), page, rank (1 = show first), severity, title,
impact, detail, ticker (the company the box is about; empty for market-wide boxes), kind (risk = market/KPI
alert, check = how far to trust an estimate or comparison).
Every company gets its own set, so the dashboard's company selector shows that company's boxes.
Severity rules (documented, researcher's choice):
  HIGH      an UNFAVOURABLE unusual move in the focal company's latest quarter (> unusual_sigma)
  MODERATE  a cost/rate driver at a 10-year extreme (percentile >= 90 or <= 10), a comparability
            caveat, or a fragile estimate (robustness < 80%)
  MONITOR   favourable unusual moves, biggest gap vs peers, timing differences, notes
Wording follows the AI rules: describes what the data shows, never says what WILL happen or WHY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.config import Settings

COLS = ["company", "page", "rank", "severity", "title", "impact", "detail", "ticker", "kind"]
ORDER = {"HIGH": 0, "MODERATE": 1, "MONITOR": 2}
DRIVERS = ("cost", "rates")            # market categories that feed the sensitivities
ROBUST_MIN = 80


def _fmt(v: float, unit: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if unit == "USD":
        a = abs(v)
        s = f"${a / 1e9:,.1f}B" if a >= 1e9 else f"${a / 1e6:,.0f}M"
        return ("-" if v < 0 else "") + s
    if unit == "x":
        return f"{v:.1f}x"
    return f"{v:.1f}%"


def _chg(v: float, unit: str) -> str:
    """Quarter-on-quarter change: % KPIs move in percentage points."""
    if unit == "%":
        return f"{v:+.1f} pp"
    return ("+" if v >= 0 else "-") + _fmt(abs(v), unit)


def brief_boxes(kpis: pd.DataFrame, pg: pd.DataFrame, snap: pd.DataFrame, s: Settings) -> list[dict]:
    from dashboard.compute import KPIS
    rows, focal = [], s.focal_company
    k = kpis[kpis["ticker"] == focal]
    if len(k):
        last = k[k["period_end"] == k["period_end"].max()]
        for r in last[last["unusual"].fillna(False)].itertuples():
            better = KPIS[r.kpi][2]
            good = (r.qoq_change > 0) == (better == "higher")
            rows.append({"severity": "MONITOR" if good else "HIGH", "mag": abs(r.qoq_change),
                         "title": f"Unusual move: {r.label}",
                         "impact": f"{_chg(r.qoq_change, r.unit)} vs the previous quarter, now {_fmt(r.value, r.unit)}; "
                                   f"larger than {s.method.unusual_sigma:g} standard deviations of its usual quarterly change",
                         "detail": f"{focal} · {r.quarter_label} · {'favourable' if good else 'unfavourable'} direction",
                         "ticker": focal})
    for r in snap[snap["category"].isin(DRIVERS)].itertuples():
        p = r.percentile_10y
        if pd.isna(p) or 10 < p < 90:
            continue
        high = p >= 90
        unit = "" if r.change_type == "points" else "%"
        rows.append({"severity": "MODERATE", "mag": abs(p - 50),
                     "title": f"{r.label} near a 10-year {'high' if high else 'low'}",
                     "impact": f"Latest {r.latest:,.2f} {r.unit}: " + (
                         f"the {'highest' if high else 'lowest'} level of the past 10 years" if p >= 99.5 or p <= 0.5 else
                         f"{'higher' if high else 'lower'} than {p if high else 100 - p:.0f}% of the past 10 years"),
                     "detail": f"1-year change {r.chg_1y:+.1f}{unit or ' pp'} · data to {pd.Timestamp(r.latest_date):%d %b %Y}",
                     "ticker": None})
    f = pg[pg["ticker"] == focal] if len(pg) else pg
    if len(f):
        f = f.assign(bad=np.where(f["better"] == "higher", -f["gap_vs_peer_avg"], f["gap_vs_peer_avg"]))
        f = f[f["unit"] == "%"].sort_values("bad", ascending=False)   # % KPIs are comparable across sizes
        if len(f) and f["bad"].iloc[0] > 0:
            r = f.iloc[0]
            rows.append({"severity": "MONITOR", "mag": 0,
                         "title": f"Biggest gap vs peers: {r['label']}",
                         "impact": f"{focal} {_fmt(r['value'], r['unit'])} vs peer average {_fmt(r['peer_avg'], r['unit'])} "
                                   f"(rank {int(r['rank'])} of {int(r['n_companies'])})",
                         "detail": f"Same quarter for all companies: {r['quarter_label']}", "ticker": focal})
    if not rows:
        rows.append({"severity": "MONITOR", "mag": 0, "title": "No unusual moves this quarter",
                     "impact": "No KPI moved by more than its usual quarterly range and no driver is at a 10-year extreme",
                     "detail": "Rules: unusual > 2 standard deviations; extreme = 10-year percentile <= 10 or >= 90", "ticker": None})
    return rows


def peer_boxes(val: pd.DataFrame, s: Settings) -> list[dict]:
    rows = []
    for c in s.in_scope:
        for cav in c.caveats:
            rows.append({"severity": "MODERATE", "mag": 1, "title": f"Comparability: {c.ticker}",
                         "impact": cav, "detail": "Shown next to this company's fuel figures and in the AI brief",
                         "ticker": c.ticker})
    if len(val):
        ends = val.dropna(subset=["ttm_end"])
        if ends["ttm_end"].nunique() > 1:
            parts = " · ".join(f"{r.ticker} to {pd.Timestamp(r.ttm_end):%d %b %Y}" for r in ends.itertuples())
            rows.append({"severity": "MONITOR", "mag": 0.5, "title": "Last 4 quarters end on different dates",
                         "impact": parts, "detail": "Some companies have reported a newer quarter; the gap closes as the others report",
                         "ticker": None})
        for r in val.itertuples():
            if r.note:
                rows.append({"severity": "MONITOR", "mag": 0, "title": f"Multiples not meaningful: {r.ticker}",
                             "impact": r.note, "detail": "Left blank rather than showing a negative ratio", "ticker": r.ticker})
    return rows


def risk_boxes(sens: pd.DataFrame, s: Settings) -> list[dict]:
    rows = []
    f = sens[sens["ticker"] == s.focal_company] if len(sens) else sens
    for r in f.itertuples():
        name = "fuel" if r.factor == "jet_fuel" else "interest-rate"
        rp = getattr(r, "robust_pct", np.nan)
        if str(r.method).startswith("not"):
            rows.append({"severity": "MONITOR", "mag": 0, "title": f"No {name} estimate for {r.ticker}",
                         "impact": "The past-quarter relationship was too weak to estimate (below the R-squared threshold)",
                         "detail": f"R-squared {r.r2:.2f}", "ticker": r.ticker})
        elif not pd.isna(rp) and rp < ROBUST_MIN:
            rows.append({"severity": "MODERATE", "mag": 100 - rp, "title": f"{name.capitalize()} estimate is fragile",
                         "impact": f"Same result in only {rp:.0f}% of alternative settings; treat as indicative",
                         "detail": f"{r.ticker} · R-squared {r.r2:.2f} · {str(r.method).split(' (')[0]}", "ticker": r.ticker})
        elif not pd.isna(rp):
            rows.append({"severity": "MONITOR", "mag": 0, "title": f"{name.capitalize()} estimate is robust",
                         "impact": f"Same result in {rp:.0f}% of alternative settings",
                         "detail": f"{r.ticker} · R-squared {r.r2:.2f} · {str(r.method).split(' (')[0]}", "ticker": r.ticker})
    rows.append({"severity": "MONITOR", "mag": -1, "title": "Linear, everything-else-equal estimates",
                 "impact": "Airline responses such as fare or capacity changes are not modelled, so large shocks are likely overstated",
                 "detail": "Scenarios beyond the largest past 4-quarter moves are flagged", "ticker": None})
    return rows


def _rank(rows: list[dict], page: str, kind: str = "check") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["o"] = df["severity"].map(ORDER)
    df = df.sort_values(["o", "mag"], ascending=[True, False]).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["page"] = page
    df["company"] = None
    df["kind"] = kind
    return df[COLS]


def attention(kpis: pd.DataFrame, pg: pd.DataFrame, snap: pd.DataFrame, sens: pd.DataFrame,
              val: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Boxes for every company in scope (`pg` is kept for the signature; gaps are recomputed per company)."""
    from dashboard.compute import peer_gaps
    frames = []
    for c in s.in_scope:
        sc = s.model_copy(update={"focal_company": c.ticker,
                                  "peers": [o.ticker for o in s.in_scope if o.ticker != c.ticker]})
        # v8 layout: the market/KPI alerts sit on Risk Drivers (page 3) above the estimate checks, in one
        # "Risks and checks" panel; page 1 only shows how many need attention (kind = 'risk', HIGH or MODERATE)
        risks = _rank(brief_boxes(kpis, peer_gaps(kpis, sc), snap, sc), "risk", "risk")
        checks = _rank(risk_boxes(sens, sc), "risk", "check")
        checks["rank"] += len(risks)
        df = pd.concat([risks, checks,
                        _rank(peer_boxes(val, sc) or [{"severity": "MONITOR", "mag": 0, "title": "No comparability notes",
                                                       "impact": "-", "detail": "-", "ticker": None}], "peer")],
                       ignore_index=True)
        df["company"] = c.ticker
        frames.append(df[COLS])
    return pd.concat(frames, ignore_index=True)
