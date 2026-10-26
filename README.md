# Executive Market-Risk Dashboard: data pipeline and AI layer

BSc thesis, Business Informatics, German International University (GIU): *Interactive Data Visualization for Executive
Decision-Making: An AI-Driven Approach to Financial Market Insights*. Fares · Supervisor: Dr. Minar El-Aasser.

**In one sentence:** every morning this code downloads free market and company data, cleans it, calculates every number
the dashboard shows, lets an AI explain those numbers (each figure checked against the calculation), and stores the
result in a database that the Power BI dashboard reads.

> **The system computes, the AI explains.** All numbers come from Python code; the AI only writes text about them,
> and a checker rejects any figure that was not computed.

## How it works

```mermaid
flowchart LR
    A[Free sources<br>SEC · FRED · FMP · GDELT] --> B[1 Bronze<br>raw downloads]
    B --> C[2 Silver<br>cleaned tables]
    C --> D[3 Gold<br>star schema]
    D --> E[4 Compute<br>KPIs · sensitivities · what-if]
    E --> F[5 AI<br>brief · chart titles · captions<br>every number checked]
    E --> G[bi views]
    F --> G
    G --> H[Power BI dashboard<br>5 pages, AI text On/Off]
```

Runs automatically every day on GitHub Actions (06:17 Cairo, backup run 08:47), which also keeps the free database awake.

## Where everything is

| Folder / file | What it is |
|---|---|
| `config/settings.yaml` | **The only place companies, peers, risk variables and thresholds are named** (company-agnostic). |
| `src/dashboard/` | The pipeline code, one file per step (table below). |
| `tests/` | Automatic tests (104). They never touch the real database. |
| `docs/METHOD_NOTES.md` | Every method decision, with the sentence to use in the thesis and its sources. |
| `docs/GITHUB_SETUP.md` | How the daily GitHub run and its secrets are set up. |
| `scripts/` | Windows double-click launchers: run everything, the news audit, save API keys. |
| `.github/workflows/daily.yml` | The daily automatic run. |
| `.env` (laptop only, never uploaded) | Passwords and API keys. |
| `data/`, `logs/` (laptop only) | Raw downloads and the output of the last local run. |

## What each step does

| Step | File | In | Out (database) |
|---|---|---|---|
| 0 Settings | `config.py` | `settings.yaml` | checked settings (stops early if a setting is wrong) |
| 1 Bronze | `bronze.py` | SEC filings, 15 FRED series, FMP share prices, GDELT news | raw files, untouched |
| 2 Silver | `silver.py` | raw files | `silver.*`: clean, typed, de-duplicated; every dropped value counted with its reason. **News test:** each headline gets a topic (fuel, labour, results, demand, finance, operations, capacity), near-duplicates are removed, and only trusted sources with a known topic are marked `usable` |
| 3 Gold | `gold.py` | silver | `gold.*`: star schema, grain stated for every table |
| 4 Compute | `compute.py`, `valuation.py`, `robustness.py`, `attention.py` | gold | KPIs, peer gaps (latest quarter all airlines reported), sensitivities, what-if, past-episode replays, valuation, rule-based alerts, and **the 'why'** (`gold.driver`): each KPI's change vs a year earlier split exactly into its parts (e.g. margin −1.1 pp = fuel −7.4 pp + other costs +6.3 pp) |
| 5 AI | `insights.py`, `ai.py` | computed results + tested headlines only | `ai.*`: headline, page-1 story (steps that link to pages), a story per page (back / what / why / news / what to do / next), chart titles and captions, options to consider |
| Views | `bi.py` | gold + compute + ai | `bi.*`: ready-made views that Power BI reads |
| Helpers | `db.py`, `tags.py`, `news_audit.py`, `setkey.py` | | database connection, find a company's SEC tags, news-filter audit, save a key |

## The database (Supabase, hosted PostgreSQL)

| Schema | Holds |
|---|---|
| `silver` | Cleaned source data: `sec_fact`, `fred_observation`, `stock_price`, `share_count`, `news_article`, `filing`, plus the audit tables `drop_summary`, `missing_quarter`, `load_log`. |
| `gold` | Star schema. Dimensions: `dim_company`, `dim_date`, `dim_variable`. Facts: `fact_financials` (company × metric × quarter), `fact_market` (variable × day), `fact_stock`, `fact_shares`, `fact_filing`, `fact_news`; plus the computed tables. |
| `ai` | Every AI brief and every attempt, with the prompt version, the check result and the model used. |
| `bi` | The views Power BI reads (`ai_brief` also carries the page stories `p1..p5_*` and steps `ch1..4_*`): `companies`, `kpi_latest`, `kpi_trend`, `market_today`, `market_history`, `whatif`, `scenarios`, `robustness`, `stock_history`, `valuation`, `attention`, `news`, `ai_brief`, `ai_recommendations`, `freshness`. |

## The AI layer in short

- **Round 1, rank:** the code computes candidate insights; the AI scores how relevant each is to an executive today.
  Final score = 0.5 significance + 0.3 relevance + 0.2 impact (adapted from LEVA, Zhao et al. 2025).
- **Round 2, brief:** a three-step story (situation, complication, resolution).
- **Round 3, story layer:** the pages read as one connected story.
  - Page 1: 2-4 steps, each one sentence with a number, each linking to the page that shows it.
  - Pages 2-5, under the charts: **back** (what the previous page showed) → **what** (the answer to the page's question)
    → **why** (from the calculated breakdown, never invented) → **news** (one tested headline, picked by id; the line
    is written by code with outlet and date) → **what to do** (one option with its trade-off) → **next** page.
  - An action title and a so-what caption per chart; 2-3 options to consider.
- **Checks (code, on every line):** every number must match a computed fact from that page or chart; a 'why' must
  name the KPI and the calculated driver ('mostly' only if that driver explains at least half); peer comparisons
  must name their quarter; no double negatives or invented hedges. A failed line is rewritten once, then replaced
  by computed text. Each run logs what was rejected and why.
- **Study switch:** in Power BI, "AI text Off" hides all AI text; everything else stays identical.

## Running it by hand (Windows)

- Everything: double-click `scripts\run_all.bat`.
- One step: `python -m dashboard.<step>` (for example `python -m dashboard.ai`), with `src` on `PYTHONPATH`.
- Tests: `python -m pytest`.
