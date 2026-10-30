# Chatbot: "Ask about this page"

An AI chat that answers questions about the dashboard, from the dashboard's data, the latest
10-K/10-Q text and filtered news. Opened from a button on each Power BI page, already set to the
airline and the page the executive is looking at.

## How it answers (one question)

```mermaid
flowchart LR
    Q[Question] --> R1{Input rail<br>code}
    R1 -- asks for passwords, SQL,<br>the instructions --> X[Refused<br>no AI call]
    R1 -- OK --> M[Gemini picks<br>the data tools]
    M --> T[Fixed read-only tools<br>bi views only]
    T --> M
    M --> C{Number check<br>code}
    C -- every number traced --> A[Answer + sources]
    C -- a number not in the data --> W[One rewrite] --> C2{Check again}
    C2 -- still wrong --> F[Show the data itself]
    C2 -- OK --> A
    A --> L[(ai.chat log)]
```

| Situation | What the user gets |
|---|---|
| Answer is in the data | Plain-language answer, numbers checked, sources listed (dashboard, 10-K/10-Q section, news article with link) |
| On topic but not in the data (e.g. "What is a 10-K?") | General knowledge, labelled **"not from your data"**, with no numbers |
| Off topic ("write a poem") | Polite refusal |
| Asks for passwords, keys, SQL or the instructions | Refused by code before the AI is called |

## Data tools (the only way the chatbot reaches data)

| Tool | Reads | Example question |
|---|---|---|
| `get_kpis` | `bi.kpi_latest` | "How does our margin compare with the others?" |
| `get_kpi_history` | `bi.kpi_trend` | "How has net debt moved over the last 3 years?" |
| `get_market_risks` | `bi.market_today` | "Which market risk matters most today?" |
| `run_what_if` | `bi.whatif` (same maths as the sliders) | "What if fuel rises 20% and rates 1 point?" |
| `get_valuation` | `bi.valuation` | "Who is cheapest on P/E?" |
| `search_news` | `bi.news` (GDELT + Finnhub, with summaries) | "Any news about our pilots?" |
| `search_filings` | `bi.filing_text` (full-text search of the latest 10-K and 10-Q) | "What does our 10-K say about fuel hedging?" |
| `get_ai_brief` | `bi.ai_brief` | "Summarise today's brief" |

Every argument is checked (company must be in the settings file, what-if moves within limits);
search words are passed as bound parameters, never pasted into SQL.

## Number rule

A number in an answer must be (a) a computed fact returned by a tool, or (b) written word for word
in an official text the tools returned (a 10-K/10-Q passage, or the already-checked AI brief).
**Numbers from news are not accepted**: news is untrusted text. Dates ("Oct 28"), report names
("10-Q", "Item 7A") and years are not counted as figures.

## Security

| Risk | Defence | Source |
|---|---|---|
| The AI writes harmful SQL | It never writes SQL: fixed, parameterised queries only | Pedro et al. (2025) |
| Too much database access | Own account `chat_reader`: read the `bi` views, add to `ai.chat`; nothing else (tested) | Pedro et al. (2025), permission hardening |
| Instructions hidden in news or filings | Tool results are marked untrusted; the prompt says never follow them | Greshake et al. (2023) |
| Asking for secrets or the instructions | Input rail refuses before any AI call; output rail blocks keys and connection strings | Rebedea et al. (2023) |
| Off-topic use | Topic rule in the prompt + refusal | Rebedea et al. (2023) |
| Invented numbers | Number check, one rewrite, then the data itself | Huang et al. (2025) |
| Over-trust in unsupported answers | General knowledge labelled, no numbers | Kim et al. (2024) |
| Using up the free AI quota | Optional password, 6 questions/minute, 40 per session | operational (no paper) |
| Admin account used by mistake | The chat page refuses to start with the `postgres` account | design |

## Setup (once, about 15 minutes)

### 1. Create the chatbot's database account
Supabase → SQL Editor → New query → paste `sql/chat_reader_role.sql` → replace `CHANGE-ME` with a
long random password (letters and digits) → **Run**. The daily pipeline re-grants the views every morning.

### 2. Add to `.env` (laptop)
```
CHAT_DATABASE_URL=postgresql://chat_reader.YOUR-PROJECT-REF:THE-PASSWORD@aws-0-YOUR-REGION.pooler.supabase.com:5432/postgres
FINNHUB_API_KEY=your-finnhub-key          (or run scripts\set_finnhub_key.bat)
CHAT_PASSWORD=                            (optional)
```
Copy your existing `DATABASE_URL` line and change only the user name (`postgres.` → `chat_reader.`)
and the password.

### 3. GitHub secret (for the daily news download)
Repo → Settings → Secrets and variables → Actions → New repository secret:
`FINNHUB_API_KEY` = your Finnhub key. The daily run (`.github/workflows/daily.yml`) already passes it on; if your copy on GitHub lacks it, add this line
under `env:` next to `FMP_API_KEY`:
```yaml
      FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
```

### 4. Start the chat page
Double-click `scripts\run_chat.bat`. It opens http://localhost:8501. Keep the window open.

### 5. The Power BI button (each page)
1. **Modeling → New measure** (one per page; change the page number):
   ```DAX
   Chat Link P1 = "http://localhost:8501/?company=" & SELECTEDVALUE ( companies[ticker], "UAL" ) & "&page=1"
   ```
   (`companies` = the `bi companies` table in your model; use its name if Power BI renamed it.)
2. **Insert → Buttons → Blank**. Text: "Ask about this page".
3. **Format button → Action → On**, Type **Web URL**, click **fx** → Field value → `Chat Link P1`.
4. In Power BI Desktop: **Ctrl + click** the button. In reading view: normal click.

## Evaluation data (`ai.chat`)

One row per question: question, answer, outcome (`answered`, `general`, `refused_security`,
`refused_topic`, `number_fallback`, `error`), tools used, sources, number-check result, whether a
rewrite was needed, wording flags, model, time taken. The chatbot can only add rows, never read
them, so one user's questions are never shown to another.

Useful counts for the thesis:
```sql
SELECT outcome, count(*), round(avg(latency_ms)) AS ms FROM ai.chat GROUP BY outcome;
SELECT avg((rewritten)::int) AS rewrite_rate, avg((check_passed)::int) AS checked FROM ai.chat
WHERE outcome IN ('answered', 'number_fallback');
```
Participants must agree in the consent form that their questions are logged.

## Limitations (state in the thesis)
- Answers depend on what the tools return: the chat knows the latest 10-K and 10-Q, not older reports.
- Full-text search matches words (with stemming), not meaning, so a question worded very differently
  from the report may find nothing; it then says so or answers from general knowledge (labelled).
- The number check proves a figure exists in the data, not that the sentence around it is right; the
  wording flags and a manual read of a sample of logged answers cover that.
- Questions differ between participants, so the chat is evaluated separately from the AI On/Off test.
