# Method notes and limitations (ready-to-adapt thesis text)

Decisions taken while building the system (Oct 2026), each with the sentence to use in the
thesis. Numbers refer to the data as of 19 Oct 2026 and change with every daily run.

## 1. Scope claim: configurable, not proven general  (audit #10)
**Claim:** the framework is *company-agnostic by design*: companies, peers, risk variables,
thresholds, news rules and scenarios are all set in one configuration file (`config/settings.yaml`);
no code names a company.
**Do not claim:** that it is proven to work for every industry. It was evaluated on one case
(US airlines: United, with Delta and American as peers).
> "The system is configuration-driven and company-agnostic by design; its evaluation is limited
> to one industry case (US airlines), so generalisation to other industries is plausible but untested."

## 2. Role of each model article  (audit #11)
- **LEVA (Zhao et al., IEEE TVCG 2025):** framework model (insight scoring, LLM narration) and
  supporting evidence for claims about LLM-enhanced visual analytics.
- **Ng & Nagalingham (IJACSA 2023):** *implementation* model for an airline BI pipeline only.
  IJACSA is a lower-tier open-access journal, so do not use it as evidence for effects.
- **Evidence for dashboard effects on decisions:** Hjelle et al. (Information & Management 2024).

## 3. News filter  (audit #9)
Raw articles are kept in bronze; silver *tags* each one (nothing is deleted, so the funnel is measurable):
1. **Relevance:** company or industry term in the headline (tickers matched as whole words).
2. **Blocklist:** stock-spam sites, content farms, state media.
3. **Duplicates:** same normalised headline kept once (best tier, then earliest).
4. **Tier** (researcher's judgement, criteria documented): 1 = established outlet with editorial
   standards and a corrections policy; 2 = specialist trade/business press; 3 = other.
   Only tiers 1-2 reach the AI.

Report the funnel from `run_news_audit.bat` (first measured: 400 -> 26 -> 19 -> 18 kept; 7 tier 2, 11 tier 3),
and the **precision** from the manual audit (`run_news_audit_score.bat`, 50 headlines rated y/n).
> "Source tiers reflect the researcher's judgement against documented criteria; a manual audit of
> N kept headlines found X% relevant (tier 2: Y%, tier 3: Z%)."

## 4. Sensitivities: estimates, robustness, wording  (audit #3, #6, #7)
- Fuel: quarterly fuel expense regressed on the quarter's average jet fuel price (slope = implied gallons).
  Rates: interest expense on the fed funds rate. COVID quarters (Q2 2020-Q4 2021) excluded from estimation.
- Thresholds (researcher's choices, in `settings.yaml` -> `method`): fuel R^2 >= 0.5, rates R^2 >= 0.3,
  unusual change > 2 sigma.
- **Robustness** (`gold.robustness`, 36 checks): fuel results unchanged under alternative R^2 cut-offs;
  including COVID raised fuel magnitudes by 8-14% (UAL -$379M -> -$433M) without changing direction
  or ranking. Rate results were fragile. Anomaly flags depend on sigma (UAL: 4 / 3 / 1 at 1.5 / 2 / 2.5).
> "Fuel sensitivities were robust to alternative thresholds; including COVID quarters increased
> magnitudes by 8-14% without changing direction or ranking. Interest-rate sensitivities were fragile
> and are reported as indicative only. Anomaly flags depend on the chosen threshold, as expected for
> a screening rule."
- **Causality:** the fuel link is near-mechanical (fuel cost = gallons x price); the rate link is a
  correlation. All texts say "based on past quarters ... estimated / linked to", never "will".
- **Comparability:** Delta owns an oil refinery supplying most of its jet fuel, so its fuel figures
  are flagged "not like-for-like". None of the three carriers hedged fuel in 2026, so no hedge adjustment.
- The thresholds are fixed in code/config, not chosen by the LLM: keeps results reproducible and
  keeps an independent ground truth for the AI error rate.

## 5. What-if simulator  (audit #8)
Two complementary methods (the "hybrid" stress-test practice):
- **Sensitivity sliders:** one or two factors, linear - the same framing as the airlines' own
  10-K Item 7A disclosures.
- **Historical scenarios:** the measured moves of past episodes (2020 COVID oil crash, 2022 Ukraine
  fuel spike, 2022-23 Fed hikes, last 12 months) replayed on today's company.
Every scenario is flagged when it exceeds the largest 4-quarter moves seen in the data.
> "The simulator is a linear, ceteris-paribus estimate. It ignores airline responses such as fare and
> capacity changes (in the 2026 fuel spike United cut about 5% of capacity), so large shocks are likely
> overstated; scenarios beyond historically observed moves are flagged as extrapolations."

## 6. AI layer evaluation  (audit #4)
Two error types, reported separately:
- **Numeric errors:** any number not traceable to a computed fact (automatic checker; one rewrite,
  then a deterministic template).
- **Possible claim errors:** exaggeration, cause or certainty words (automatic flag, never blocks),
  plus a manual read of a random sample of about 20 briefs.
Prompt version is stored with every brief (v2 since 19 Oct 2026: rule 6, estimates not "will").
The daily GitHub run stores one brief per day, which builds the evaluation sample.

### 6b. Page notes and options to consider (prompt v3, 19 Oct 2026)
- One brief per company (UAL, DAL, AAL), so the text follows the company chosen on the dashboard.
- Round 3 (same LLM, same facts-only rule): a 2-3 sentence note per dashboard page and 2-3 options to
  CONSIDER, each with why now (a number from the facts), the trade-off and the page holding the evidence.
- Same number check over all of it (one rewrite, then the computed summaries and no options that day)
  and the same wording flags.
- Design choice: each option points to its evidence page and states a trade-off, a light "cognitive
  forcing" step that reduces blind acceptance of AI advice (Bucinca, Malaya & Gajos 2021) - the
  over-reliance risk shown by Spatharioti et al. 2025 and Klingbeil et al. 2024.
- Limitation to state: the AI sees only the dashboard's computed facts, not company strategy, contracts
  or hedging programmes, so its options are prompts for discussion, not advice.
> "Options are AI-generated from verified facts and always shown with their evidence and trade-off; the
> executive decides."

### 6c. Evaluation design for the AI parts
- Study switch "AI text On / Off" on every page (synced): Off hides the summary, page notes and options;
  everything else is identical, so "dashboard" vs "dashboard + AI" is a clean comparison.
- Option quality is rated separately from number accuracy: a sample of about 10 options rated 1-5 for
  usefulness and realism by 2-3 people with finance or aviation background (or the participants), because the
  number check proves accuracy, not quality.
- Each page answers ONE question (shown under the company name): 1 Where do we stand today, and what should
  we consider? 2 How do we compare with the other two airlines? 3 Which market risks move our results, and by
  how much? 4 What happens to this quarter if fuel or rates move? 5 How did we get here, quarter by quarter?

## 7. Data freshness and operations  (audit #2, #5)
- Wording: **daily-refreshed**, not "real-time" (market data daily; financials quarterly; GDP quarterly with a lag).
- Runs automatically every morning on GitHub Actions; this also keeps the free Supabase database awake.
- Backup for the defense: frozen study brief, database snapshot one week before, short screen recording.

## 8. Participants  (audit #1 - open)
Students and professionals act as proxy users for executives; state it as a limitation and, if
possible, report results separately for professionals.

## 9. Share prices and valuation (Peer Benchmark page)
- **Source:** Financial Modeling Prep (FMP) free plan, end-of-day closes, about 5 years; 3 calls a day.
  Shares outstanding from each 10-K/10-Q cover page (SEC `dei:EntityCommonStockSharesOutstanding`).
- **Definitions:** index = close / close on the first common day of the 3-year window x 100;
  market cap = latest close x latest reported shares; EV = market cap + net debt;
  P/E on the last 4 quarters' net income (blank when a loss); EV/Revenue on the last 4 quarters;
  P/B blank when equity is negative. Each row states the quarter its "last 4 quarters" end on
  (companies report on different dates, so for a few weeks a year they differ).
- **Net debt (same for every company, also used by the KPI card):** total debt = long-term + current debt,
  both including finance leases, excluding operating leases; minus liquidity = cash and equivalents +
  short-term investments. Fixed on 19 Oct 2026: the first version took whichever SEC line each company
  used, so Delta's excluded leases and current debt and American's counted only bare cash (missing about
  $6.7B of short-term investments). Real effect: AAL net debt 24.8B -> 21.2B, DAL 5.3B -> 9.1B, UAL 14.1B -> 9.8B.
- **Cross-check:** American reports total debt of $34.7B at 31 Mar 2026 *including operating leases and
  pensions* and $10.8B liquidity *including its undrawn credit line*. Our $27.7B debt + about $7B operating
  leases + pensions, and our $7.3B liquidity + the undrawn line, reconcile with those figures.
- **Known break:** United changed its debt line in Dec 2023; the new line is about $2-3B higher than the old
  parts added up (it also counts other financing liabilities). United's net debt before 2024 is therefore
  slightly understated. This sits at the start of the 3-year window.
- **Data fix:** one filing had an impossible cover date (AAL Q2 2026 10-Q: 17 Jul 2027, filed Jul 2026);
  a cover date later than the filing date is replaced by the filing date.
- **Limitations:** raw closes, not dividend-adjusted (Delta pays a small dividend, so its return is
  slightly understated); FMP free data is for personal use, cited as a data source.
> "Share prices are daily closes from Financial Modeling Prep; valuation multiples use standard
> trailing-four-quarter definitions and are left blank where not meaningful (losses, negative equity)."

---

# Verified references (peer-reviewed, 2021-2026 unless marked)

## Literature: the five keywords and the research gap (17)
1. Bach, B. et al. (2023). Dashboard design patterns. *IEEE TVCG*.
2. Shao, H., Martinez-Maldonado, R., Echeverria, V., Yan, L., & Gasevic, D. (2024). Data storytelling in data visualisation: Does it enhance the efficiency and effectiveness of information retrieval and insights comprehension? *Proc. CHI '24*. (Get the DOI from the ACM Digital Library.)
3. Islam, M. S. et al. (2024). DataNarrative: Automated data-driven storytelling with visualizations and texts. *EMNLP 2024*.
4. Hjelle, S., Mikalef, P., Altwaijry, N., & Parida, V. (2024). Organizational decision making and analytics: An experimental study on dashboard visualizations. *Information & Management*, 61(6), 104011. doi:10.1016/j.im.2024.104011
5. Csaszar, F. A., Ketkar, H., & Kim, H. (2024). Artificial intelligence and strategic decision-making: Evidence from entrepreneurs and investors. *Strategy Science*, 9(4), 322-345. doi:10.1287/stsc.2024.0190
6. Kong, Y. et al. (2024). Large language models for financial and investment management. *Journal of Portfolio Management*, 51(2).
7. Klingbeil, A., Grutzner, C., & Schreck, P. (2024). Trust and reliance on AI. *Computers in Human Behavior*, 160, 108352.
8. Huang, L. et al. (2025). A survey on hallucination in large language models. *ACM TOIS*.
9. Spatharioti, S. E., Rothschild, D., Goldstein, D. G., & Hofman, J. M. (2025). Effects of LLM-based search on decision making: Speed, accuracy, and overreliance. *Proc. CHI 2025*.
10. Gathani, S. et al. (2022). Augmenting decision making via interactive what-if analysis. *CIDR 2022*.
11. Gunklach, J., Jacob, K., & Michalczyk, S. (2023). Beyond dashboards? Designing data stories for effective use in business intelligence and analytics. *ECIS 2023*, Paper 327.
12. Zhao, Y. et al. (2025). LEVA: Using large language models to enhance visual analytics. *IEEE TVCG*.
13. Doshi, A. R., Bell, J. J., Mirzayev, E., & Vanneste, B. (2025). Generative artificial intelligence and evaluating strategic decisions. *Strategic Management Journal*. doi:10.1002/smj.3677
14. Ng & Nagalingham (2023). Implementation of business intelligence solutions for US airlines. *IJACSA*, 14(1). (implementation model only)
15. Pamuk, M., & Schumann, M. (2024). Towards AI dashboards in financial services. *Machine Learning and Knowledge Extraction*, 6(3), 1720-1761. doi:10.3390/make6030085
16. Lavalle, A., Mate, A., Santos, M. Y., Guimaraes, P., Trujillo, J., & Santos, A. (2025). A methodology for the systematic design of storytelling dashboards applied to Industry 4.0. *Data & Knowledge Engineering*, 156. doi:10.1016/j.datak.2025.102410
17. Perez, A. S. et al. (2025). An LLM-based approach for insight generation in data analysis. *NAACL 2025*.
17a. Bucinca, Z., Malaya, M. B., & Gajos, K. Z. (2021). To trust or to think: Cognitive forcing functions can reduce overreliance on AI in AI-assisted decision-making. *Proc. ACM Hum.-Comput. Interact.* 5(CSCW1), Article 188.

## Variables: why each risk factor is in the model (4)
18. Horobet, A., Zlatea, M. L. E., Belascu, L., & Dumitrescu, D. G. (2022). Oil price volatility and airlines' stock returns. *Journal of Business Economics and Management*, 23(2), 284-304. doi:10.3846/jbem.2022.16094
19. Blampied, N., & Mahadeo, S. M. R. (2024). Airline industry equities under external uncertainty shocks. *Economics Letters*, 244, 111994. doi:10.1016/j.econlet.2024.111994
20. Chi, J. (2023). Explaining US travel behavior with perceived threat of pandemic, consumer sentiment, and economic policy uncertainty. *Transport Policy*, 137, 90-99. doi:10.1016/j.tranpol.2023.04.019
21. Hanson, D., Toru Delibasi, T., Gatti, M., & Cohen, S. (2022). How do changes in economic activity affect air passenger traffic? *Journal of Air Transport Management*, 98. doi:10.1016/j.jairtraman.2021.102147

Verified and available if an examiner asks for more per variable: Kang et al. 2021 (NAJEF 57),
Gabbiadini et al. 2026 (JATM 130), Alici 2024 (ANDULI 22(25)), Fasanya et al. 2022 (PLoS ONE 17(5),
uses the same infectious-disease index), Atems & Yimga 2021 (JATM 97), Cai, Zhang & Zhang 2025
(Economics of Transportation 41), Song, Su & Qin 2025 and Cai, Fu & Zhang 2025 (Transport Policy).

## Data sources (cite as data, not as evidence - age rule does not apply)
- Baker, Bloom & Davis (2016), *QJE* 131(4) - economic policy uncertainty and EMV trackers (FRED).
- Caldara & Iacoviello (2022), *AER* 112(4) - geopolitical risk index (planned).
- FRED (Federal Reserve Bank of St. Louis); SEC EDGAR XBRL company facts; GDELT DOC 2.0;
  Financial Modeling Prep (end-of-day share prices).

## Classics (theory and definitions only - confirm with Dr. Minar)
Simon (1947, 1960); Sweller (1988); Shneiderman (1996); Tufte (2001); Eppler & Mengis (2004);
Tetlock (2007); Kahneman & Klein (2009); Segel & Heer (2010).

## Industry context (not peer-reviewed; label as context)
Oracle (2023) decision-making survey; McKinsey (2019) decision making in the age of urgency; Gartner (2024/2025).

## Still to verify / not used
- InReAcTable (ACM UIST 2025): check authors, pages and DOI before citing.
- "Interactive visualizations and LLM-driven explanations for transparent multi-objective portfolio
  management" (Springer 2025): not verified, do not cite until opened.
- Dropped (fail the rule): Chen, Roll & Ross 1986; Gallet & Doucouliagos 2014; Baker et al. 2020 (NBER WP);
  Cranfield 2012 MSc thesis; Data-to-Dashboard, MIND and other arXiv preprints.
