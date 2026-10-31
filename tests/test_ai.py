"""Step 6 tests: the number checker and the two-round flow, with a fake LLM (no internet)."""
import os

import pytest

from dashboard.ai import MockLLM, Round1, check_numbers, check_wording, generate, round1_prompt, round2_prompt
from dashboard.config import load_settings
from dashboard.insights import Fact, Insight, fmt

FACTS = [Fact("fuel.latest", "Jet fuel", 4.5612, "USD/gal"),
         Fact("fuel.chg_1y", "Jet fuel 1y change", 116.71, "%"),
         Fact("exp.10pct", "Impact of +10% fuel", -378_900_000.0, "USD"),
         Fact("oi", "Operating income", 1_096_000_000.0, "USD"),
         Fact("share", "Fuel share", 30.83, "%"),
         Fact("gap", "Gap vs peers", 4.31, "pp")]


@pytest.mark.parametrize("brief", [
    "Jet fuel is $4.56/gal, up 116.7% in a year.",
    "A 10% rise would cut quarterly operating income by about $379M from $1.10B.",   # abs of a negative, M and B
    "Fuel is 30.8% of costs, 4.3 pp above peers in Q2 2026.",                         # quarter/year ignored
    "Operating income was $1.1B; fuel nearly 31% of costs.",                          # coarser rounding is fine
    "A 100bp move, ranked 1 of 3.",                                                   # small constants allowed
])
def test_checker_accepts_numbers_from_the_facts(brief):
    c = check_numbers(brief, FACTS)
    assert c.passed, c.unmatched


@pytest.mark.parametrize("brief, bad", [
    ("Margins fell to 2.9% this quarter.", "2.9%"),
    ("Fuel costs rose to $5.2B.", "$5.2B"),
    ("Jet fuel is $4.80/gal.", "$4.80/gal"),
    ("A 10% rise would cut income by $450M.", "$450M"),
])
def test_checker_catches_invented_numbers(brief, bad):
    c = check_numbers(brief, FACTS)
    assert not c.passed and bad in c.unmatched


def test_formatting():
    assert fmt(-378_900_000, "USD") == "-$379M"
    assert fmt(1_096_000_000, "USD") == "$1.10B" and fmt(30.83, "%") == "30.8%" and fmt(3.17, "x") == "3.2x"


def insight(i, sig=0.5, imp=0.5, facts=None):
    return Insight(id=f"t:{i}", type=f"type{i}", title=f"Insight {i}", summary=f"Summary {i}",
                   facts=facts if facts is not None else [Fact(f"f{i}", f"fact {i}", 10.0 + i, "%")],
                   significance=sig, impact=imp)


def r1(cands, rel):
    return {"ranking": [{"id": c.id, "relevance": rel.get(c.id, 0.1), "why": "x"} for c in cands]}


def test_relevance_from_llm_changes_the_ranking():
    s = load_settings()
    cands = [insight(1, sig=0.6), insight(2, sig=0.5), insight(3, sig=0.1), insight(4, sig=0.1)]
    def reply(prompt, schema):
        if schema is Round1:
            return r1(cands, {"t:4": 1.0})                         # LLM says #4 matters most
        return {"headline": "H", "brief": "Value is 14.0% now.", "fact_ids_used": ["f4"]}
    res = generate(s, cands, MockLLM(reply))
    assert res.selected[0].id == "t:4"                              # 0.5*0.1+0.3*1+0.2*0.5 = 0.45 beats 0.43
    assert res.check.passed and not res.fallback_used


def test_rejected_draft_is_rewritten_with_feedback():
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]
    seen = []
    def reply(prompt, schema):
        if schema is Round1:
            return r1(cands, {})
        seen.append(prompt)
        bad = len(seen) == 1
        return {"headline": "H", "brief": "It is 99.9%." if bad else "It is 11.0%.", "fact_ids_used": ["f1"]}
    res = generate(s, cands, MockLLM(reply))
    assert [a["passed"] for a in res.attempts] == [False, True]
    assert "99.9%" in seen[1] and "rejected" in seen[1]             # feedback names the bad number
    assert res.brief == "It is 11.0%."


def test_two_failures_fall_back_to_template():
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]
    reply = lambda p, schema: r1(cands, {}) if schema is Round1 else \
        {"headline": "H", "brief": "Up 77.7%.", "fact_ids_used": []}
    res = generate(s, cands, MockLLM(reply))
    assert res.fallback_used and res.check.passed                   # template only uses computed text
    assert res.brief.startswith("Summary")


def test_prompts_mark_news_as_untrusted_and_list_display_values():
    cands = [insight(1)]
    assert "id: t:1" in round1_prompt(cands, "United")
    p2 = round2_prompt(cands, "United")
    assert 'display: "11.0%"' in p2
    from dashboard.ai import SYSTEM
    assert "untrusted" in SYSTEM and "ONLY the facts" in SYSTEM


DB = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not DB, reason="set TEST_DATABASE_URL to run database tests")
def test_store_and_freeze():
    from sqlalchemy import create_engine, text
    from dashboard.ai import store
    eng = create_engine(DB.replace("postgresql://", "postgresql+psycopg://"))
    with eng.begin() as con:
        con.execute(text("DROP SCHEMA IF EXISTS ai CASCADE"))
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]
    reply = lambda p, schema: r1(cands, {}) if schema is Round1 else \
        {"headline": "H", "brief": "It is 11.0%.", "fact_ids_used": ["f1"]}
    a = store(eng, s, generate(s, cands, MockLLM(reply)), freeze=True)
    b = store(eng, s, generate(s, cands, MockLLM(reply)), freeze=True)
    with eng.connect() as con:
        frozen = con.execute(text("SELECT brief_id FROM ai.brief WHERE frozen")).scalars().all()
        n_att = con.execute(text("SELECT count(*) FROM ai.attempt")).scalar()
    assert frozen == [b] and a != b and n_att == 2                   # only the newest frozen brief


def test_invalid_answer_counts_as_failed_attempt_not_crash():
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]
    calls = {"n": 0}
    def reply(prompt, schema):
        if schema is Round1:
            return r1(cands, {})
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("model returned invalid JSON twice")
        return {"headline": "H", "brief": "It is 11.0%.", "fact_ids_used": ["f1"]}
    res = generate(s, cands, MockLLM(reply))
    assert [a["passed"] for a in res.attempts] == [False, True] and not res.fallback_used


def test_pick_diverse_skips_same_theme():
    from dashboard.ai import pick_diverse
    a = insight(1); a.group = "market:cost"; a.type = "market_position"
    b = insight(2); b.group = "market:cost"; b.type = "market_position"       # Brent after jet fuel
    c = insight(3); c.type = "exposure"
    d = insight(4); d.type = "peer_gap"
    assert [x.id for x in pick_diverse([a, b, c, d], 3)] == ["t:1", "t:3", "t:4"]


def test_wording_check_flags_exaggeration_and_cause_but_not_plain_text():
    terms = load_settings().ai.flag_terms
    assert check_wording("Fuel hit a Historic high due to the conflict.", terms) == ["exaggeration: historic", "cause: due to"]
    assert check_wording("Jet fuel is the highest in 10 years; margin fell 2.1 points.", terms) == []
    assert check_wording("Recorded revenue rose.", terms) == []                 # whole words only: 'recorded' != 'record'


def test_flags_are_stored_but_do_not_block_a_correct_brief():
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]
    reply = lambda p, schema: r1(cands, {}) if schema is Round1 else \
        {"headline": "Record fuel", "brief": "It is 11.0% because of demand.", "fact_ids_used": ["f1"]}
    res = generate(s, cands, MockLLM(reply))
    assert res.check.passed and not res.fallback_used                          # numbers fine -> brief kept
    assert res.flags == ["exaggeration: record", "cause: because"]             # ...but counted as possible claim error
    assert res.attempts[0]["flags"] == res.flags


def test_certainty_wording_is_flagged_and_prompt_has_rule_6():
    from dashboard.ai import SYSTEM, PROMPT_VERSION
    terms = load_settings().ai.flag_terms
    assert check_wording("Higher fuel will cost United $379M.", terms) == ["certainty: will cost"]
    assert check_wording("Fuel is estimated to cost $379M based on past quarters.", terms) == []
    assert "never say that something WILL happen" in SYSTEM and PROMPT_VERSION == "v8"



def test_brief_can_be_about_any_company():
    from dashboard.ai import settings_for
    from dashboard.config import load_settings
    from dashboard.insights import peer_insights
    import pandas as pd
    s = settings_for(load_settings(), "DAL")
    assert s.focal_company == "DAL" and set(s.peers) == {"UAL", "AAL"}
    pg = pd.DataFrame([{"kpi": "operating_margin_pct", "label": "Operating margin", "unit": "%", "better": "higher",
                        "quarter_label": "Q2 2026", "ticker": t, "value": v, "rank": r, "n_companies": 3,
                        "peer_avg": 99.0, "gap_vs_peer_avg": 99.0}            # focal-relative columns must be ignored
                       for t, v, r in [("DAL", 9.0, 1), ("UAL", 6.0, 2), ("AAL", 3.0, 3)]])
    ins = peer_insights(pg, "DAL")[0]
    f = {x.id: x.value for x in ins.facts}
    assert f["peer.operating_margin_pct.peer_avg"] == 4.5 and f["peer.operating_margin_pct.gap"] == 4.5



def _narrative_items():
    from dashboard.insights import Fact, Insight
    return [Insight(id="exp", type="exposure", title="Exposure to jet fuel", summary="A 10% rise in jet fuel is estimated to cost $379M a quarter.",
                    facts=[Fact("exposure.fuel.impact_10pct", "fuel", -379e6, "USD")], significance=0.8, impact=1.0),
            Insight(id="mkt", type="market_position", title="Jet fuel high", summary="Jet fuel is $4.56/gal.",
                    facts=[Fact("jet_fuel.latest", "jet fuel", 4.56, "USD/gal")], significance=0.9, impact=1.0)]


def _round3(detail):
    return {"page_briefing": "Jet fuel is $4.56/gal.", "page_peers": "United is mid-pack.", "page_risks": "A 10% rise costs about $379M.",
            "page_whatif": "Shocks are estimated from past quarters.", "page_detail": "Margins moved.",
            "recommendations": [{"title": "Review fuel exposure for next quarter", "detail": detail, "page": 3}]}


def test_page_notes_and_options_are_checked():
    from dashboard.ai import MockLLM, Round3, narrate
    s = load_settings()
    good = narrate(s, _narrative_items(), MockLLM(lambda p, schema: _round3("Fuel is $4.56/gal and a 10% rise costs about $379M; trade-off: no gain if prices fall.")))
    assert good.check.passed and not good.fallback_used and good.recommendations[0]["page"] == 3
    assert set(good.pages) == {"page_briefing", "page_peers", "page_risks", "page_whatif", "page_detail"}
    bad = narrate(s, _narrative_items(), MockLLM(lambda p, schema: _round3("A rise costs $999M.")))
    # v5: only the failed option is dropped; the page notes (all correct) are kept as AI text
    assert not bad.fallback_used and "No AI recommendation" in bad.recommendations[0]["title"]
    assert bad.pages["page_risks"] == "A 10% rise costs about $379M." and bad.coverage["options"] == 0
    allbad = narrate(s, _narrative_items(), MockLLM(lambda p, schema: dict(_round3("A rise costs $999M."),
                     page_briefing="$881M", page_peers="$882M", page_risks="$883M", page_whatif="$884M", page_detail="$885M")))
    assert allbad.fallback_used and "$379M" in allbad.pages["page_risks"]   # template = the computed summary
    from dashboard.ai import round3_prompt
    prompt = round3_prompt(_narrative_items(), "United Airlines")
    assert "CONSIDER" in prompt and "trade-off" in prompt and 'display: "-$379M"' in prompt


# ---------------------------------------------------------------- v4 storytelling
def test_brief_is_a_three_step_story_and_parts_are_kept():
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]
    def reply(prompt, schema):
        if schema is Round1:
            return r1(cands, {})
        assert "situation" in prompt and "complication" in prompt and "resolution" in prompt
        return {"headline": "H", "situation": "Value is 11.0% now.", "complication": "That is a cost.",
                "resolution": "Consider the trend at 11.0%.", "fact_ids_used": ["f1"]}
    res = generate(s, cands, MockLLM(reply))
    assert res.brief == "Value is 11.0% now. That is a cost. Consider the trend at 11.0%."
    assert res.parts == {"situation": "Value is 11.0% now.", "complication": "That is a cost.", "resolution": "Consider the trend at 11.0%."}


def test_captions_and_start_here_are_number_checked():
    from dashboard.ai import CHARTS, MockLLM, narrate, round3_prompt
    s = load_settings()
    story = [{"text": "A 10% fuel rise costs about $379M.", "page": 3}, {"text": "Jet fuel is $4.56/gal.", "page": 2}]
    ok = dict(_round3("Fuel is $4.56/gal; trade-off: none."), story=story,
              captions={"jet_fuel": "Jet fuel is $4.56/gal today.", "fuel_exposure": "A 10% rise costs about $379M."})
    good = narrate(s, _narrative_items(), MockLLM(lambda p, schema: ok))
    assert not good.fallback_used and good.start_page == 3 and good.captions["jet_fuel"].startswith("Jet fuel")
    assert [c["page"] for c in good.chapters] == [3, 2]          # v7: start here = first chapter of the story
    bad = dict(ok, captions={"jet_fuel": "Jet fuel is $9.99/gal today.",                 # invented number
                             "fuel_exposure": "A 10% rise costs about $379M."})
    res = narrate(s, _narrative_items(), MockLLM(lambda p, schema: bad))
    assert "jet_fuel" not in res.captions and res.captions["fuel_exposure"].startswith("A 10%")
    assert res.start_reason and res.attempts == 2
    prompt = round3_prompt(_narrative_items(), "United Airlines")
    assert "story" in prompt and all(k in prompt for k in CHARTS)


# ---------------------------------------------------------------- v7 page stories
def _driver(strength="mostly", main="fuel"):
    from dashboard.insights import Fact, Insight
    return Insight(id="driver:operating_margin_pct", type="driver", title="Why operating margin changed",
                   summary="Operating margin fell 1.1 pp (Q3 2026 vs Q3 2025), mostly because fuel cost took 7.4 pp.",
                   facts=[Fact("why.operating_margin_pct.now", "UAL operating margin Q3 2026", 8.1, "%"),
                          Fact("why.operating_margin_pct.change", "change", -1.1, "pp"),
                          Fact("why.operating_margin_pct.fuel", "part from fuel", -7.4, "pp"),
                          Fact("why.operating_margin_pct.other_costs", "part from other costs", 6.3, "pp")],
                   significance=0.8, impact=0.9, group=f"{strength}:{main}")


def test_why_may_say_mostly_because_only_where_the_arithmetic_does():
    from dashboard.ai import _why_ok
    d, facts = [_driver()], _driver().facts
    assert _why_ok("Margin fell 1.1 pp, mostly because fuel cost took 7.4 pp.", d, facts)[0]
    assert not _why_ok("Margin fell 1.1 pp, mostly because of weak demand.", d, facts)[0]       # other cause
    assert not _why_ok("Margin fell 1.1 pp, mostly because fuel took 7.4 pp.", [_driver("partly")], facts)[0]
    assert not _why_ok("Margin fell because of fuel.", [_driver("mixed", "")], facts)[0]       # no single driver
    assert _why_ok("Margin fell 1.1 pp with no single driver: fuel and other costs moved in opposite ways.", [_driver("mixed", "")], facts)[0]
    assert not _why_ok("Operating margin fell 1.1 pp in Q3 2026.", d, facts)[0]                 # names no driver


def test_page_story_news_must_match_the_page_topic_and_is_written_by_code():
    import datetime as dt
    from dashboard.ai import MockLLM, narrate
    s = load_settings()
    items = _narrative_items() + [_driver()]
    news = [{"id": "n1", "title": "Jet fuel jumps on refinery outages", "outlet": "Reuters", "date": dt.date(2026, 10, 22),
             "topic": "fuel", "url": "https://x/1"},
            {"id": "n2", "title": "Pilots vote to strike", "outlet": "Skift", "date": dt.date(2026, 10, 21),
             "topic": "labour", "url": "https://x/2"}]
    reply = dict(_round3("Fuel is $4.56/gal; trade-off: none."),
                 page_risks={"back": "Page 1 showed fuel is the main pressure.", "what": "A 10% rise costs about $379M.",
                             "why": "Margin fell 1.1 pp, mostly because fuel cost took 7.4 pp.", "news_id": "n1",
                             "todo": "Check fuel options on page 4.", "next_page": 4},
                 page_whatif={"what": "Shocks are estimated from past quarters.", "news_id": "n2"})   # labour not allowed here
    res = narrate(s, items, MockLLM(lambda p, schema: reply), news)
    r3 = res.stories["page_risks"]
    assert r3["why"].startswith("Margin fell") and r3["next_page"] == 4 and r3["ai"]["why"]
    assert r3["news"] == "In the news: Reuters, 22 Oct: \u201cJet fuel jumps on refinery outages\u201d"
    assert "Pilots" not in res.stories["page_whatif"]["news"]
    assert res.coverage["news"] >= 1
    # the link back is code-written, word for word from the previous page's 'what'
    assert r3["back"] == "Page 2: " + res.stories["page_peers"]["what"] and not r3["ai"]["back"]


# ---------------------------------------------------------------- time limits (daily run must finish)
def test_used_up_time_budget_gives_template_without_calling_the_model():
    import time
    from dashboard.ai import narrate
    s = load_settings()
    calls = []
    llm = MockLLM(lambda p, schema: calls.append(schema) or {})
    llm.deadline = time.time() - 1                                   # budget already spent
    res = generate(s, [insight(i) for i in range(1, 4)], llm)
    assert res.fallback_used and res.check.passed and not calls      # round 1 skipped, round 2 template
    n = narrate(s, _narrative_items(), llm)
    assert n.fallback_used and not calls


def test_round1_failure_ranks_on_computed_parts_only():
    s = load_settings()
    cands = [insight(i) for i in range(1, 4)]

    def reply(p, schema):
        if schema is Round1:
            raise TimeoutError("timed out")
        return {"headline": "H", "brief": "Summary.", "fact_ids_used": []}
    res = generate(s, cands, MockLLM(reply))
    assert all(c.relevance == 0.0 for c in res.candidates) and not res.fallback_used


def test_timeouts_are_configured():
    s = load_settings()
    assert 10 <= s.ai.request_timeout_s <= 600 and s.ai.time_budget_s <= 1200


# ---------------------------------------------------------------- v5 chart stories
def test_chart_story_has_action_title_and_so_what_caption_each_checked_alone():
    from dashboard.ai import CHART_ROOM, MockLLM, narrate, round3_prompt
    s = load_settings()
    seen = []

    def reply(p, schema):
        seen.append(p)
        title = "United loses most if fuel rises: $379M a quarter" if len(seen) > 1 else "Fuel costs $999M"
        return dict(_round3("Fuel is $4.56/gal; trade-off: none."), captions={
            "fuel_exposure": {"title": title, "caption": "That is the estimated quarterly cost of a 10% rise."},
            "peer_bars": {"title": "x" * (CHART_ROOM["peer_bars"][0] + 1), "caption": ""}})
    res = narrate(s, _narrative_items(), MockLLM(reply))
    assert res.titles["fuel_exposure"].startswith("United loses most")      # fixed in the targeted rewrite
    assert "fuel_exposure title" in seen[1] and "Rewrite ONLY those items" in seen[1]
    assert "peer_bars" not in res.titles                                     # too long for its panel: dropped
    assert res.coverage["titles"] == "1/8" and res.coverage["captions"] == "1/8"
    prompt = round3_prompt(_narrative_items(), "United Airlines")
    assert "evidence insights: exp" in prompt and "action title" in prompt and "so-what" in prompt


# ---------------------------------------------------------------- v6 numbers must match THEIR chart
def test_number_from_another_chart_is_rejected_even_if_it_is_a_real_fact():
    from dashboard.ai import MockLLM, narrate
    s = load_settings()
    story = {"fuel_exposure": {"title": "Jet fuel is $4.56/gal", "caption": "", "fact_ids": []},     # jet-fuel chart's number
             "jet_fuel": {"title": "Jet fuel at $4.56/gal", "caption": "", "fact_ids": ["jet_fuel.latest"]},
             "sensitivity": {"title": "A 10% rise costs $379M", "caption": "", "fact_ids": ["jet_fuel.latest"]}}  # cites another chart
    res = narrate(s, _narrative_items(), MockLLM(lambda p, schema: dict(_round3("Fuel is $4.56/gal; trade-off: none."),
                                                                        captions=story)))
    assert res.titles == {"jet_fuel": "Jet fuel at $4.56/gal"}
    assert res.coverage["titles"] == "1/8"


def test_double_negative_and_start_here_on_page_1_are_rejected():
    from dashboard.ai import MockLLM, _item_ok, narrate
    assert not _item_ok("A 10% rise reduces operating income by -$379M.", _narrative_items()[0].facts)[0]
    assert _item_ok("A 10% rise cuts operating income by $379M.", _narrative_items()[0].facts)[0]
    s = load_settings()
    res = narrate(s, _narrative_items(), MockLLM(lambda p, schema: dict(_round3("Fuel is $4.56/gal; trade-off: none."),
                                                                        start_page=1, start_reason="Fuel is $4.56/gal.")))
    assert res.start_reason == "" and res.coverage["start_here"] is False


def test_chapter_and_back_checks():
    from dashboard.ai import _back_ok, _chapter_ok
    from dashboard.insights import Fact
    f = [Fact("x", "UAL operating margin", 6.2, "%")]
    assert not _chapter_ok("Margin is 6.2%. That is mostly...", f, 2)[0]        # unfinished
    assert not _chapter_ok("Jet fuel is $4.63 a gallon.", f, 2)[0]               # number not on page 2
    assert _chapter_ok("UAL margin is 6.2% in Q2 2026, ahead of peers.", f, 2)[0]
    assert not _chapter_ok("UAL margin is 6.2%, ahead of peers.", f, 2)[0]       # peer claim without quarter
    assert not _back_ok("Page 2 showed UAL peer positioning.", f)[0]             # vague, no number
    assert _back_ok("Page 2 showed UAL's margin is 6.2%.", f)[0]


def test_todo_and_sign_checks():
    from dashboard.ai import _item_ok, _todo_ok
    from dashboard.insights import Fact
    f = [Fact("x", "UAL net debt change", -354e6, "USD")]
    assert not _todo_ok("Weigh protection against hedging costs.", f)[0]
    assert _todo_ok("Review net debt, which fell $354M.", f)[0]
    assert not _item_ok("Net debt fell -$354M.", f)[0]
    assert _item_ok("Net debt fell $354M.", f)[0]


def test_peer_claims_name_their_quarter():
    from dashboard.ai import _peer_quarter_ok
    assert not _peer_quarter_ok("UAL free cash flow reaches $267M, ranking first among peers")
    assert _peer_quarter_ok("UAL free cash flow of $267M ranked first among peers in Q2 2026")
    assert _peer_quarter_ok("Free cash flow fell $1.43B in Q3 2026")


def test_why_explains_the_same_metric():
    from dashboard.ai import _same_metric, _why_ok
    from dashboard.insights import Fact, Insight
    nd = Insight(id="driver:net_debt", type="driver", title="Why net debt changed", summary="", group="mostly:liquidity", significance=0.8, impact=0.9,
                 facts=[Fact("why.net_debt.change", "Net debt change", -354e6, "USD"),
                        Fact("why.net_debt.liquidity", "Part from cash", -1.32e9, "USD")])
    fcf = Insight(id="driver:free_cash_flow", type="driver", title="Why free cash flow changed", summary="", significance=0.8, impact=0.9,
                  group="mostly:operating_cash_flow",
                  facts=[Fact("why.free_cash_flow.change", "FCF change", -1.43e9, "USD"),
                         Fact("why.free_cash_flow.operating_cash_flow", "Part from OCF", -753e6, "USD")])
    facts = nd.facts + fcf.facts
    drv = _same_metric([nd, fcf], "Net debt fell $354M in Q3 2026.")
    assert not _why_ok("Mostly because operating cash flow fell $753M.", drv, facts)[0]      # FCF's why on a net-debt page
    assert _why_ok("Net debt fell $354M, mostly because cash rose $1.32B.", drv, facts)[0]
    assert not _why_ok("Operating income changed due to revenue and cost shifts.", [nd, fcf], facts)[0]


def test_quoted_numbers_are_unquoted_and_sign_checked():
    from dashboard.ai import Chapter, _item_ok, _unquote
    from dashboard.insights import Fact
    c = _unquote(Chapter(text='So a 10% rise cuts income by "-$417M".', page=3))
    assert c.text == "So a 10% rise cuts income by $417M."               # quotes and the extra minus removed
    assert _item_ok(c.text, [Fact("x", "impact", -417e6, "USD")])[0]
    assert not _item_ok("Fuel took -7.4 pp of margin.", [Fact("y", "fuel part", -7.4, "pp")])[0]   # untidied text is still caught


def test_down_word_minus_is_tidied_but_rose_minus_is_not():
    from dashboard.ai import Chapter, _unquote
    t = lambda x: _unquote(Chapter(text=x, page=3)).text
    assert t("Net debt fell -$354M in Q3 2026.") == "Net debt fell $354M in Q3 2026."
    assert t("A 10% rise reduces quarterly operating income by about -$417M.") == "A 10% rise reduces quarterly operating income by about $417M."
    assert t("Fuel cost took -7.4 pp of margin.") == "Fuel cost took 7.4 pp of margin."
    assert t("Net debt rose -$354M.") == "Net debt rose -$354M."


def test_long_gap_down_word_minus_is_tidied():
    from dashboard.ai import Chapter, _unquote
    t = _unquote(Chapter(text="So a 10% rise in jet fuel reduces UAL quarterly operating income by -$417M.", page=4)).text
    assert t.endswith("by $417M.")


def test_chapters_are_tidied():
    from dashboard.ai import _tidy_chapter
    assert _tidy_chapter("UAL Free Cash Flow Reaches $267M, Ranking 1 Among Peers (Q2 2026).") == \
        "UAL free cash flow reaches $267M, ranking 1 among peers (Q2 2026)."
    assert _tidy_chapter("Against peers, DAL net debt fell to $9.06B in Q3 2026.") == "DAL net debt fell to $9.06B in Q3 2026."
    assert _tidy_chapter("Against peers, AAL operating margin ranks 3 of 3 at 2.7% (Q2 2026).").startswith("Against peers")


def test_step_repeating_the_headline_is_detected():
    from dashboard.ai import _same_point
    h = "UAL free cash flow reaches $267M, ranking 1 of 3 among peers (Q2 2026)"
    assert _same_point("UAL free cash flow reaches $267M, ranking 1 of 3 among peers in Q2 2026.", h)
    assert not _same_point("A 10% rise in jet fuel reduces quarterly operating income by $417M.", h)


def test_what_to_do_is_code_written_and_follows_the_data():
    from dashboard.ai import code_todos
    from dashboard.insights import Fact, Insight
    def peer(gap):
        return Insight(id="peer:operating_margin_pct", type="peer_gap", title="t", summary="s", significance=0.5,
                       impact=0.5, facts=[Fact("peer.operating_margin_pct.gap", "gap", gap, "pp")])
    lead, behind = code_todos([peer(5.0)], "Delta Air Lines, Inc."), code_todos([peer(-5.0)], "American Airlines Group Inc.")
    assert "Delta Air Lines' margin lead" in lead["page_peers"][0] and "cutting costs" not in lead["page_peers"][0]
    assert "American Airlines' margin gap" in behind["page_peers"][0] and "cutting costs" in behind["page_peers"][0]
    assert [v[1] for v in lead.values()] == [3, 4, 5, 1]            # always points to the next page


def test_against_peers_with_only_a_time_comparison_is_dropped():
    from dashboard.ai import _tidy_chapter
    assert _tidy_chapter("Against peers, net debt fell $354M in Q3 2026 vs a year earlier.") == "Net debt fell $354M in Q3 2026 vs a year earlier."


def test_recency_lowers_older_quarter_insights():
    from dashboard.insights import Fact, Insight, recency
    mk = lambda i, lab: Insight(id=i, type="t", title="t", summary="s", significance=0.5, impact=0.5,
                                facts=[Fact(i, lab, 1.0, "%")])
    r = recency([mk("old", "UAL FCF (Q2 2026)"), mk("new", "UAL margin Q3 2026"), mk("mkt", "Jet fuel price")])
    assert r == {"old": 0.8, "new": 1.0, "mkt": 1.0}


def test_period_names_are_not_figures():
    from dashboard.ai import check_numbers
    from dashboard.insights import Fact
    assert check_numbers("Replaying the last 12 months costs $4.47B.", [Fact("x", "impact", -4.47e9, "USD")]).passed


def test_a_level_keeps_its_minus():
    from dashboard.ai import Chapter, _unquote
    t = lambda x: _unquote(Chapter(text=x, page=5)).text
    assert t("Free cash flow fell to -$351M in Q2 2026.") == "Free cash flow fell to -$351M in Q2 2026."
    assert t("Free cash flow fell -$815M to -$351M.") == "Free cash flow fell $815M to -$351M."


def test_why_quarter_names_the_quarter_of_the_calculated_why():
    from dashboard.ai import _why_quarter
    from dashboard.insights import Fact, Insight
    w = Insight(id="driver:operating_margin_pct", type="driver", title="Why operating margin changed", summary="",
                facts=[Fact("why.operating_margin_pct.change", "Operating margin change, Q3 2026 vs Q3 2025", -1.1, "pp")],
                significance=0.8, impact=0.9)
    assert _why_quarter([w], "Operating margin fell 1.1 pp as fuel took 7.4 pp.") == "Q3 2026 vs Q3 2025"
    assert _why_quarter([], "anything") == ""
