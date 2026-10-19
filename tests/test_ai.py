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
