"""Chat page opened from the Power BI button: "Ask about this page".

The button opens  http://localhost:8501/?company=UAL&page=3  so the chat starts on the airline
and the dashboard page the executive is looking at.

Run:  scripts\\run_chat.bat     (or: streamlit run src/dashboard/chat_app.py)
Needs in .env: GEMINI_API_KEY, CHAT_DATABASE_URL (read-only chat_reader role), optionally CHAT_PASSWORD.
"""
from __future__ import annotations

import hmac
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # streamlit runs this file directly

import streamlit as st
from dotenv import load_dotenv

from dashboard.chat import GeminiChat, Tools, answer, chat_engine, log
from dashboard.config import PROJECT_ROOT, load_settings

load_dotenv(PROJECT_ROOT / ".env")
PAGE_NAMES = ["Executive Briefing", "Peer Benchmark", "Risk Drivers", "What-if", "Company Detail"]
STARTERS = {
    "1": ["What changed most for us this quarter, and why?", "Which market risk matters most today?"],
    "2": ["Why is our margin different from Delta's?", "How does our valuation compare with the others?"],
    "3": ["How much does a 10% rise in jet fuel cost us per quarter?", "What does our 10-K say about fuel hedging?"],
    "4": ["What if fuel rises 20% and rates rise 1 point?", "How reliable is the fuel estimate?"],
    "5": ["Which quarter was unusual, and why?", "What is the latest news about our pilots?"],
}
BADGE = {"answered": "✓ Numbers checked against your data", "general": "General knowledge — not from your data",
         "number_fallback": "Shown: the data itself (an AI answer could not be verified)"}

st.set_page_config(page_title="Ask the dashboard", page_icon="💬", layout="centered")
s = load_settings()
tickers = [c.ticker for c in s.in_scope]


# ---------------------------------------------------------------- password (if CHAT_PASSWORD is set)
def gate() -> None:
    pw = os.environ.get("CHAT_PASSWORD", "")
    if not pw or st.session_state.get("ok"):
        return
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, pw):
        st.session_state["ok"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


gate()


# ---------------------------------------------------------------- backend (once per server)
@st.cache_resource
def backend(code_version: str):            # a new version string reloads the chat code without a restart
    eng = chat_engine()
    return eng, Tools(s, eng), GeminiChat(s)


try:
    eng, tools, model = backend("2026-10-31")
except SystemExit as e:                                 # missing key / wrong account: say what to fix
    st.error(f"Chat is not set up yet: {e}")
    st.stop()

# ---------------------------------------------------------------- where the user came from
qp = st.query_params
ss = st.session_state
ss.setdefault("sid", uuid.uuid4().hex[:12])
ss.setdefault("turns", [])                              # [(question, ChatResult)]
ss.setdefault("times", [])
company0 = qp.get("company", s.focal_company).upper()
page0 = qp.get("page", "1")
c1, c2 = st.columns(2)
ticker = c1.selectbox("Company", tickers, index=tickers.index(company0) if company0 in tickers else 0,
                      format_func=lambda t: s.company(t).name)
page = c2.selectbox("Dashboard page", [str(i) for i in range(1, 6)],
                    index=int(page0) - 1 if page0 in [str(i) for i in range(1, 6)] else 0,
                    format_func=lambda p: f"{p} {PAGE_NAMES[int(p) - 1]}")
st.caption("Answers come from the dashboard's data, the latest 10-K/10-Q and filtered news. "
           "Every number is checked. General knowledge is labelled. Not investment advice.")


# ---------------------------------------------------------------- show the conversation
def show(q, r) -> None:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(r.answer.replace("$", "\\$"))     # '$' would start a maths formula in Markdown
        if r.outcome in BADGE:
            st.caption(BADGE[r.outcome])
        if r.sources:
            with st.expander("Sources"):
                for src in r.sources:
                    label = f"{src.get('type')}: {src.get('title')}"
                    extra = " · ".join(x for x in (src.get("outlet"), src.get("date")) if x)
                    line = f"[{label}]({src['url']})" if src.get("url") else label
                    st.markdown(f"- {line}" + (f" ({extra})" if extra else ""))


for q, r in ss.turns:
    show(q, r)


# ---------------------------------------------------------------- ask
def limited() -> str | None:
    now = time.time()
    ss.times = [t for t in ss.times if now - t < 60]
    if len(ss.turns) >= s.chat.max_per_session:
        return "You've reached the question limit for this session. Reload the page to start a new one."
    if len(ss.times) >= s.chat.max_per_minute:
        return "Please wait a moment before asking again."
    return None


starter = None
if not ss.turns:
    cols = st.columns(2)
    for i, q in enumerate(STARTERS[page]):
        if cols[i].button(q, use_container_width=True):
            starter = q
question = st.chat_input("Ask about this page…") or starter
if question:
    if msg := limited():
        st.warning(msg)
    else:
        ss.times.append(time.time())
        with st.spinner("Looking at the data…"):
            hist = [(q, r.answer) for q, r in ss.turns]
            r = answer(question, ticker, s, tools, model, page=page, history=hist)
        log(eng, r, question, ticker, page, ss.sid)
        ss.turns.append((question, r))
        show(question, r)
