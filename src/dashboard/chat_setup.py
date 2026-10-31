"""One-time chatbot setup, run on the laptop (scripts\\setup_chat.bat).

1. Creates (or resets) the read-only database account `chat_reader` with a RANDOM password,
   using sql/chat_reader_role.sql and your admin DATABASE_URL from .env. Nobody types the password.
2. Writes CHAT_DATABASE_URL to .env (same server, user chat_reader, the new password).
3. Checks the new account can read the dashboard views and cannot read anything else.
4. Asks for the Finnhub key if .env has none (Enter = skip).
Safe to run again: it just gives the account a new password.
"""
from __future__ import annotations

import os
import re
import secrets
import string

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from dashboard.config import PROJECT_ROOT

ENV = PROJECT_ROOT / ".env"
ROLE = "chat_reader"


def set_env(name: str, value: str, path=ENV) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [l for l in lines if not l.strip().startswith(name + "=")] + [f"{name}={value}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chat_url(admin_url: str, password: str) -> str:
    """Same server and database as the admin URL, user chat_reader. Supabase's pooler names users
    'postgres.<project-ref>', so that becomes 'chat_reader.<project-ref>'."""
    u = make_url(re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", admin_url))
    user = u.username or "postgres"
    new = ROLE + user[len("postgres"):] if user.startswith("postgres") else ROLE
    return u.set(username=new, password=password).render_as_string(hide_password=False)


def setup(admin_url: str) -> str:
    """Create or reset the account; return CHAT_DATABASE_URL."""
    pw = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    sql = (PROJECT_ROOT / "sql" / "chat_reader_role.sql").read_text(encoding="utf-8").replace("CHANGE-ME", pw)
    eng = create_engine(re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", admin_url))
    with eng.begin() as con:
        if con.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": ROLE}).first():
            sql = sql.replace(f"CREATE ROLE {ROLE} LOGIN PASSWORD", f"ALTER ROLE {ROLE} WITH LOGIN PASSWORD")
        con.execute(text(sql))
    from dashboard.bi import grant_chat_reader
    grant_chat_reader(eng)                       # the views that exist today (also re-granted every morning)
    eng.dispose()
    return chat_url(admin_url, pw)


def check(url: str) -> list[str]:
    """What the new account can and cannot do (each line printed for the user)."""
    out = []
    eng = create_engine(url)
    with eng.connect() as con:
        n = con.execute(text("SELECT count(*) FROM information_schema.views WHERE table_schema = 'bi'")).scalar()
        out.append(f"OK   reads the dashboard views ({n} visible)")
    for bad in ("SELECT 1 FROM gold.dim_company LIMIT 1", "SELECT 1 FROM ai.chat LIMIT 1"):
        try:
            with eng.connect() as con:
                con.execute(text(bad))
            out.append(f"FAIL could run: {bad}")
        except Exception:
            out.append(f"OK   blocked: {bad.split(' FROM ')[1].split()[0]}")
    eng.dispose()
    return out


def main() -> None:
    load_dotenv(ENV)
    admin = os.environ.get("DATABASE_URL", "").strip()
    if not admin:
        raise SystemExit("No DATABASE_URL in .env - the admin connection is needed once to create the chat account.")
    print("Creating the chatbot's read-only database account...")
    url = setup(admin)
    set_env("CHAT_DATABASE_URL", url)
    print("Saved CHAT_DATABASE_URL in .env (random password, never shown).")
    for line in check(url):
        print("  " + line)
    if not os.environ.get("FINNHUB_API_KEY", "").strip():
        key = input("\nPaste your Finnhub API key (right-click to paste), or press Enter to skip: ").strip()
        if key:
            set_env("FINNHUB_API_KEY", key)
            print(f"Saved. Key ends with ...{key[-4:]}")
    print("\nDone. Next: double-click scripts\\run_chat.bat, then Ctrl + click 'Ask AI about this page' in Power BI.")


if __name__ == "__main__":
    main()
