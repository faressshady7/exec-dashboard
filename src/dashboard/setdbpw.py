"""Change the Supabase admin password saved in .env (after resetting it in the Supabase dashboard).

Asks for the NEW password (typed hidden, never shown), puts it into DATABASE_URL in .env (same server and
user), and tests the connection. Run:  scripts\\set_db_password.bat
"""
from __future__ import annotations

import getpass
from urllib.parse import quote, urlsplit, urlunsplit

from dashboard.config import PROJECT_ROOT

ENV = PROJECT_ROOT / ".env"


def with_password(url: str, password: str) -> str:
    """Same URL, new password (URL-encoded, so @ : / % in the password are safe)."""
    p = urlsplit(url)
    if not p.username or not p.hostname:
        raise SystemExit("DATABASE_URL in .env does not look like postgresql://user:password@host:port/db")
    host = p.hostname + (f":{p.port}" if p.port else "")
    return urlunsplit((p.scheme, f"{p.username}:{quote(password, safe='')}@{host}", p.path, p.query, p.fragment))


def main() -> None:
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    old = next((l.split("=", 1)[1].strip().strip('"') for l in lines if l.startswith("DATABASE_URL=")), "")
    if not old:
        raise SystemExit("No DATABASE_URL in .env yet.")
    pw = getpass.getpass("Paste the NEW Supabase database password (it stays hidden), then press Enter: ").strip()
    if len(pw) < 8:
        raise SystemExit("That looks too short - nothing was saved.")
    new = with_password(old, pw)
    from sqlalchemy import create_engine, text
    url = new.replace("postgresql://", "postgresql+psycopg://", 1) if new.startswith("postgresql://") else new
    try:
        with create_engine(url).connect() as c:
            c.execute(text("select 1"))
    except Exception as e:
        raise SystemExit(f"Could not connect with the new password - nothing was saved. ({type(e).__name__})")
    ENV.write_text("\n".join([l for l in lines if not l.startswith("DATABASE_URL=")] + [f"DATABASE_URL={new}"]) + "\n",
                   encoding="utf-8")
    print("Connected. Saved the new DATABASE_URL in .env (password never shown).")


if __name__ == "__main__":
    main()
