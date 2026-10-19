"""Database connection (Supabase = hosted PostgreSQL).

The connection string holds your database PASSWORD, so it is never written in
settings.yaml or in the code. It lives in a file called `.env` in the project
folder (git-ignored), as one line:

    DATABASE_URL=postgresql+psycopg://postgres.<project-ref>:<PASSWORD>@aws-0-<region>.pooler.supabase.com:5432/postgres

Check the connection:  python -m dashboard.db
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from dashboard.config import PROJECT_ROOT, load_settings

SCHEMAS = ("silver", "gold", "ai")


def database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL") or load_settings().project.database_url
    # forgive common copy-paste mistakes (e.g. a GitHub secret pasted as the whole .env line)
    url = url.strip().strip('"').strip("'").strip()
    if url.upper().startswith("DATABASE_URL="):
        url = url.split("=", 1)[1].strip().strip('"').strip("'")
    # Supabase shows "postgresql://..."; SQLAlchemy needs the driver name
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        url = re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", url)
    return url


def masked(url: str) -> str:
    return re.sub(r"(://[^:/]+:)[^@]+@", r"\1*****@", url)


@lru_cache(maxsize=1)
def engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True, connect_args={"connect_timeout": 15})


def create_schemas(eng: Engine | None = None) -> None:
    with (eng or engine()).begin() as con:
        for s in SCHEMAS:
            con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {s}"))


def main() -> None:
    url = database_url()
    print(f"Connecting to {masked(url)} ...")
    if "localhost" in url:
        print("  (this is a LOCAL database - put your Supabase string in .env to use Supabase)")
    try:
        with engine().connect() as con:
            version = con.execute(text("select version()")).scalar()
        create_schemas()
        with engine().connect() as con:
            found = con.execute(text(
                "select schema_name from information_schema.schemata "
                "where schema_name = any(:s) order by 1"), {"s": list(SCHEMAS)}).scalars().all()
    except Exception as e:
        msg = str(e).splitlines()[0]
        print(f"\nFAILED: {msg}")
        if "password authentication failed" in msg:
            print("-> Wrong password in .env (reset it in Supabase: Project Settings > Database).")
        elif "could not translate host" in msg or "Network is unreachable" in msg or "timeout" in msg.lower():
            print("-> Use the SESSION POOLER string (port 5432, host ...pooler.supabase.com), "
                  "not the direct one - the direct host needs IPv6.")
        raise SystemExit(1)
    print(f"OK  {version.split(',')[0]}")
    print(f"Schemas ready: {', '.join(found)}")


if __name__ == "__main__":
    main()
