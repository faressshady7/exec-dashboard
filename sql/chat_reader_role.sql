-- ============================================================================
-- Chatbot database account: run ONCE in Supabase (SQL Editor -> New query -> Run).
-- 1. Replace CHANGE-ME below with a long random password (letters and digits only).
-- 2. Put the connection string in .env as CHAT_DATABASE_URL (see docs/CHATBOT.md).
--
-- Least privilege: the chatbot can READ the bi views (what the dashboard shows) and
-- ADD rows to its own log. It cannot read silver/gold/ai, change or delete anything,
-- or read back other people's chats. The daily pipeline re-grants the bi views after
-- rebuilding them (dashboard.bi.grant_chat_reader).
-- ============================================================================

CREATE ROLE chat_reader LOGIN PASSWORD 'CHANGE-ME';
ALTER ROLE chat_reader SET statement_timeout = '5s';        -- no long-running queries

-- READ: the dashboard views only
CREATE SCHEMA IF NOT EXISTS bi;
GRANT USAGE ON SCHEMA bi TO chat_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA bi TO chat_reader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bi TO chat_reader;    -- formatting functions used inside the views

-- WRITE: its own question log, insert only (no SELECT: chats cannot be read back)
CREATE SCHEMA IF NOT EXISTS ai;
CREATE TABLE IF NOT EXISTS ai.chat (
    chat_id       BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id    TEXT,
    ticker        TEXT,
    page          TEXT,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    tools_used    JSONB NOT NULL DEFAULT '[]'::jsonb,
    sources       JSONB NOT NULL DEFAULT '[]'::jsonb,
    check_passed  BOOLEAN,
    unmatched     JSONB NOT NULL DEFAULT '[]'::jsonb,
    rewritten     BOOLEAN NOT NULL DEFAULT FALSE,
    general       BOOLEAN NOT NULL DEFAULT FALSE,
    flags         JSONB NOT NULL DEFAULT '[]'::jsonb,
    model         TEXT,
    latency_ms    INTEGER
);
GRANT USAGE ON SCHEMA ai TO chat_reader;
GRANT INSERT ON ai.chat TO chat_reader;
GRANT USAGE ON SEQUENCE ai.chat_chat_id_seq TO chat_reader;
