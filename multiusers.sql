-- multiusers.sql
-- Supabase setup for multiusers.py (멀티유저 + 멀티세션 RAG, user 테이블 인증)
-- Run in Supabase SQL Editor. 기존 multi-session-ref.sql(app_sessions)과 별도 스키마입니다.

DROP FUNCTION IF EXISTS match_vector_documents(vector(1536), int, uuid, text);
DROP FUNCTION IF EXISTS match_vector_documents(vector(1536), int, uuid, text, uuid);

DROP TABLE IF EXISTS vector_documents CASCADE;
DROP TABLE IF EXISTS chat_messages CASCADE;
DROP TABLE IF EXISTS chat_sessions CASCADE;
DROP TABLE IF EXISTS "user" CASCADE;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE "user" (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    login_id text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_login_id ON "user" (login_id);

CREATE TABLE chat_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES "user" (id) ON DELETE CASCADE,
    title text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_sessions_user_updated ON chat_sessions (user_id, updated_at DESC);

CREATE TABLE chat_messages (
    id bigserial PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES "user" (id) ON DELETE CASCADE,
    session_id uuid NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    position int NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_messages_user_session ON chat_messages (user_id, session_id);
CREATE INDEX idx_chat_messages_session_pos ON chat_messages (session_id, position);

CREATE TABLE vector_documents (
    id bigserial PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES "user" (id) ON DELETE CASCADE,
    session_id uuid REFERENCES chat_sessions (id) ON DELETE CASCADE,
    file_name text NOT NULL,
    content text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536) NOT NULL
);

CREATE INDEX idx_vector_documents_user_id ON vector_documents (user_id);
CREATE INDEX idx_vector_documents_user_session ON vector_documents (user_id, session_id);

CREATE INDEX idx_vector_documents_embedding ON vector_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

ALTER TABLE "user" DISABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions DISABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages DISABLE ROW LEVEL SECURITY;
ALTER TABLE vector_documents DISABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION match_vector_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 10,
    filter_session_id uuid DEFAULT NULL,
    filter_draft_key text DEFAULT NULL,
    filter_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    session_id uuid,
    file_name text,
    content text,
    similarity double precision
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        v.id,
        v.session_id,
        v.file_name,
        v.content,
        (1 - (v.embedding <=> query_embedding))::double precision AS similarity
    FROM vector_documents v
    WHERE
        (filter_user_id IS NOT NULL AND v.user_id = filter_user_id)
        AND (
            (
                filter_session_id IS NOT NULL
                AND v.session_id = filter_session_id
            )
            OR (
                filter_draft_key IS NOT NULL
                AND btrim(filter_draft_key) <> ''
                AND v.session_id IS NULL
                AND coalesce(v.metadata ->> 'draft_key', '') = filter_draft_key
            )
        )
    ORDER BY v.embedding <=> query_embedding
    LIMIT greatest(match_count, 1);
$$;

COMMENT ON TABLE "user" IS 'App-local accounts (not Supabase Auth)';
COMMENT ON TABLE chat_sessions IS 'Chat sessions per user';
COMMENT ON TABLE chat_messages IS 'Ordered messages per session';
COMMENT ON TABLE vector_documents IS 'RAG chunks; user_id required; session_id NULL = draft';
