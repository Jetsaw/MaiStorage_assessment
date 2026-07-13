CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS corpus (
    version uuid PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('staging', 'active', 'superseded', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz
);

CREATE TABLE IF NOT EXISTS active_corpus (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    version uuid NOT NULL REFERENCES corpus(version)
);

CREATE TABLE IF NOT EXISTS source (
    id uuid PRIMARY KEY,
    corpus_version uuid NOT NULL REFERENCES corpus(version),
    canonical_url text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('html', 'pdf')),
    title text NOT NULL,
    document_version text,
    product_version text,
    release_date date,
    retrieved_at timestamptz NOT NULL,
    content_hash text NOT NULL,
    raw_object_uri text NOT NULL,
    authority_rank smallint NOT NULL,
    status text NOT NULL CHECK (status IN ('active', 'superseded', 'rejected')),
    UNIQUE (corpus_version, canonical_url)
);

CREATE TABLE IF NOT EXISTS source_section (
    id uuid PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    heading_path text NOT NULL,
    page integer CHECK (page IS NULL OR page > 0),
    raw_text text NOT NULL,
    normalized_text text NOT NULL,
    section_hash text NOT NULL
);

CREATE TABLE IF NOT EXISTS product (
    id uuid PRIMARY KEY,
    corpus_version uuid NOT NULL REFERENCES corpus(version),
    name text NOT NULL,
    family text NOT NULL,
    category text NOT NULL,
    positioning text,
    source_section_id uuid NOT NULL REFERENCES source_section(id),
    UNIQUE (corpus_version, name)
);

CREATE TABLE IF NOT EXISTS product_spec (
    id uuid PRIMARY KEY,
    product_id uuid NOT NULL REFERENCES product(id) ON DELETE CASCADE,
    field text NOT NULL,
    numeric_value numeric,
    unit text,
    text_value text NOT NULL,
    is_estimated boolean NOT NULL DEFAULT false,
    source_section_id uuid NOT NULL REFERENCES source_section(id)
);

CREATE TABLE IF NOT EXISTS compatibility_rule (
    id uuid PRIMARY KEY,
    corpus_version uuid NOT NULL REFERENCES corpus(version),
    product_version text NOT NULL,
    field text NOT NULL,
    operator text NOT NULL CHECK (operator IN ('eq', 'gte', 'lte', 'contains', 'contains_all')),
    expected_value jsonb NOT NULL,
    source_section_id uuid NOT NULL REFERENCES source_section(id)
);

CREATE TABLE IF NOT EXISTS document_chunk (
    id uuid PRIMARY KEY,
    source_section_id uuid NOT NULL REFERENCES source_section(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    content text NOT NULL,
    textsearch tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector,
    token_count integer NOT NULL CHECK (token_count > 0),
    UNIQUE (source_section_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunk_textsearch_idx ON document_chunk USING gin (textsearch);
CREATE INDEX IF NOT EXISTS source_active_url_idx ON source (canonical_url) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS product_lookup_idx ON product (corpus_version, lower(name));

CREATE TABLE IF NOT EXISTS agent_run (
    id uuid PRIMARY KEY,
    question text NOT NULL,
    route text NOT NULL,
    answer text NOT NULL,
    citation_status text NOT NULL,
    trace jsonb NOT NULL,
    evidence jsonb NOT NULL,
    corpus_version uuid NOT NULL REFERENCES corpus(version),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_run_created_idx ON agent_run (created_at DESC);

CREATE TABLE IF NOT EXISTS chat_session (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_message (
    id bigserial PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL CHECK (length(content) > 0),
    provider text,
    model text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_message_session_idx ON chat_message (session_id, id);

-- ponytail: the initial corpus is tiny, so exact vector scans are sufficient.
-- Add a fixed embedding dimension and HNSW index after the model is selected and load tests justify it.
