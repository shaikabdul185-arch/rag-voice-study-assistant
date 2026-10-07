"""PostgreSQL + pgvector connection and schema."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from .config import EMBED_DIM

KINDS = ("notes", "exam", "answer_key")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('notes', 'exam', 'answer_key')),
    sha256      TEXT NOT NULL,
    num_pages   INT  NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id INT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INT  NOT NULL,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS documents_kind ON documents (kind);
"""


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # The vector type doesn't exist until init_schema() creates the extension.
        pass
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    with conn.transaction():
        conn.execute(SCHEMA)


def stats(conn: psycopg.Connection) -> list[tuple[str, int, int]]:
    """(kind, documents, chunks) per kind."""
    return conn.execute(
        """
        SELECT d.kind, COUNT(DISTINCT d.id), COUNT(c.id)
        FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.kind ORDER BY d.kind
        """
    ).fetchall()
