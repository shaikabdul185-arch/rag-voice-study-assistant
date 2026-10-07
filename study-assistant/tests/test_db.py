"""Integration test against a real Postgres + pgvector. Skipped unless TEST_DATABASE_URL is set.

Uses the offline hash embedder so no model download is needed.
WARNING: it drops and recreates the tables, so point it at a throwaway database.
"""
import os
from pathlib import Path

import pytest

from study_assistant.db import connect, init_schema, stats
from study_assistant.embeddings import HashEmbedder
from study_assistant.ingest import ingest_file, iter_files
from study_assistant.retrieval import PgVectorRetriever

URL = os.environ.get("TEST_DATABASE_URL")
SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"

pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def conn():
    with connect(URL) as c:
        c.execute("DROP TABLE IF EXISTS chunks, documents")
        c.commit()
        init_schema(c)
        yield c


def test_ingest_and_search(conn):
    emb = HashEmbedder()
    results = [ingest_file(conn, emb, f) for f in iter_files([SAMPLE])]
    assert all(r.status == "ingested" for r in results)
    assert {k for k, _, _ in stats(conn)} == {"notes", "exam", "answer_key"}

    # Re-ingesting unchanged files is a no-op.
    assert all(ingest_file(conn, emb, f).status == "unchanged" for f in iter_files([SAMPLE]))

    hits = PgVectorRetriever(conn, emb).search("Coffman conditions deadlock circular wait", k=3, kinds=["notes"])
    assert hits[0].source == "operating_systems.md" and hits[0].page == 2

    key_hits = PgVectorRetriever(conn, emb).search("Q2 FCFS SJF average waiting time", k=1, kinds=["answer_key"])
    assert key_hits[0].source == "midterm_2025_key.md" and key_hits[0].page == 1
