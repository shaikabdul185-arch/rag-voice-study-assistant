"""Vector search over stored chunks, returning page-cited hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import psycopg

from .embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    source: str  # file name, e.g. "os_notes.pdf"
    page: int
    kind: str
    text: str
    score: float  # cosine similarity, higher is closer

    @property
    def citation(self) -> str:
        return f"[{self.source} p.{self.page}]"


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, kinds: Sequence[str] | None = None) -> list[Hit]: ...


class PgVectorRetriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder):
        self.conn = conn
        self.embedder = embedder

    def search(self, query: str, k: int = 5, kinds: Sequence[str] | None = None) -> list[Hit]:
        vec = self.embedder.embed([query])[0]
        sql = """
            SELECT d.title, c.page, d.kind, c.content, 1 - (c.embedding <=> %(v)s) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %(v)s
            LIMIT %(k)s
        """
        params: dict = {"v": vec, "k": k}
        where = ""
        if kinds:
            where = "WHERE d.kind = ANY(%(kinds)s)"
            params["kinds"] = list(kinds)
        rows = self.conn.execute(sql.format(where=where), params).fetchall()
        return [Hit(source=r[0], page=r[1], kind=r[2], text=r[3], score=float(r[4])) for r in rows]


def format_hits(hits: Sequence[Hit]) -> str:
    """Render hits as the context block the model reads, one citation header per chunk."""
    if not hits:
        return "No matching passages found in the indexed notes."
    blocks = [f"{h.citation} ({h.kind}, similarity {h.score:.2f})\n{h.text}" for h in hits]
    return "\n\n---\n\n".join(blocks)
