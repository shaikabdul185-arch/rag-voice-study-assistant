"""Load notes / exam papers, split them into page-aware chunks, embed, and store."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from .db import KINDS
from .embeddings import Embedder

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}
# Page breaks in text files: a form feed, or a line that is exactly "---"/"<!-- page -->".
_PAGE_BREAK = re.compile(r"\f|^\s*(?:---|<!--\s*page\s*-->)\s*$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_FOLDER_KINDS = {
    "notes": "notes", "note": "notes",
    "exams": "exam", "exam": "exam", "papers": "exam",
    "keys": "answer_key", "key": "answer_key", "answer_keys": "answer_key", "answers": "answer_key",
}


@dataclass(frozen=True)
class Chunk:
    page: int  # 1-based
    index: int  # position within the document
    text: str


@dataclass(frozen=True)
class IngestResult:
    path: str
    status: str  # "ingested" | "unchanged" | "skipped"
    chunks: int = 0
    detail: str = ""


def load_pages(path: Path) -> list[str]:
    """Return the document's text, one string per page."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return [page.extract_text() or "" for page in PdfReader(path).pages]
    text = path.read_text(encoding="utf-8", errors="replace")
    return _PAGE_BREAK.split(text)


def _split_long(text: str, size: int) -> list[str]:
    """Split a paragraph longer than `size` on sentence boundaries, then word boundaries,
    hard-wrapping only a single word that is itself longer than `size`."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size + 1)
            if cut <= 0:
                cut = size
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if sentence:
            pieces.append(sentence)
    return pieces


def chunk_page(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Greedy paragraph packing with a character overlap carried between chunks."""
    text = text.strip()
    if not text:
        return []
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            units.extend(_split_long(para, size) if len(para) > size else [para])

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        # Start the overlap at a word boundary so chunks don't begin mid-word.
        if " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        current = f"{tail} {unit}".strip() if tail else unit
        if len(current) > size:  # the overlap pushed it over; drop the overlap
            current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_document(pages: list[str], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one page."""
    out: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        for text in chunk_page(page_text, size, overlap):
            out.append(Chunk(page=page_no, index=len(out), text=text))
    return out


def infer_kind(path: Path) -> str | None:
    for part in reversed(path.parent.parts):
        kind = _FOLDER_KINDS.get(part.lower())
        if kind:
            return kind
    name = path.stem.lower()
    if "key" in name or "solution" in name or "answers" in name:
        return "answer_key"
    if "exam" in name or "paper" in name or "midterm" in name or "final" in name:
        return "exam"
    return None


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED))
        elif p.is_file():
            files.append(p)
    return files


def ingest_file(
    conn: psycopg.Connection,
    embedder: Embedder,
    path: Path,
    kind: str | None = None,
    size: int = 1000,
    overlap: int = 150,
) -> IngestResult:
    if path.suffix.lower() not in SUPPORTED:
        return IngestResult(str(path), "skipped", detail=f"unsupported type {path.suffix}")
    kind = kind or infer_kind(path) or "notes"
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    key = str(path.resolve())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = conn.execute("SELECT sha256, kind FROM documents WHERE path = %s", (key,)).fetchone()
    if row and row[0] == digest and row[1] == kind:
        return IngestResult(str(path), "unchanged")

    pages = load_pages(path)
    chunks = chunk_document(pages, size, overlap)
    if not chunks:
        return IngestResult(str(path), "skipped", detail="no extractable text (scanned PDF?)")
    vectors = embedder.embed([c.text for c in chunks])

    with conn.transaction():
        doc_id = conn.execute(
            """
            INSERT INTO documents (path, title, kind, sha256, num_pages)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (path) DO UPDATE
                SET title = EXCLUDED.title, kind = EXCLUDED.kind, sha256 = EXCLUDED.sha256,
                    num_pages = EXCLUDED.num_pages, ingested_at = now()
            RETURNING id
            """,
            (key, path.name, kind, digest, len(pages)),
        ).fetchone()[0]
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, chunk_index, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [(doc_id, c.page, c.index, c.text, vec) for c, vec in zip(chunks, vectors)],
            )
    return IngestResult(str(path), "ingested", chunks=len(chunks), detail=kind)
