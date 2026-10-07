"""Settings loaded from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

EMBED_DIM = 384  # all-MiniLM-L6-v2; the schema's vector(...) column must match


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, real env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    database_url: str
    claude_model: str
    claude_effort: str
    embed_model: str
    embed_backend: str
    whisper_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 150


def load_settings() -> Settings:
    _load_dotenv(Path.cwd() / ".env")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql://study:study@localhost:5432/study"),
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=os.environ.get("CLAUDE_EFFORT", "medium"),
        embed_model=os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_backend=os.environ.get("EMBED_BACKEND", "sentence-transformers"),
        whisper_model=os.environ.get("WHISPER_MODEL", "base.en"),
    )
