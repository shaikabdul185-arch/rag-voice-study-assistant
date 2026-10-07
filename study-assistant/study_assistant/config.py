"""Settings loaded from the environment (and an optional .env file)."""

from __future__ import annotations

import os
import re
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
        key, value = key.strip(), value.strip()
        if value[:1] in ('"', "'") and value[-1:] == value[:1] and len(value) > 1:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # drop inline "  # comment"
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
    use_fallbacks: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 150


def load_settings() -> Settings:
    # The current folder first, then the project folder (next to pyproject.toml), so `study`
    # finds your .env even when run from elsewhere. Real environment variables always win.
    _load_dotenv(Path.cwd() / ".env")
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql://study:study@localhost:5432/study"),
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=os.environ.get("CLAUDE_EFFORT", "medium"),
        embed_model=os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_backend=os.environ.get("EMBED_BACKEND", "sentence-transformers"),
        whisper_model=os.environ.get("WHISPER_MODEL", "base.en"),
        use_fallbacks=os.environ.get("CLAUDE_FALLBACKS", "on").lower() not in ("off", "0", "false", "no"),
    )
