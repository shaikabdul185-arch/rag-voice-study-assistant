"""Text embeddings.

The real backend is a local sentence-transformers model (no API key, runs on CPU).
The "hash" backend is a deterministic bag-of-words stub used by the tests so they
run offline without downloading model weights; it is not meant for real use.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import EMBED_DIM


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        dim = self.model.get_sentence_embedding_dimension()
        if dim != EMBED_DIM:
            raise ValueError(
                f"{model_name} produces {dim}-d vectors but the schema expects {EMBED_DIM}; "
                "change EMBED_DIM in config.py and re-run `study init-db` on a fresh database"
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


class HashEmbedder:
    """Feature-hashed bag of words, L2-normalized. Test stub only."""

    _token = re.compile(r"[a-z0-9]+")

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._token.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[row, h % EMBED_DIM] += 1.0 if (h >> 63) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@lru_cache(maxsize=4)
def get_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown EMBED_BACKEND {backend!r}")
