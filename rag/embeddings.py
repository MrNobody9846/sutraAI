import hashlib
import math
import re


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
DEFAULT_EMBEDDING_DIM = 384


def _tokens(text: str) -> list[str]:
    words = TOKEN_PATTERN.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def embed_text(text: str, dim: int | None = None) -> list[float]:
    """Deterministic local hashing embedding.

    This avoids model downloads during demos while still producing vectors that
    pgvector can rank by cosine distance. For production, swap this module for a
    real embedding model and keep the rest of the pipeline unchanged.
    """
    size = dim or DEFAULT_EMBEDDING_DIM
    vector = [0.0] * size
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"
