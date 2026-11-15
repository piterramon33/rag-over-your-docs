"""Step 3: turn text into vectors.

EMBED_MODEL=hash is a dependency-free bag-of-words embedder used by the
tests and the offline demo. It is NOT semantic; use a real model for real docs.
"""
import hashlib
import math
import re
from functools import lru_cache

STOP = set(
    "a an the of to in on at for and or is are was were be been do does did what which who "
    "when where how why it its this that with as by from can should must i you we they".split()
)


def _hash_embed(text: str, dim: int = 512) -> list[float]:
    vec = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], model: str) -> list[list[float]]:
    if model == "hash":
        return [_hash_embed(t) for t in texts]
    vecs = _model(model).encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 64)
    return vecs.tolist()
