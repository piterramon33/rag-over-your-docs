"""Step 4: the vector index (Chroma, persisted on disk, cosine similarity)."""
from dataclasses import dataclass

import chromadb

from .chunk import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, 1.0 = identical


class VectorIndex:
    def __init__(self, path: str, collection: str):
        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vecs: list[list[float]], batch: int = 1000) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._col.add(
                ids=[c.id for c in part],
                embeddings=vecs[i : i + batch],
                documents=[c.text for c in part],
                metadatas=[{"source": c.source, "page": c.page or 0, "n": c.n} for c in part],
            )

    def count(self) -> int:
        return self._col.count()

    def search(self, query_vec: list[float], k: int = 5) -> list[Hit]:
        if self.count() == 0:
            return []
        res = self._col.query(query_embeddings=[query_vec], n_results=min(k, self.count()))
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            c = Chunk(text, meta["source"], meta["page"] or None, meta["n"])
            hits.append(Hit(c, 1.0 - dist))
        return hits
