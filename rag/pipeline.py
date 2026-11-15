"""The whole thing, in the same seven steps as the project card."""
import re
from dataclasses import dataclass, field

from . import config
from .chunk import chunk
from .embed import embed
from .index import Hit, VectorIndex
from .llm import llm, offline_llm
from .load import load


@dataclass
class Answer:
    text: str
    sources: dict[int, str] = field(default_factory=dict)  # citation number -> "file, p. N"
    hits: list[Hit] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True if it refuses, or cites at least one passage and only passages it was given."""
        if self.text.strip().lower().startswith(config.IDK.lower()):
            return True
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", self.text)}
        return bool(cited) and cited <= set(range(1, len(self.hits) + 1))

    def __str__(self) -> str:
        if not self.sources:
            return self.text
        lines = "\n".join(f"  [{n}] {label}" for n, label in sorted(self.sources.items()))
        return f"{self.text}\n\nSources:\n{lines}"


def _index() -> VectorIndex:
    return VectorIndex(config.INDEX_DIR, config.COLLECTION)


def ingest(folder: str = "my_docs") -> int:
    docs = load(folder)                                             # 1
    chunks = chunk(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)   # 2
    vecs = embed([c.text for c in chunks], config.EMBED_MODEL)      # 3
    index = _index()                                                # 4
    index.reset()
    index.add(chunks, vecs)
    return len(chunks)


def rag(question: str, offline: bool = False) -> Answer:
    qvec = embed([question], config.EMBED_MODEL)[0]
    hits = _index().search(qvec, k=config.TOP_K)                    # 5
    hits = [h for h in hits if h.score >= config.MIN_SCORE]
    if not hits:  # nothing relevant enough: refuse without spending tokens
        return Answer(config.IDK)
    text = (offline_llm if offline else llm)(question, hits)        # 6
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    sources = {n: hits[n - 1].chunk.label for n in cited if 1 <= n <= len(hits)}
    return Answer(text, sources, hits)
