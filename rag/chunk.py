"""Step 2: split pages into overlapping chunks.

Sizes are given in tokens and converted with the usual rule of thumb
(1 token ~ 0.75 English words), so no tokenizer download is needed.
Chunks never cross a page boundary, which is what lets every answer
cite one exact page.
"""
import re
from dataclasses import dataclass

from .load import Page

WORDS_PER_TOKEN = 0.75


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    n: int  # position of the chunk inside its page/file

    @property
    def id(self) -> str:
        return f"{self.source}:{self.page or 0}:{self.n}"

    @property
    def label(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source


def split_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    win = max(1, int(size * WORDS_PER_TOKEN))
    step = max(1, win - int(overlap * WORDS_PER_TOKEN))
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    out = []
    for start in range(0, len(spans), step):
        window = spans[start : start + win]
        out.append(text[window[0][0] : window[-1][1]])
        if start + win >= len(spans):
            break
    return out


def chunk(pages: list[Page], size: int = 400, overlap: int = 50) -> list[Chunk]:
    chunks = []
    for p in pages:
        for n, piece in enumerate(split_text(p.text, size, overlap)):
            chunks.append(Chunk(piece, p.source, p.page, n))
    return chunks
