import os

os.environ["EMBED_MODEL"] = "hash"

from rag.chunk import chunk, split_text
from rag.embed import embed
from rag.load import Page
from rag.pipeline import Answer


def test_split_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    parts = split_text(text, size=400, overlap=50)
    assert all(len(p.split()) <= 300 for p in parts)  # 400 tokens ~ 300 words
    assert parts[0].split()[-37:] == parts[1].split()[:37]  # 50 tokens ~ 37 words shared
    assert parts[0].startswith("w0 ") and parts[-1].endswith("w999")


def test_short_text_is_one_chunk():
    assert split_text("just a few words", 400, 50) == ["just a few words"]


def test_chunks_keep_source_and_page():
    cs = chunk([Page("alpha beta", "a.pdf", 3), Page("gamma", "n.md", None)])
    assert cs[0].label == "a.pdf, p. 3"
    assert cs[1].label == "n.md"
    assert len({c.id for c in cs}) == 2


def test_hash_embed_is_normalised_and_similar_texts_score_higher():
    a, b, c = embed(["frozen storage temperature", "temperature of frozen storage", "canteen hours"], "hash")
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))
    assert abs(dot(a, a) - 1) < 1e-9
    assert dot(a, b) > dot(a, c)


def test_grounded_flag():
    assert Answer("I don't know").grounded
    assert not Answer("It is 4 C.").grounded  # no citation
    assert not Answer("It is 4 C [3].", hits=[object()]).grounded  # cites a passage it never saw
    assert Answer("It is 4 C [1].", hits=[object()]).grounded
