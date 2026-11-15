"""Step 6: ask the LLM to answer from the retrieved chunks only, with [n] citations."""
import re

from . import config
from .embed import STOP
from .index import Hit

SYSTEM = f"""You answer questions using ONLY the numbered context passages.
Rules:
- After every claim, cite the passage it came from as [n], e.g. "The limit is 4 C [2]."
- Use only passage numbers that appear in the context.
- If the context does not contain the answer, reply with exactly: {config.IDK}
- Do not use outside knowledge. Be concise."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.label})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def llm(question: str, hits: list[Hit]) -> str:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env, or run with --offline.")
    from openai import OpenAI

    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def offline_llm(question: str, hits: list[Hit]) -> str:
    """No-LLM stand-in: returns the sentence that best overlaps the question.

    Lets you run the whole pipeline and the eval without an API key.
    It copies text instead of reasoning, so treat its pass rate as a smoke test.
    """
    q = set(re.findall(r"[a-z0-9]+", question.lower())) - STOP
    best, best_score, best_n = "", 0, 1
    for n, h in enumerate(hits, 1):
        for sent in re.split(r"(?<=[.!?])\s+|\n+", h.chunk.text):
            score = len(q & set(re.findall(r"[a-z0-9]+", sent.lower())))
            if score > best_score:
                best, best_score, best_n = sent.strip(), score, n
    return f"{best} [{best_n}]" if best else config.IDK
