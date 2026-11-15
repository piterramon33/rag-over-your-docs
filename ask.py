"""Ask a question:  python ask.py "What is the max temperature for fresh pork?"

Without a question it starts an interactive prompt. --offline needs no API key.
"""
import argparse

from rag import rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--offline", action="store_true", help="extractive answers, no LLM call")
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved chunks and scores")
    args = ap.parse_args()

    def answer(q: str) -> None:
        a = rag(q, offline=args.offline)
        if args.show_chunks:
            for i, h in enumerate(a.hits, 1):
                print(f"--- [{i}] {h.chunk.label}  score={h.score:.2f}\n{h.chunk.text[:300]}\n")
        print(a, "\n")

    if args.question:
        answer(" ".join(args.question))
        return
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
