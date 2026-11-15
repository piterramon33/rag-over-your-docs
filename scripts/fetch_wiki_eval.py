"""Download rag-datasets/rag-mini-wikipedia (Hugging Face) as a ready-made corpus + eval set.

    python scripts/fetch_wiki_eval.py --questions 100
    python ingest.py data/wiki
    python eval.py eval_sets/wiki.jsonl

Writes one .txt per passage to data/wiki/ and eval_sets/wiki.jsonl.
Many gold answers are short ("yes", "1809"), so the regex check is rough here:
good for comparing two configurations, not as an absolute score.
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

NAME = "rag-datasets/rag-mini-wikipedia"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=100)
    args = ap.parse_args()

    out = Path("data/wiki")
    out.mkdir(parents=True, exist_ok=True)
    passages = load_dataset(NAME, "text-corpus")["passages"]
    for i, row in enumerate(passages):
        (out / f"passage_{row.get('id', i)}.txt").write_text(row["passage"], encoding="utf-8")

    qa = load_dataset(NAME, "question-answer")["test"]
    Path("eval_sets").mkdir(exist_ok=True)
    with open("eval_sets/wiki.jsonl", "w", encoding="utf-8") as f:
        for row in list(qa)[: args.questions]:
            f.write(json.dumps({"question": row["question"], "expected": re.escape(str(row["answer"]).strip())}) + "\n")
    print(f"{len(passages)} passages -> data/wiki, {min(args.questions, len(qa))} questions -> eval_sets/wiki.jsonl")
