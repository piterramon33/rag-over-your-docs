"""Regex eval: prints a pass rate you can quote.

    python eval.py eval_sets/sample.jsonl [--offline] [--min 0.8]

Each line of the test set is JSON:
    {"question": "...", "expected": "regex the answer must match"}
    {"question": "...", "answerable": false}      # must answer "I don't know"

An answerable question passes when the answer matches `expected` AND carries a
valid [n] citation. An unanswerable one passes when the app says "I don't know".
Start simple like this, then add an LLM judge.
"""
import argparse
import json
import re
import sys

from rag import rag
from rag.config import IDK


def passed(case: dict, answer) -> bool:
    refused = answer.text.strip().lower().startswith(IDK.lower())
    if case.get("answerable", True) is False:
        return refused
    cited = re.search(r"\[\d+\]", answer.text)
    correct = re.search(case["expected"], answer.text, re.I)
    return bool(cited and correct and answer.grounded and not refused)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_set")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--min", type=float, default=0.0, help="exit 1 if pass rate is below this (0-1)")
    args = ap.parse_args()

    with open(args.test_set, encoding="utf-8") as f:
        test_set = [json.loads(line) for line in f if line.strip()]

    n_pass = 0
    for case in test_set:
        a = rag(case["question"], offline=args.offline)
        ok = passed(case, a)
        n_pass += ok
        if not ok:
            print(f"FAIL  {case['question']}\n      got: {a.text[:160]!r}")
    rate = n_pass / len(test_set)
    print(f"pass rate: {rate:.0%} ({n_pass}/{len(test_set)})")
    sys.exit(0 if rate >= args.min else 1)


if __name__ == "__main__":
    main()
