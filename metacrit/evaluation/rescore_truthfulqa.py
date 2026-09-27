#!/usr/bin/env python3
"""Rescore a saved TruthfulQA jsonl with the current judge_truthfulqa prompt.

Reads results/truthfulqa__<method>.jsonl, calls GPT-4o on each row's saved
`prediction` against the gold lists, and rewrites the `score` field. The
model's answer is NOT re-generated -- only the judge is re-run.

A timestamped backup of the original file is written next to it before any edit.

Usage:
    python metacrit/evaluation/rescore_truthfulqa.py results/truthfulqa__dmc.jsonl
    python metacrit/evaluation/rescore_truthfulqa.py results/truthfulqa__reflectevo.jsonl
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # metacrit/
from run_baselines import LLM, JUDGE_MODEL, judge_truthfulqa


def rescore(jsonl_path):
    path = Path(jsonl_path)
    if not path.exists():
        sys.exit(f"not found: {path}")
    backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, backup)
    print(f"backup -> {backup}")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judge = LLM(JUDGE_MODEL)

    flipped, n = 0, len(rows)
    try:
        from tqdm import tqdm
        it = tqdm(rows, desc=path.name)
    except ImportError:
        it = rows
    for r in it:
        old = r.get("score")
        new = judge_truthfulqa(judge, r["question"], r["prediction"], r["gold"])
        if old != new:
            flipped += 1
        r["score"] = new

    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    truthful = sum(r["score"] for r in rows) / max(1, n)
    print(f"{path.name}: n={n}  truthful_rate={truthful:.4f}  flipped={flipped}")
    return truthful


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", help="results/truthfulqa__<method>.jsonl path(s)")
    args = ap.parse_args()
    for p in args.jsonl:
        rescore(p)


if __name__ == "__main__":
    main()
