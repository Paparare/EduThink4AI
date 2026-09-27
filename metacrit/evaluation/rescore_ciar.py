"""Re-score CIAR jsonls using a correct exact-match scorer.

CIAR gold answers are stored as the str() of a Python list, e.g.
"['1.5', '3/2']" -- a set of accepted normalized forms. The original
score_structured("exact_match", ...) stripped brackets/punct and required the
whole flattened blob ("1.532") to appear in the prediction, so a correct
prediction like "1.5 m/s" was marked FALSE. This script reparses the gold
into the list of acceptable answers and marks TRUE if the cleaned prediction
contains ANY accepted form (or equals it exactly).
"""
import argparse
import ast
import json
import re
import shutil
import sys
import time
from pathlib import Path


def _norm(s):
    return re.sub(r"[^0-9a-z.\-/%]", "", str(s).lower())


def parse_gold(gold_field):
    """gold_field is usually str(list) e.g. "['0.5']" -- return list of strings."""
    if gold_field is None:
        return []
    if isinstance(gold_field, list):
        return [str(x) for x in gold_field]
    s = str(gold_field).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return [str(v)]
    except (ValueError, SyntaxError):
        return [s]


def score_one(prediction, gold_field):
    p = _norm(prediction)
    accepted = parse_gold(gold_field)
    for a in accepted:
        g = _norm(a)
        if not g:
            continue
        if p == g or g in p or p in g:
            return 1.0
    return 0.0


def rescore(path):
    path = Path(path)
    if not path.exists():
        print(f"skip: {path} not found")
        return
    backup = path.with_suffix(path.suffix + f".bak.scorer.{int(time.time())}")
    shutil.copy2(path, backup)

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    flipped = 0
    for r in rows:
        old = r.get("score")
        new = score_one(r["prediction"], r["gold"])
        r["score_old"] = old
        r["score"] = new
        if old != new:
            flipped += 1

    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    acc = sum(r["score"] for r in rows) / max(1, n)
    old_acc = sum((r["score_old"] or 0.0) for r in rows) / max(1, n)
    print(f"{path.name}: n={n}  old={old_acc:.4f}  new={acc:.4f}  flipped={flipped}  backup -> {backup.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+")
    args = ap.parse_args()
    for p in args.jsonl:
        rescore(p)


if __name__ == "__main__":
    main()
