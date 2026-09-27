"""Consolidate all per-dataset jsonls into results/summary_all.csv."""
import csv
import json
import pathlib

R = pathlib.Path('results')
DATASETS = ['truthfulqa', 'ciar', 'bold', 'honest', 'creak', 'mmlu', 'bbq', 'crows_pairs']
METHODS = ['reflectevo', 'dmc']

METRIC_LABEL = dict(
    open_qa='truthful_rate',
    exact_match='accuracy', binary='accuracy', mcq4='accuracy',
    mcq3='accuracy', pair='accuracy',
    generation_tox='toxic_rate(lower=better)',
    generation_honest='honest_score(lower=better)',
)

rows = []
for ds in DATASETS:
    for m in METHODS:
        p = R / f'{ds}__{m}.jsonl'
        if not p.exists():
            continue
        items = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        n = len(items)
        task = items[0]['task']
        scores = [x['score'] for x in items if x.get('score') is not None]
        metric = METRIC_LABEL[task]
        value = sum(scores) / len(scores) if scores else float('nan')
        row = dict(dataset=ds, method=m, n=n, metric=metric, value=round(value, 4))
        if ds == 'truthfulqa':
            row['strict'] = round(sum(x.get('score_strict', 0) for x in items) / n, 4)
            row['inclusion'] = round(sum(x.get('score_inclusion', 0) for x in items) / n, 4)
            row['max_merged'] = row['value']
        if m == 'dmc':
            confs = [x['confidence'] for x in items if x.get('confidence') is not None and x.get('score') is not None]
            corr = [int(x['score']) for x in items if x.get('confidence') is not None and x.get('score') is not None]
            if len(set(corr)) == 2:
                from sklearn.metrics import roc_auc_score
                row['metacognition_auroc'] = round(roc_auc_score(corr, confs), 4)
        rows.append(row)

out = R / 'summary_all.csv'
keys = ['dataset', 'method', 'n', 'metric', 'value', 'strict', 'inclusion', 'max_merged', 'metacognition_auroc']
with out.open('w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f'wrote {out}')
print()
hdr = f'{"dataset":12s} {"method":10s} {"n":>4s}  {"metric":24s} {"value":>7s}  {"strict":>7s}  {"incl":>7s}  {"max":>7s}  {"auroc":>7s}'
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(f'{r["dataset"]:12s} {r["method"]:10s} {r["n"]:>4d}  {r["metric"]:24s} {r["value"]:>7.4f}  '
          f'{str(r.get("strict","-")):>7s}  {str(r.get("inclusion","-")):>7s}  '
          f'{str(r.get("max_merged","-")):>7s}  {str(r.get("metacognition_auroc","-")):>7s}')
