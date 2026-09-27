# Data

The benchmarks are not redistributed in this repository. Download them from the original
sources (each under its own license) and place them here with the file names below.

| File in `data/` | Benchmark | Source | License | Columns / fields read |
|---|---|---|---|---|
| `TruthfulQA.csv` | TruthfulQA | [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) (`TruthfulQA.csv`) | Apache-2.0 | `question`, `best_answer` (plus `category` / `type`) |
| `CIAR.json` | CIAR | [Skytliang/Multi-Agents-Debate](https://github.com/Skytliang/Multi-Agents-Debate) | see source repo | list of `{question, answer, explanation}`; `answer` is a list of accepted forms |
| `bold_female.csv`, `bold_male.csv` | BOLD (gender domain; 776 actresses + 776 actors down-sampled from 1587) | [amazon-science/bold](https://github.com/amazon-science/bold) | CC BY-SA 4.0 | one row per prompt: `domain`, `name`, `category`, `prompts` (actress / actor prompts respectively) |
| `honest.csv` | HONEST (`en_queer_nonqueer`, 705 items) | [MilaNLProc/honest](https://huggingface.co/datasets/MilaNLProc/honest) | MIT | `template_masked`, `identity`, `category`, `type` |
| `mmlu_test.csv` | MMLU (test) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | MIT | `Question`, `A`, `B`, `C`, `D`, `Answer` (letter), `Subject` |
| `bbq/*.jsonl` | BBQ | [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ) (`data/<Category>.jsonl`) | CC BY 4.0 | official JSONL (`context`, `question`, `ans0..2`, `label`, `context_condition`, `category`) |
| `crows_pairs_anonymized.csv` | CrowS-Pairs | [nyu-mll/crows-pairs](https://github.com/nyu-mll/crows-pairs) | CC BY-SA 4.0 | `sent_more`, `sent_less`, `stereo_antistereo`, `bias_type` |
| *(loaded from the Hub)* | CREAK | [amydeng2000/CREAK](https://huggingface.co/datasets/amydeng2000/CREAK) | see dataset card | — |

Notes

- `metacrit/run_metacrit.py` and `metacrit/run_baselines.py` load TruthfulQA, MMLU and CREAK from
  the Hugging Face Hub. BOLD comes from the two local CSVs when present (otherwise from the Hub,
  filtered to the gender domain and down-sampled the same way). `honest.csv`, `bbq/` and
  `crows_pairs_anonymized.csv` are **required** locally with `datasets>=4`, which dropped the
  script-based Hub loaders for these datasets. CIAR is always read from `data/CIAR.json`.
- The `ablation_*.py` scripts read the local files above. `ablation_creak.py` loads CREAK from the Hub.
- `mep_baseline.py` can also run MAFALDA (`gold_standard_dataset.jsonl`, not used in the paper).
