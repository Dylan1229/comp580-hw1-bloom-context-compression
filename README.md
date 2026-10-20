# COMP 480/580 HW1 — Bloom Filters for LLM Context Compression

Fanjiang Ye (fy27), Kaijian Wang (kw118). Results, plots and answers to the questions: [REPORT.md](REPORT.md).

Files
- `bloom.py` — Bloom filter, bit-sliced per-chunk Bloom index, and the context compressor (no Bloom filter libraries; hashing = keyed BLAKE2b + enhanced double hashing).
- `run_experiments.py` — runs every experiment (standalone FPR study, offline (R, k) sweep, Gemini baseline + Bloom runs) and writes CSVs and plots to `results/`.
- `REPORT.md` — implementation, results, plots, answers to the questions.

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/sahiljoshi515/Assignment1_COMP_580 data_repo
git -C data_repo checkout 62e826a7f4649dbb6420c9965cc94028d92046b8
export GEMINI_API_KEY=<your key>
```

## Reproduce everything

```bash
python run_experiments.py --data-dir data_repo --out results
```

This runs 13 configurations x 80 questions (= 1040 Gemini calls, i.e. more than one free-tier day, `gemini-3.5-flash-lite`, temperature 0, seed 0) plus the offline sweeps, then writes:

| file | content |
|---|---|
| `results/fpr_standalone.csv` | measured vs theoretical FPR of a single filter, n=1000, R in 1k..32k, k in 1..10 |
| `results/sweep_offline.csv` | per context, R in {64..8192}, k in {1,2,3,4,6,8}: memory, measured/theoretical/query-term FPR, retained context, estimated input tokens, gold-evidence recall |
| `results/llm_predictions.csv` | every question x configuration: prediction, correctness, input tokens (Gemini `usage_metadata`), latency |
| `results/llm_summary.csv` | accuracy, tokens, latency, index memory, peak heap per configuration |
| `results/plots/*.png` | memory vs FPR, memory vs tokens, tokens vs accuracy, measured vs theoretical FPR, R vs retained context |

Held-out contexts: put `<name>.txt`, `<name>_questions.json` and `<name>_answers.json` into the usual `contexts/`, `questions/`, `answers/` folders and pass the names (or `all` for every complete triple in the data dir):

```bash
python run_experiments.py --data-dir <dir> --contexts <name1> <name2> --out results_heldout
```

Options: `--skip-llm` (offline stages only, no API key needed, ~30 s), `--llm-grid quick` (5 configs), `--workers N` (concurrent requests, default 4), `--no-cache`.
The Gemini free tier allows 500 requests/day for this model. If the daily quota runs out, the script stops the LLM stage cleanly (finished configurations are saved, plots are still made); re-running the same command later resumes from the cache. Gemini responses are cached in `results/llm_cache.jsonl`, which is committed. With it, running the command above re-creates every reported number without any API calls. To query Gemini afresh, delete the file or pass `--no-cache`. Gemini is not fully deterministic even at temperature 0, so a fresh run can differ by a question or two. Everything except LLM latency is deterministic (fixed seeds, keyed hashing).
