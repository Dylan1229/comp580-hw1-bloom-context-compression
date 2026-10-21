"""Run every experiment for COMP 480/580 HW1 and write CSVs + plots to --out.

Stages
  1. fpr     : standalone BloomFilter, measured vs theoretical FPR over an (R, k) grid.
  2. sweep   : chunk-index Bloom filters on the 4 v1 contexts over an (R, k) grid --
               FPR, memory, retained context, estimated input tokens, gold-evidence recall.
               No LLM calls.
  3. llm     : full-context baseline, exact-set oracle, and Bloom configurations, all with
               gemini-3.5-flash-lite and the same prompt; accuracy, real input tokens
               (usage_metadata.prompt_token_count), latency, memory.
  4. plots

Usage:  GEMINI_API_KEY=... python run_experiments.py --data-dir data_repo --out results
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import re
import threading
import time
import tracemalloc
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from bloom import (BloomContextCompressor, BloomFilter, features, theoretical_fpr)

CONTEXTS = ["antikythera_dossier_v1", "dwh_dossier_v1", "harbour_puzzle_v1",
            "lunarleaper_2609_11453"]
MODEL = "gemini-3.5-flash-lite"
SEED = 0

PROMPT = """You are given a document (it may be an excerpt, with omitted parts marked [...]) and a question about it.
Answer using only the information in the document.
Reply with only the final answer -- a number (with its unit, if any), a name, a date or a short phrase. Do not explain.

Document:
<<<
{context}
>>>

Question: {question}
Answer:"""

SWEEP_R = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
SWEEP_K = [1, 2, 3, 4, 6, 8]
DEFAULT_K, DEFAULT_R = 4, 1024


def llm_configs(grid):
    cfgs = [{"name": "baseline", "method": "full"},
            {"name": "exact", "method": "exact"}]
    Rs = [128, 256, 512, 1024, 2048, 4096] if grid == "full" else [256, 1024, 4096]
    cfgs += [{"name": f"bloom_R{R}_k{DEFAULT_K}", "method": "bloom", "R": R, "k": DEFAULT_K}
             for R in Rs]
    if grid == "full":
        cfgs += [{"name": f"bloom_R512_k{k}", "method": "bloom", "R": 512, "k": k}
                 for k in [1, 2, 3, 6, 8]]
    return cfgs


# --------------------------------------------------------------------------- data

def resolve_contexts(data_dir, names):
    """``names`` is a list of context names, or ["all"] for every contexts/<name>.txt that
    has both questions/<name>_questions.json and answers/<name>_answers.json."""
    if names != ["all"]:
        return names
    root = Path(data_dir)
    return sorted(p.stem for p in (root / "contexts").glob("*.txt")
                  if (root / "questions" / f"{p.stem}_questions.json").exists()
                  and (root / "answers" / f"{p.stem}_answers.json").exists())


def load_data(data_dir, names):
    data = {}
    for name in names:
        ctx = (Path(data_dir) / "contexts" / f"{name}.txt").read_text(encoding="utf-8")
        qs = json.loads((Path(data_dir) / "questions" / f"{name}_questions.json").read_text())
        ans = json.loads((Path(data_dir) / "answers" / f"{name}_answers.json").read_text())
        gold = {a["qid"]: a for a in ans["answers"]}
        data[name] = {"context": ctx, "questions": qs["questions"], "gold": gold}
    return data


# --------------------------------------------------------------------------- grading

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*(thousand|million|billion|trillion)\b)?",
                  re.I)


def numbers(s):
    out = []
    for m in _NUM.finditer(s.replace("−", "-")):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if m.group(2):
            v *= _SCALE[m.group(2).lower()]
        out.append(v)
    return out


def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower().replace("±", "+/-")
    s = s.replace("∘", "°")
    s = re.sub(r"[^\w\s+/.°%$-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().strip(".").strip()
    return re.sub(r"^(the|a|an) ", "", s)


def grade(pred, gold):
    """Numeric answers: first number in prediction within relative tolerance of a gold
    number. Text answers: case-insensitive normalised match (or the gold phrase appearing
    whole inside a short prediction). Aliases are always accepted."""
    golds = [gold["expected"]] + list(gold.get("aliases", []))
    p = norm_text(pred)
    gn = [norm_text(g) for g in golds]
    if p in gn:
        return True
    if gold["grading"]["normaliser"] == "numeric":
        pn = numbers(pred)
        tol = gold["grading"].get("numeric_tolerance_rel", 0.005)
        gvals = [v for g in golds for v in numbers(g)[:1]]
        return bool(pn) and any(abs(pn[0] - g) <= tol * max(abs(g), 1e-12) for g in gvals)
    limit = max(len(gold["expected"].split()) + 6, 10)
    return len(p.split()) <= limit and any(
        re.search(rf"(?<![\w]){re.escape(g)}(?![\w])", p) for g in gn if g)


# --------------------------------------------------------------------------- LLM

class QuotaExhausted(RuntimeError):
    """Daily request quota used up; finished configurations are kept (and cached)."""


class LLM:
    def __init__(self, model, cache_path, thinking_level=None, use_cache=True):
        from google.genai import types
        self.client = None  # created on the first cache miss, so cached runs need no API key
        self.types = types
        self.model = model
        kw = {"temperature": 0.0, "seed": SEED}
        if thinking_level:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        self.config = types.GenerateContentConfig(**kw)
        self.cfg_key = json.dumps({"model": model, "thinking": thinking_level, **{
            k: v for k, v in kw.items() if k != "thinking_config"}}, sort_keys=True)
        self.cache_path = Path(cache_path)
        self.cache, self.lock = {}, threading.Lock()
        if use_cache and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec
        self.use_cache = use_cache

    def __call__(self, prompt):
        key = hashlib.sha256((self.cfg_key + prompt).encode()).hexdigest()
        if self.use_cache and key in self.cache:
            return self.cache[key]
        from google import genai
        from google.genai import errors
        with self.lock:
            if self.client is None:
                self.client = genai.Client()
        attempts = 30
        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                r = self.client.models.generate_content(model=self.model, contents=prompt,
                                                        config=self.config)
            except (errors.APIError, OSError) as e:
                code = getattr(e, "code", None)
                if code == 429 and "PerDay" in str(e):
                    raise QuotaExhausted(str(e)[:300]) from e
                if code not in (None, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                    raise
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            dt = time.perf_counter() - t0
            u = r.usage_metadata
            rec = {"key": key, "text": (r.text or "").strip(), "latency": dt,
                   "prompt_tokens": u.prompt_token_count if u else None,
                   "output_tokens": (u.candidates_token_count or 0) if u else None,
                   "thought_tokens": (u.thoughts_token_count or 0) if u else None}
            with self.lock:
                self.cache[key] = rec
                with self.cache_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            return rec


# --------------------------------------------------------------------------- stage 1

def run_fpr_standalone(out):
    rng = random.Random(SEED)
    n = 1000
    items = [f"item-{rng.getrandbits(64):016x}" for _ in range(n)]
    probes = [f"probe-{rng.getrandbits(64):016x}" for _ in range(20000)]
    rows = []
    for R in [1000, 2000, 4000, 8000, 16000, 32000]:
        for k in range(1, 11):
            bf = BloomFilter(R, k, seed=SEED)
            for it in items:
                bf.add(it)
            assert all(it in bf for it in items)  # no false negatives
            fp = sum(p in bf for p in probes) / len(probes)
            rows.append({"R_bits": R, "k": k, "n": n, "bits_per_item": R / n,
                         "memory_bytes": bf.nbytes, "measured_fpr": fp,
                         "theoretical_fpr": theoretical_fpr(n, R, k),
                         "fill_ratio": bf.fill_ratio()})
    write_csv(out / "fpr_standalone.csv", rows)
    return rows


# --------------------------------------------------------------------------- stage 2

def gold_recall(compressed, gold):
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    c = norm(compressed.replace("[...]", " "))
    spans = gold.get("gold_spans", [])
    return sum(norm(s["quote"]) in c for s in spans) / len(spans) if spans else 1.0


def index_fpr(comp, n_probes=2000):
    """Measured FPR of the per-chunk filters using random non-member probes, plus the
    theoretical value averaged over chunks (each chunk has its own n)."""
    rng = random.Random(SEED)
    probes = [f"\x00probe{rng.getrandbits(64):x}" for _ in range(n_probes)]  # never a feature
    pos = sum(int(comp.index.query(p).sum()) for p in probes)
    measured = pos / (n_probes * comp.n_chunks)
    theory = float(np.mean([theoretical_fpr(n, comp.R, comp.k) for n in comp.n_items]))
    return measured, theory


def query_term_fpr(comp, data_q):
    """FPR on the real question terms: positives in chunks that do not contain the term."""
    fp = neg = 0
    for q in data_q:
        for t in features(q["question"]):
            hits = comp.index.query(t)
            truth = np.array([t in f for f in comp.chunk_feats])
            fp += int((hits & ~truth).sum())
            neg += int((~truth).sum())
    return fp / max(neg, 1)


def run_sweep(data, out, chars_per_token):
    rows = []
    for name, d in data.items():
        ctx = d["context"]
        overhead = len(PROMPT) - len("{context}{question}")
        for R in SWEEP_R:
            for k in SWEEP_K:
                comp = BloomContextCompressor(ctx, R=R, k=k, seed=SEED)
                m_fpr, t_fpr = index_fpr(comp)
                q_fpr = query_term_fpr(comp, d["questions"])
                kept, toks, rec, rt = [], [], [], []
                for q in d["questions"]:
                    text, info = comp.compress(q["question"])
                    kept.append(len(text) / len(ctx))
                    toks.append((len(text) + len(q["question"]) + overhead) / chars_per_token)
                    rec.append(gold_recall(text, d["gold"][q["qid"]]))
                    rt.append(info["retrieval_time"])
                rows.append({"context": name, "R_bits": R, "k": k, "n_chunks": comp.n_chunks,
                             "mean_items_per_chunk": float(comp.n_items.mean()),
                             "memory_bytes": comp.index_bytes,
                             "measured_fpr": m_fpr, "theoretical_fpr": t_fpr,
                             "query_term_fpr": q_fpr,
                             "retained_frac": float(np.mean(kept)),
                             "est_input_tokens": float(np.mean(toks)),
                             "gold_recall": float(np.mean(rec)),
                             "build_time_s": comp.build_time,
                             "retrieval_time_s": float(np.mean(rt))})
            print(f"  sweep {name} R={R} done", flush=True)
    write_csv(out / "sweep_offline.csv", rows)
    return rows


# --------------------------------------------------------------------------- stage 3

def peak_memory(ctx, questions, R, k, exact):
    """Peak Python heap (tracemalloc) for building the index and compressing all questions."""
    tracemalloc.start()
    comp = BloomContextCompressor(ctx, R=R, k=k, seed=SEED, exact=exact)
    for q in questions:
        comp.compress(q["question"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def run_llm(data, out, grid, workers, thinking_level, use_cache):
    llm = LLM(MODEL, out / "llm_cache.jsonl", thinking_level, use_cache)
    pred_rows, summary = [], []
    for cfg in llm_configs(grid):
        jobs = []
        mem_index = mem_peak = 0
        for name, d in data.items():
            qs = d["questions"]
            if cfg["method"] == "full":
                for q in qs:
                    jobs.append((name, q, d["context"], 0.0, 1.0, None))
                continue
            exact = cfg["method"] == "exact"
            R, k = cfg.get("R", 0), cfg.get("k", 0)
            comp = BloomContextCompressor(d["context"], R=R, k=k, seed=SEED, exact=exact)
            mem_index += comp.index_bytes
            mem_peak = max(mem_peak, peak_memory(d["context"], qs, R, k, exact))
            build = comp.build_time / len(qs)  # index built once per context, amortised
            for q in qs:
                text, info = comp.compress(q["question"])
                jobs.append((name, q, text, build + info["retrieval_time"],
                             len(text) / len(d["context"]), gold_recall(text, d["gold"][q["qid"]])))

        def work(job):
            name, q, text, rtime, frac, rec = job
            res = llm(PROMPT.format(context=text, question=q["question"]))
            return job, res

        try:
            with cf.ThreadPoolExecutor(workers) as ex:
                results = list(ex.map(work, jobs))
        except QuotaExhausted as e:
            print(f"  daily Gemini quota exhausted during {cfg['name']}; stopping LLM stage "
                  f"({len(summary)} configs finished). Re-run the same command later to resume "
                  f"from the cache.\n  {e}", flush=True)
            break
        rows = []
        for (name, q, text, rtime, frac, rec), res in results:
            gold = data[name]["gold"][q["qid"]]
            rows.append({"config": cfg["name"], "method": cfg["method"],
                         "R_bits": cfg.get("R", ""), "k": cfg.get("k", ""),
                         "context": name, "qid": q["qid"], "tier": gold.get("tier", ""),
                         "question": q["question"], "expected": gold["expected"],
                         "prediction": res["text"], "correct": int(grade(res["text"], gold)),
                         "input_tokens": res["prompt_tokens"],
                         "retained_frac": frac, "gold_recall": rec,
                         "retrieval_latency_s": rtime, "llm_latency_s": res["latency"],
                         "total_latency_s": rtime + res["latency"]})
        pred_rows += rows
        s = {"config": cfg["name"], "method": cfg["method"], "R_bits": cfg.get("R", ""),
             "k": cfg.get("k", ""),
             "accuracy": np.mean([r["correct"] for r in rows]),
             "total_input_tokens": int(sum(r["input_tokens"] or 0 for r in rows)),
             "mean_input_tokens": float(np.mean([r["input_tokens"] or 0 for r in rows])),
             "mean_retained_frac": float(np.mean([r["retained_frac"] for r in rows])),
             "total_latency_s": float(sum(r["total_latency_s"] for r in rows)),
             "mean_latency_s": float(np.mean([r["total_latency_s"] for r in rows])),
             "median_latency_s": float(np.median([r["total_latency_s"] for r in rows])),
             "mean_retrieval_latency_ms": 1e3 * float(np.mean([r["retrieval_latency_s"] for r in rows])),
             "index_memory_bytes": mem_index, "peak_heap_bytes": mem_peak,
             "n_questions": len(rows)}
        for tier in ["T1_lookup", "T2_multihop", "T3_aggregate"]:
            hits = [r["correct"] for r in rows if r["tier"] == tier]
            s[f"acc_{tier}"] = float(np.mean(hits)) if hits else ""
        summary.append(s)
        print(f"  {cfg['name']:<18} acc={s['accuracy']:.3f} tokens={s['total_input_tokens']:,} "
              f"latency={s['total_latency_s']:.1f}s index={mem_index / 1024:.1f}KB", flush=True)
        write_csv(out / "llm_predictions.csv", pred_rows)
        write_csv(out / "llm_summary.csv", summary)
    return summary


# --------------------------------------------------------------------------- plots

def make_plots(out, fpr_rows, sweep_rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)

    def agg(rows, key, fn=np.sum):
        g = defaultdict(list)
        for r in rows:
            g[(r["R_bits"], r["k"])].append(r[key])
        return {kk: fn(v) for kk, v in g.items()}

    mem = agg(sweep_rows, "memory_bytes", np.sum)
    fpr_m = agg(sweep_rows, "measured_fpr", np.mean)
    fpr_t = agg(sweep_rows, "theoretical_fpr", np.mean)
    tok = agg(sweep_rows, "est_input_tokens", np.mean)
    kept = agg(sweep_rows, "retained_frac", np.mean)
    recall = agg(sweep_rows, "gold_recall", np.mean)
    cmap = plt.get_cmap("viridis")

    # 1. memory vs FPR (index, all 4 contexts)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, k in enumerate(SWEEP_K):
        xs = [mem[(R, k)] / 1024 for R in SWEEP_R]
        c = cmap(i / (len(SWEEP_K) - 1))
        ax.plot(xs, [max(fpr_m[(R, k)], 1e-7) for R in SWEEP_R], "o-", color=c, label=f"k={k}")
        ax.plot(xs, [max(fpr_t[(R, k)], 1e-7) for R in SWEEP_R], "--", color=c, alpha=0.6)
    ax.set(xscale="log", yscale="log", xlabel="Bloom filter memory, all 4 contexts (KB)",
           ylabel="false positive rate", title="Memory vs FPR (solid: measured, dashed: theory)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "memory_vs_fpr.png", dpi=150)
    plt.close(fig)

    # 2. memory vs LLM input tokens
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, k in enumerate(SWEEP_K):
        ax.plot([mem[(R, k)] / 1024 for R in SWEEP_R], [tok[(R, k)] for R in SWEEP_R], "o-",
                color=cmap(i / (len(SWEEP_K) - 1)), alpha=0.8, label=f"k={k} (estimated)")
    if summary:
        bl = next(s for s in summary if s["method"] == "full")
        bloom = [s for s in summary if s["method"] == "bloom" and s["k"] == DEFAULT_K]
        ax.plot([s["index_memory_bytes"] / 1024 for s in bloom],
                [s["mean_input_tokens"] for s in bloom], "s", color="crimson", ms=8,
                label=f"k={DEFAULT_K} (measured, Gemini)")
        ax.axhline(bl["mean_input_tokens"], color="k", ls=":", label="full-context baseline")
    ax.set(xscale="log", yscale="log", xlabel="Bloom filter memory, all 4 contexts (KB)",
           ylabel="mean LLM input tokens / question", title="Memory vs LLM input tokens")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "memory_vs_tokens.png", dpi=150)
    plt.close(fig)

    # 3. tokens vs accuracy
    if summary:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, s in enumerate(summary):
            color = {"full": "k", "exact": "tab:green"}.get(s["method"], "tab:blue")
            if s["method"] == "bloom" and s["k"] != DEFAULT_K:
                color = "tab:orange"
            n = s.get("n_questions") or 80  # older summaries lack the column
            se = np.sqrt(s["accuracy"] * (1 - s["accuracy"]) / n)  # binomial std. error
            ax.errorbar(s["mean_input_tokens"], s["accuracy"], yerr=se, fmt="o", color=color,
                        ms=6, elinewidth=0.8, alpha=0.85, capsize=2)
            ax.annotate(s["config"].replace("bloom_", ""), (s["mean_input_tokens"], s["accuracy"]),
                        fontsize=7, xytext=(4, 3 if i % 2 else -10), textcoords="offset points")
        ax.set(xscale="log", xlabel="mean LLM input tokens / question", ylabel="accuracy (80 Q)",
               title="Input tokens vs QA accuracy (bars: ±1 binomial SE)")
        ax.scatter([], [], color="k", label="full context")
        ax.scatter([], [], color="tab:green", label="exact sets (no FP)")
        ax.scatter([], [], color="tab:blue", label=f"Bloom, vary R (k={DEFAULT_K})")
        ax.scatter([], [], color="tab:orange", label="Bloom, vary k (R=512)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pdir / "tokens_vs_accuracy.png", dpi=150)
        plt.close(fig)

    # 4. standalone filter: measured vs theory
    fig, ax = plt.subplots(figsize=(5, 4.5))
    m = [r["measured_fpr"] for r in fpr_rows]
    t = [r["theoretical_fpr"] for r in fpr_rows]
    ax.scatter(t, m, c=[r["k"] for r in fpr_rows], cmap="viridis", s=18)
    lo = max(min(min(m), min(t)), 1e-6)
    ax.plot([lo, 1], [lo, 1], "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="theoretical FPR", ylabel="measured FPR",
           title="Standalone Bloom filter (n=1000)")
    fig.tight_layout()
    fig.savefig(pdir / "fpr_measured_vs_theory.png", dpi=150)
    plt.close(fig)

    # 5. R vs retained context & gold-evidence recall
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    for i, k in enumerate(SWEEP_K):
        c = cmap(i / (len(SWEEP_K) - 1))
        ax1.plot(SWEEP_R, [kept[(R, k)] for R in SWEEP_R], "o-", color=c, label=f"k={k}")
        ax1.plot(SWEEP_R, [recall[(R, k)] for R in SWEEP_R], "x:", color=c, alpha=0.7)
    ax1.set(xscale="log", xlabel="R (bits per chunk filter)",
            ylabel="fraction", title="Retained context (solid) and gold-evidence recall (dotted)")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "R_vs_retained_recall.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- utils

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    def conv(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with open(path) as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data_repo")
    ap.add_argument("--contexts", nargs="+", default=CONTEXTS,
                    help="context names (default: the 4 v1 contexts), or 'all' for every context "
                         "in DATA_DIR with matching questions/answers files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-llm", action="store_true", help="only run offline stages 1, 2, 4")
    ap.add_argument("--llm-grid", choices=["full", "quick"], default="full")
    ap.add_argument("--workers", type=int, default=4, help="concurrent Gemini requests")
    ap.add_argument("--thinking-level", default=None,
                    help="optional Gemini thinking level (minimal/low/medium/high)")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached Gemini responses")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = resolve_contexts(args.data_dir, args.contexts)
    print(f"contexts: {', '.join(names)}")
    data = load_data(args.data_dir, names)

    print("[1/4] standalone Bloom filter FPR")
    fpr_rows = run_fpr_standalone(out)

    summary = []
    if not args.skip_llm:
        print("[3/4] LLM runs (baseline, exact oracle, Bloom configs)")
        summary = run_llm(data, out, args.llm_grid, args.workers, args.thinking_level,
                          not args.no_cache)
    elif (out / "llm_summary.csv").exists():
        summary = read_csv(out / "llm_summary.csv")

    # Calibrate chars->tokens from the measured baseline when available.
    cpt = 4.0
    if summary:
        bl = next(s for s in summary if s["method"] == "full")
        chars = sum(len(PROMPT.format(context=d["context"], question=q["question"]))
                    for d in data.values() for q in d["questions"])
        cpt = chars / bl["total_input_tokens"]
    print(f"[2/4] offline (R, k) sweep  (chars/token = {cpt:.2f})")
    sweep_rows = run_sweep(data, out, cpt)

    print("[4/4] plots")
    make_plots(out, fpr_rows, sweep_rows, summary)
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
