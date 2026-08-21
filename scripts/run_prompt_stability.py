#!/usr/bin/env python
"""Prompt-stability experiment, one model per invocation.

Test the whole pipeline on CPU first, no GPU and no quota:

    python scripts/run_prompt_stability.py --model llama3b --backend mock

Then the real thing on Kaggle:

    python scripts/run_prompt_stability.py --model llama3b
    python scripts/run_prompt_stability.py --model gemma9b
    python scripts/run_prompt_stability.py --model qwen4b
    python scripts/run_prompt_stability.py --model deepseek8b

One model per process, because vLLM does not reliably release GPU memory
until the process exits.

Writes per model:
    results/raw_<model>.jsonl       one row per generation, full text kept
    results/summary_<model>.csv     the metrics
    results/failures_<model>.csv    why format validation failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cot_faith.data import load_questions, stratified_sample, validate
from cot_faith.generate import ModelSpec, generate_mock, generate_vllm
from cot_faith.metrics import failure_breakdown, summarize
from cot_faith.parsing import parse_answer
from cot_faith.prompts import build_prompt, load_body, to_chat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="key in configs/models.yaml")
    p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    p.add_argument("--n-questions", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="data/chainscope")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tp", type=int, default=None,
                   help="override tensor_parallel_size")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    questions = load_questions(a.data_root)
    v = validate(questions)
    if not v["ok"]:
        sys.exit(f"dataset validation FAILED: {v}")
    print(f"dataset ok: {v['questions']} questions, {v['pairs']} pairs, "
          f"{v['templates']} templates", flush=True)

    sample = stratified_sample(questions, a.n_questions, seed=a.seed)
    print(f"sample: {len(sample)} questions across "
          f"{len({q.template for q in sample})} templates", flush=True)

    body = load_body(a.data_root)
    chats = [to_chat(build_prompt(body, q.q_str)) for q in sample]

    cfg = yaml.safe_load(open(a.models_config))[a.model]
    spec = ModelSpec.from_config(a.model, cfg, tp_override=a.tp)
    print(f"model: {spec.hf_id}  backend={a.backend}  "
          f"max_new_tokens={spec.max_new_tokens}  thinking={spec.thinking}",
          flush=True)

    if a.backend == "mock":
        gens = generate_mock(chats, spec, a.seed, golds=[q.gold for q in sample])
    else:
        gens = generate_vllm(chats, spec, a.seed)

    records = []
    raw_path = out / f"raw_{a.model}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for q, (text, truncated, n_tok) in zip(sample, gens):
            r = parse_answer(text, truncated=truncated)
            row = {
                "model": a.model,
                "hf_id": spec.hf_id,
                "backend": a.backend,
                "seed": a.seed,
                "qid": q.qid,
                "pair_id": q.pair_id,
                "template": q.template,
                "prop_id": q.prop_id,
                "comparison": q.comparison,
                "gold": q.gold,
                "gen_tokens": n_tok,
                **r.as_dict(),
            }
            records.append(row)
            fh.write(json.dumps({**row, "question": q.q_str,
                                 "generation": text}, ensure_ascii=False) + "\n")

    summary = summarize(records)
    failures = failure_breakdown(records)
    summary.to_csv(out / f"summary_{a.model}.csv", index=False)
    failures.to_csv(out / f"failures_{a.model}.csv", index=False)

    print("\n=== metrics ===")
    print(summary.to_string(index=False))
    print("\n=== why format validation failed ===")
    print(failures.to_string(index=False) if len(failures) else "  none")
    print(f"\nraw generations: {raw_path}")


if __name__ == "__main__":
    main()
