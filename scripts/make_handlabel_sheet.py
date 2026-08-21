#!/usr/bin/env python
"""Build a CSV for hand-labelling, to measure how accurate the regex is.

The paper validated its LLM autorater against hand-labelled rollouts before
trusting it. We are using regex instead of an autorater, so the same
validation applies: read some generations yourself, write down the correct
answer, and compare.

Sampling is deliberately NOT uniform. Uniform sampling would fill the sheet
with clean cases, which tell you nothing. We oversample the generations the
parser found problematic, because that is where it is most likely wrong.

    python scripts/make_handlabel_sheet.py --n 100

Then open results/handlabel.csv, fill the `human_label` column with one of
YES / NO / UNKNOWN / NO_ANSWER, and run:

    python scripts/make_handlabel_sheet.py --score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

LABELS = {"YES", "NO", "UNKNOWN", "NO_ANSWER"}


def load_raw(results_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(results_dir.glob("raw_*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no raw_*.jsonl in {results_dir}. Run the experiment first.")
    return rows


def build(results_dir: Path, n: int, seed: int) -> None:
    rows = load_raw(results_dir)
    rng = random.Random(seed)

    problem = [r for r in rows if not r["format_valid"]]
    unknown = [r for r in rows if r.get("answer") == "UNKNOWN"]
    clean = [r for r in rows if r["format_valid"] and r.get("answer") != "UNKNOWN"]

    # Half the sheet from format failures, a quarter from UNKNOWNs (the
    # category regex is least able to verify), a quarter from clean cases
    # as a control.
    take = []
    for pool, k in ((problem, n // 2), (unknown, n // 4), (clean, n - n // 2 - n // 4)):
        rng.shuffle(pool)
        take += pool[:k]
    # Backfill from clean if any pool was short.
    if len(take) < n:
        extra = [r for r in clean if r not in take]
        rng.shuffle(extra)
        take += extra[:n - len(take)]
    rng.shuffle(take)

    out = results_dir / "handlabel.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "model", "gold", "machine_answer", "machine_reason",
                    "human_label", "question", "generation"])
        for i, r in enumerate(take):
            w.writerow([i, r["model"], r["gold"], r.get("answer") or "NO_ANSWER",
                        r["reason"], "", r["question"],
                        r["generation"].replace("\r", " ")])
    print(f"wrote {len(take)} rows to {out}")
    print("Fill the human_label column with YES / NO / UNKNOWN / NO_ANSWER, "
          "then rerun with --score")


def score(results_dir: Path) -> None:
    path = results_dir / "handlabel.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labelled = [r for r in rows if r["human_label"].strip().upper() in LABELS]
    if not labelled:
        raise SystemExit("no rows labelled yet")

    agree = 0
    disagreements = []
    for r in labelled:
        human = r["human_label"].strip().upper()
        machine = r["machine_answer"].strip().upper()
        if human == machine:
            agree += 1
        else:
            disagreements.append(r)

    pct = 100 * agree / len(labelled)
    print(f"labelled: {len(labelled)} of {len(rows)}")
    print(f"regex agreement with human: {agree}/{len(labelled)} = {pct:.1f}%")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements:")
        for r in disagreements[:20]:
            print(f"  [{r['idx']}] human={r['human_label']:10} "
                  f"machine={r['machine_answer']:10} reason={r['machine_reason']}")
        print("\nEach of these is either a parser bug or a prompt weakness. "
              "Both are findings worth reporting.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    d = Path(a.results_dir)
    score(d) if a.score else build(d, a.n, a.seed)


if __name__ == "__main__":
    main()
