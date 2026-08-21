#!/usr/bin/env python
"""Combine per-model results into the two tables that go in the writeup.

    python scripts/combine_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cot_faith.metrics import failure_breakdown, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    d = Path(a.results_dir)

    records = []
    for p in sorted(d.glob("raw_*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                r.pop("generation", None)
                r.pop("question", None)
                records.append(r)
    if not records:
        sys.exit(f"no raw_*.jsonl in {d}")

    summary = summarize(records)
    failures = failure_breakdown(records)
    summary.to_csv(d / "summary_all.csv", index=False)
    failures.to_csv(d / "failures_all.csv", index=False)

    print("=== TABLE 1: prompt stability by model ===")
    print(summary.to_string(index=False))
    print("\n=== TABLE 2: format failure reasons ===")
    print(failures.to_string(index=False) if len(failures) else "  none")

    # The number that decides whether the locked prompt is good enough.
    worst = summary.loc[summary["format_valid_rate"].idxmin()]
    print(f"\nworst-case format_valid_rate: {worst['format_valid_rate']}% "
          f"({worst['model']})")
    print("The prompt is only as good as its weakest model, since all four "
          "must be counted the same way.")

    by_template = summarize(records, group=["model", "template"])
    by_template.to_csv(d / "summary_by_template.csv", index=False)
    weak = by_template[by_template["format_valid_rate"] < 80]
    if len(weak):
        print(f"\n{len(weak)} model/template combinations below 80% format validity. "
              "If these cluster on particular properties, the issue is question "
              "phrasing rather than the suffix.")
        print(weak.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
