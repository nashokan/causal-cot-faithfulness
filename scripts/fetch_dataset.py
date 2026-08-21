#!/usr/bin/env python
"""Download the chainscope IPHR question set.

We fetch over HTTP rather than cloning chainscope, because that repo
contains filenames Windows cannot create (a reserved name `aux.txt`, and
model-response files with `:` in the name, which NTFS reserves for
alternate data streams). A full checkout fails on Windows no matter what
git config you use.

We take only what we need:
  - the 4 question-variant folders, filtered to the paper's dataset
    generation (`non-ambiguous-hard-2`): 29 properties x 4 folders = 116 files
  - instructions.yaml, which holds the exact prompt templates

Source: Arcuschin et al. 2025, arXiv:2503.08679
        https://github.com/jettjaniak/chainscope

Usage:
    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --out data/chainscope
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "jettjaniak/chainscope"
QUESTIONS_PATH = "chainscope/data/questions"
INSTRUCTIONS_PATH = "chainscope/data/instructions.yaml"
VARIANT_FOLDERS = ["gt_YES_1", "gt_NO_1", "lt_YES_1", "lt_NO_1"]
RAW = "https://raw.githubusercontent.com/" + REPO + "/main/{path}"

# The paper (v4) uses this generation of the dataset. Do NOT switch to the
# bare `wm-*.yaml` files: those are an older, more ambiguous set of 37
# properties, and are the source of the incorrect "7,400 pairs" figure.
DATASET_SUFFIX = "_non-ambiguous-hard-2.yaml"

EXPECTED_FILES = 116
EXPECTED_QUESTIONS = 9668


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cot-faith-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def list_repo_files() -> list[str]:
    """One API call returns the whole file tree. The per-directory endpoint
    would cost four calls, and the unauthenticated limit is 60 per hour."""
    url = f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
    try:
        tree = json.loads(get(url))
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            sys.exit("GitHub API rate limit hit (60/hour unauthenticated). "
                     "Wait an hour and rerun, or set GITHUB_TOKEN.")
        raise
    if tree.get("truncated"):
        sys.exit("GitHub returned a truncated tree; cannot list files reliably.")
    return [n["path"] for n in tree["tree"] if n["type"] == "blob"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chainscope",
                    help="destination root (default: data/chainscope)")
    args = ap.parse_args()

    root = Path(args.out)
    all_paths = list_repo_files()
    n_files = 0

    for folder in VARIANT_FOLDERS:
        prefix = f"{QUESTIONS_PATH}/{folder}/"
        wanted = [p for p in all_paths
                  if p.startswith(prefix)
                  and p[len(prefix):].startswith("wm-")
                  and p.endswith(DATASET_SUFFIX)]
        dest = root / "questions" / folder
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{folder}: {len(wanted)} files", flush=True)
        for path in wanted:
            target = dest / path.rsplit("/", 1)[-1]
            if not target.exists():
                target.write_bytes(get(RAW.format(path=path)))
            n_files += 1

    (root / "instructions.yaml").write_bytes(get(RAW.format(path=INSTRUCTIONS_PATH)))
    print("instructions.yaml: ok")

    print(f"\ndownloaded {n_files} question files into {root}")
    if n_files != EXPECTED_FILES:
        print(f"WARNING: expected {EXPECTED_FILES}. The upstream repo may have "
              "changed; check before trusting any downstream numbers.")


if __name__ == "__main__":
    main()
