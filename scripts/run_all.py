#!/usr/bin/env python
"""Run every model, build a PDF report, and push results to GitHub.

    python scripts/run_all.py

Kaggle recycles sessions without warning and wipes /kaggle/working when it
does. Nothing here relies on you downloading anything in time:

  Incremental.  Each model runs in its own subprocess and writes results to
                disk immediately.
  Pushed.       After EVERY model, results and the PDF are committed and
                force-pushed to the `results` branch of the repo. A session
                death costs at most the model that was mid-run.
  Resumable.    Rerunning pulls the `results` branch first, so completed
                models are skipped and the run continues where it stopped.

Requires GH_TOKEN in the environment (a GitHub personal access token with
`repo` scope). Without it the run still works, but results live only on the
Kaggle disk and you are back to downloading them by hand.

A model that fails is recorded in the report and does not stop the others.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_MODELS = ["llama3b", "deepseek8b", "qwen4b", "gemma9b"]
RESULTS_BRANCH = "results"


def sh(cmd: list[str], cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def remote_with_token(token: str) -> str | None:
    """Rewrite the origin URL to embed the token, so pushes need no prompt."""
    url = sh(["git", "remote", "get-url", "origin"]).stdout.strip()
    m = re.match(r"https://(?:[^@]+@)?github\.com/(.+?)(?:\.git)?$", url)
    return f"https://{token}@github.com/{m.group(1)}.git" if m else None


def git_ready() -> bool:
    """True when we can commit and push. Embeds the token in the remote URL
    so pushes are non-interactive."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token or not (REPO / ".git").exists():
        return False
    sh(["git", "config", "user.email", "runner@kaggle.local"])
    sh(["git", "config", "user.name", "kaggle-runner"])
    url = remote_with_token(token)
    if url:                       # None for non-GitHub remotes; harmless
        sh(["git", "remote", "set-url", "origin", url])
    return True


def pull_previous(results: Path) -> None:
    """Restore results from a previous session so the run can resume."""
    if not git_ready():
        return
    r = sh(["git", "fetch", "origin", RESULTS_BRANCH])
    if r.returncode != 0:
        print("no previous results branch, starting fresh", flush=True)
        return
    sh(["git", "checkout", "-B", RESULTS_BRANCH, f"origin/{RESULTS_BRANCH}"])
    n = len(list(results.glob("raw_*.jsonl"))) if results.exists() else 0
    print(f"restored {n} completed model(s) from the results branch", flush=True)


def push(msg: str) -> None:
    if not git_ready():
        print("  [no GH_TOKEN: results are on the Kaggle disk only]", flush=True)
        return
    sh(["git", "checkout", "-B", RESULTS_BRANCH])
    sh(["git", "add", "-f", "results"])          # -f: results/ is gitignored
    c = sh(["git", "commit", "-m", msg])
    if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr):
        print(f"  commit failed: {(c.stdout + c.stderr)[-200:]}", flush=True)
        return
    p = sh(["git", "push", "-f", "origin", RESULTS_BRANCH])
    print("  pushed to results branch" if p.returncode == 0
          else f"  push failed: {(p.stdout + p.stderr)[-200:]}", flush=True)


def already_done(results: Path, model: str) -> bool:
    f = results / f"raw_{model}.jsonl"
    return f.exists() and f.stat().st_size > 0


def run_model(model: str, results: Path, n_questions: int,
              extra: list[str]) -> dict:
    cmd = [sys.executable, str(REPO / "scripts" / "run_prompt_stability.py"),
           "--model", model, "--out-dir", str(results),
           "--n-questions", str(n_questions), *extra]
    print(f"\n{'=' * 70}\nRUNNING {model}\n{'=' * 70}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    ok = proc.returncode == 0
    msg = None
    if not ok:
        err = (proc.stderr or "").strip().splitlines()
        msg = next((l for l in reversed(err)
                    if l.strip() and not l.startswith(" ")), "unknown error")
        print(f"FAILED after {elapsed:.0f}s: {msg}", flush=True)
    else:
        print(f"done in {elapsed:.0f}s", flush=True)

    return {"model": model, "ok": ok, "seconds": round(elapsed, 1), "error": msg}


def snapshot(results: Path, status: list[dict], msg: str) -> None:
    """Write status, rebuild the PDF, push. Safe to call repeatedly."""
    (results / "run_status.json").write_text(json.dumps(status, indent=2))
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_report.py"),
                    "--results-dir", str(results),
                    "--out", str(results / "prompt_stability_report.pdf")],
                   capture_output=True, text=True)
    push(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun models that already have results")
    args, extra = ap.parse_known_args()

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    if not args.no_pull:
        pull_previous(results)
        results.mkdir(parents=True, exist_ok=True)

    status_file = results / "run_status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else []
    done = {s["model"] for s in status if s["ok"]}

    for model in args.models:
        if not args.force and (model in done or already_done(results, model)):
            print(f"SKIP {model} (already have results)", flush=True)
            continue
        status = [s for s in status if s["model"] != model]
        status.append(run_model(model, results, args.n_questions, extra))
        snapshot(results, status, f"results: {model}")

    snapshot(results, status, "results: final")

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for s in status:
        mark = "ok  " if s["ok"] else "FAIL"
        print(f"  {mark} {s['model']:12} {s['seconds']:>7.0f}s"
              f"{'' if s['ok'] else '  ' + str(s['error'])[:80]}")
    print(f"\nreport: {results / 'prompt_stability_report.pdf'}")
    print(f"branch: {RESULTS_BRANCH}")


if __name__ == "__main__":
    main()
