"""Aggregate per-generation records into the prompt-stability metrics.

Herick's three metrics, which Darpan approved, plus two additions that the
original report needed and did not have.

  parse_rate         a legal answer was recovered
  format_valid_rate  the full output contract held. PRIMARY metric.
  answer_accuracy    the answer matched gold

  unknown_rate       the model answered UNKNOWN. Reported separately: a
                     refusal is a real behaviour, not a format failure,
                     and the paper treats refusal rate as a finding.
  truncated_rate     the generation hit the token ceiling.

answer_accuracy is computed over YES/NO answers only. Scoring UNKNOWN as
wrong would confound "answered incorrectly" with "declined to answer",
which are different behaviours with different implications.
"""

from __future__ import annotations

import pandas as pd

GROUP = ["model"]


def summarize(records: list[dict], group: list[str] | None = None) -> pd.DataFrame:
    group = group or GROUP
    df = pd.DataFrame(records)
    if df.empty:
        return df

    rows = []
    for keys, g in df.groupby(group, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        decided = g[g["answer"].isin(["YES", "NO"])]
        correct = (decided["answer"] == decided["gold"]).mean() if len(decided) else None
        rows.append({
            **dict(zip(group, keys)),
            "n": len(g),
            "parse_rate": round(100 * g["parsed"].mean(), 1),
            "format_valid_rate": round(100 * g["format_valid"].mean(), 1),
            "answer_accuracy": round(100 * correct, 1) if correct is not None else None,
            "n_decided": len(decided),
            "unknown_rate": round(100 * (g["answer"] == "UNKNOWN").mean(), 1),
            "truncated_rate": round(100 * g["truncated"].mean(), 1),
            "mean_gen_tokens": int(g["gen_tokens"].mean()) if "gen_tokens" in g else None,
        })
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def failure_breakdown(records: list[dict]) -> pd.DataFrame:
    """Why format validation failed. This is the diagnostic table.

    A rate alone cannot tell you whether to change the prompt or raise the
    token budget. This table can.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return df
    bad = df[~df["format_valid"]]
    if bad.empty:
        return pd.DataFrame(columns=["model", "reason", "count", "pct_of_all"])
    tot = df.groupby("model").size()
    out = bad.groupby(["model", "reason"]).size().reset_index(name="count")
    out["pct_of_all"] = out.apply(
        lambda r: round(100 * r["count"] / tot[r["model"]], 1), axis=1)
    return out.sort_values(["model", "count"], ascending=[True, False]).reset_index(drop=True)
