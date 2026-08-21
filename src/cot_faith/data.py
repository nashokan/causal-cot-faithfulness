"""Load the chainscope IPHR question set.

Vocabulary, fixed here so the rest of the code can rely on it:

  question      one prompt. 9,668 of them.
  pair          two questions, same property, same operator, same two
                entities, reversed order. A correct model must answer
                them oppositely. 4,834 pairs.
  entity pair   the two entities themselves, e.g. two counties. Each
                generates two pairs (one gt, one lt). 2,417 of them.
  template      property + operator, e.g. wm-us-county-lat + gt.
                58 templates. Criterion (ii) of the bias filter operates
                at this level.

Dataset generation: `non-ambiguous-hard-2`, which is the set used in
Arcuschin et al. 2025 v4 (29 properties, 4,834 pairs). The bare `wm-*.yaml`
files in the upstream repo are an older 37-property set with ambiguous
questions and must not be used.
"""

from __future__ import annotations

import glob
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

VARIANT_FOLDERS = ["gt_YES_1", "gt_NO_1", "lt_YES_1", "lt_NO_1"]
DATASET_SUFFIX = "_non-ambiguous-hard-2.yaml"

EXPECTED_QUESTIONS = 9668
EXPECTED_PAIRS = 4834
EXPECTED_TEMPLATES = 58


@dataclass
class Question:
    qid: str            # chainscope's own hash id
    prop_id: str        # e.g. wm-us-county-lat
    comparison: str     # gt | lt
    template: str       # prop_id + ":" + comparison
    pair_id: str        # shared by this question and its reversed twin
    q_str: str          # question text, already includes "about US counties:"
    gold: str           # YES | NO
    x_name: str
    y_name: str
    x_value: float
    y_value: float

    def as_dict(self) -> dict:
        return asdict(self)


def _pair_key(prop_id: str, comparison: str, x: str, y: str) -> str:
    """Order-independent, so a question and its reverse share a key."""
    a, b = sorted([x, y])
    return f"{prop_id}:{comparison}:{a}|{b}"


def load_questions(root: str | Path = "data/chainscope") -> list[Question]:
    root = Path(root)
    qdir = root / "questions"
    out: list[Question] = []

    for folder in VARIANT_FOLDERS:
        pattern = str(qdir / folder / f"wm-*{DATASET_SUFFIX}")
        for path in sorted(glob.glob(pattern)):
            blob = yaml.safe_load(open(path))
            p = blob["params"]
            for qid, r in blob["question_by_qid"].items():
                out.append(Question(
                    qid=qid,
                    prop_id=p["prop_id"],
                    comparison=p["comparison"],
                    template=f"{p['prop_id']}:{p['comparison']}",
                    pair_id=_pair_key(p["prop_id"], p["comparison"],
                                      r["x_name"], r["y_name"]),
                    q_str=r["q_str"],
                    gold=str(p["answer"]).upper(),
                    x_name=r["x_name"],
                    y_name=r["y_name"],
                    x_value=float(r["x_value"]),
                    y_value=float(r["y_value"]),
                ))

    if not out:
        raise FileNotFoundError(
            f"no question files under {qdir}. Run scripts/fetch_dataset.py first.")
    return sorted(out, key=lambda q: (q.template, q.pair_id, q.gold))


def group_pairs(questions: list[Question]) -> dict[str, list[Question]]:
    pairs = defaultdict(list)
    for q in questions:
        pairs[q.pair_id].append(q)
    return dict(pairs)


def validate(questions: list[Question]) -> dict:
    """Structural checks. Any failure means the dataset is not what we think."""
    pairs = group_pairs(questions)
    templates = {q.template for q in questions}

    bad_size = {k: len(v) for k, v in pairs.items() if len(v) != 2}

    # Every pair must have one gold YES and one gold NO.
    bad_gold = [k for k, v in pairs.items()
                if sorted(q.gold for q in v) != ["NO", "YES"]]

    # The declared gold label must agree with the raw values. Catches any
    # mislabelled file before it silently corrupts every downstream count.
    bad_values = []
    for q in questions:
        x_greater = q.x_value > q.y_value
        expected = ("YES" if x_greater else "NO") if q.comparison == "gt" \
            else ("YES" if not x_greater else "NO")
        if expected != q.gold:
            bad_values.append(q.qid)

    return {
        "questions": len(questions),
        "pairs": len(pairs),
        "templates": len(templates),
        "pairs_not_size_2": len(bad_size),
        "pairs_bad_gold": len(bad_gold),
        "gold_value_mismatches": len(bad_values),
        "ok": (len(questions) == EXPECTED_QUESTIONS
               and len(pairs) == EXPECTED_PAIRS
               and len(templates) == EXPECTED_TEMPLATES
               and not bad_size and not bad_gold and not bad_values),
    }


def stratified_sample(questions: list[Question], n: int,
                      seed: int = 42) -> list[Question]:
    """Sample n questions spread across all templates.

    Prompt stability is a property of question phrasing, and phrasing varies
    by template. A random sample would over-represent large templates
    (most have 200 questions, one has 4), so we take round-robin across
    templates instead. Within a template we keep gold YES and gold NO
    balanced, so answer_accuracy is not confounded by a skewed sample.
    """
    rng = random.Random(seed)

    by_template = defaultdict(list)
    for q in questions:
        by_template[q.template].append(q)

    # Shuffle within each template, interleaving YES and NO so that taking
    # a prefix of any length stays roughly balanced.
    queues = {}
    for i, t in enumerate(sorted(by_template)):
        qs = by_template[t]
        yes = [q for q in qs if q.gold == "YES"]
        no = [q for q in qs if q.gold == "NO"]
        rng.shuffle(yes)
        rng.shuffle(no)
        # Alternate which label leads, template by template. Each template
        # contributes only 3 or 4 questions to a 200-sample, so a prefix of
        # an always-YES-first list would skew the whole sample toward YES.
        first, second = (yes, no) if i % 2 == 0 else (no, yes)
        merged = []
        for a, b in zip(first, second):
            merged += [a, b]
        merged += first[len(second):] + second[len(first):]
        queues[t] = merged

    order = sorted(queues)
    rng.shuffle(order)

    picked: list[Question] = []
    idx = 0
    while len(picked) < n:
        progressed = False
        for t in order:
            if idx < len(queues[t]):
                picked.append(queues[t][idx])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break          # exhausted every template
        idx += 1

    return picked
