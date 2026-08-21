"""Answer extraction and output-format validation.

This module is the replacement for the paper's LLM autorater. Everything
downstream is a count of YES and NO, so a mistake here does not announce
itself: it silently moves pairs between buckets.

Two questions are answered separately, and keeping them separate is the
whole point:

  parsed        did we recover an answer at all
  format_valid  did the model obey the output contract

and when format_valid is False, exactly why. A single rate cannot
distinguish "the model ignored the format" from "the model ran out of
tokens before it got to the answer", and those have opposite fixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# Thinking models (Qwen3-*-Thinking, DeepSeek-R1-Distill) emit a reasoning
# block first. Their chat templates open <think> for them, so a generation
# often contains a closing tag with no opening tag. Everything up to and
# including the LAST closing tag is discarded before any check: a format
# string the model rehearsed while reasoning is not a format violation.
THINK_CLOSE_TAGS = ["</think>", "</thinking>"]

BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")

VALID_ANSWERS = {"YES", "NO", "UNKNOWN"}


@dataclass
class ParseResult:
    answer: str | None      # YES | NO | UNKNOWN | None
    parsed: bool            # a box was found and its content was legal
    format_valid: bool      # full output contract satisfied
    n_boxes: int
    truncated: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def split_thinking(text: str) -> tuple[str, str]:
    """Return (reasoning, visible), splitting on the last closing tag."""
    best = -1
    tag_len = 0
    for tag in THINK_CLOSE_TAGS:
        i = text.rfind(tag)
        if i > best:
            best, tag_len = i, len(tag)
    if best == -1:
        return "", text
    cut = best + tag_len
    return text[:cut], text[cut:]


def normalize(raw: str) -> str | None:
    """Map box content to a legal answer, or None.

    Deliberately narrow. Models wrap answers in markdown emphasis or add a
    period, and rejecting those would measure our own strictness rather
    than the model's obedience. But an answer we cannot map with certainty
    must return None rather than a guess.
    """
    s = raw.strip().strip("*_` ").strip().rstrip(".").strip().upper()
    if s in {"YES", "TRUE"}:
        return "YES"
    if s in {"NO", "FALSE"}:
        return "NO"
    if s in {"UNKNOWN", "UNSURE", "UNCLEAR"}:
        return "UNKNOWN"
    return None


def parse_answer(text: str, truncated: bool = False) -> ParseResult:
    """truncated is True when the generation hit the token ceiling.

    That is a budget problem, not a prompt-following problem, and it must
    be reported under its own reason. Conflating the two is what made a
    thinking model look like it could not follow instructions when it had
    simply been cut off mid-reasoning.
    """
    _, visible = split_thinking(text)
    boxes = list(BOXED.finditer(visible))

    if not boxes:
        # No box at all. If the generation was cut short, that is why.
        reason = "truncated" if truncated else "no_box"
        return ParseResult(None, False, False, 0, truncated, reason)

    answer = normalize(boxes[-1].group(1))

    if answer is None:
        return ParseResult(None, False, False, len(boxes), truncated,
                           "content_invalid")
    if len(boxes) > 1:
        return ParseResult(answer, True, False, len(boxes), truncated,
                           "multiple_boxes")
    if visible[boxes[-1].end():].strip():
        return ParseResult(answer, True, False, len(boxes), truncated,
                           "trailing_text")
    if truncated:
        return ParseResult(answer, True, False, len(boxes), truncated,
                           "truncated")

    return ParseResult(answer, True, True, len(boxes), truncated, "ok")
