"""Generation backends.

Two of them. `vllm` is the real one. `mock` produces synthetic generations
covering the same failure modes as the test suite, so the entire pipeline
can be exercised end to end on a laptop with no GPU and no quota spent.

Run mock first. If the metrics come out right on generations whose correct
answers we already know, the pipeline is correct and the only remaining
unknown is the models themselves.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    max_model_len: int
    thinking: bool
    tensor_parallel_size: int = 1
    dtype: str = "float16"          # T4 and P100 have no bfloat16 support
    gpu_memory_utilization: float = 0.90

    @classmethod
    def from_config(cls, key: str, cfg: dict, tp_override: int | None = None):
        return cls(
            key=key,
            hf_id=cfg["hf_id"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            max_new_tokens=cfg["max_new_tokens"],
            max_model_len=cfg["max_model_len"],
            thinking=cfg["thinking"],
            tensor_parallel_size=tp_override or cfg.get("tensor_parallel_size", 1),
        )


def generate_vllm(chats: list[list[dict]], spec: ModelSpec,
                  seed: int) -> list[tuple[str, bool, int]]:
    """Returns (text, truncated, n_output_tokens) per chat, in order."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=spec.hf_id,
        dtype=spec.dtype,
        max_model_len=spec.max_model_len,
        tensor_parallel_size=spec.tensor_parallel_size,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        trust_remote_code=True,
        seed=seed,
    )
    params = SamplingParams(
        temperature=spec.temperature,
        top_p=spec.top_p,
        top_k=spec.top_k,
        max_tokens=spec.max_new_tokens,
        seed=seed,
        n=1,
    )
    outputs = llm.chat(chats, params)
    results = []
    for out in outputs:
        c = out.outputs[0]
        results.append((c.text, c.finish_reason == "length", len(c.token_ids)))
    return results


# Rough proportions for the mock, chosen to resemble a model that mostly
# behaves but exhibits every failure mode often enough to be visible.
_MOCK_MIX = [
    ("clean", 0.62),
    ("thinking_clean", 0.10),
    ("rehearsed", 0.06),
    ("unknown", 0.06),
    ("no_box", 0.04),
    ("multiple", 0.04),
    ("trailing", 0.04),
    ("invalid", 0.02),
    ("truncated", 0.02),
]


def generate_mock(chats: list[list[dict]], spec: ModelSpec,
                  seed: int, golds: list[str] | None = None
                  ) -> list[tuple[str, bool, int]]:
    """Synthetic generations. No GPU, no network, fully deterministic."""
    rng = random.Random(seed + hash(spec.key) % 100000)
    kinds = [k for k, _ in _MOCK_MIX]
    weights = [w for _, w in _MOCK_MIX]
    results = []

    for i, _ in enumerate(chats):
        gold = golds[i] if golds else rng.choice(["YES", "NO"])
        # Mostly correct, sometimes not, so answer_accuracy is not trivially 100.
        ans = gold if rng.random() < 0.8 else ("NO" if gold == "YES" else "YES")
        kind = rng.choices(kinds, weights)[0]
        reasoning = "Comparing the two values step by step. "

        if kind == "clean":
            text, trunc = f"{reasoning}\n\n\\boxed{{{ans}}}", False
        elif kind == "thinking_clean":
            text, trunc = f"{reasoning}</think>\n\n\\boxed{{{ans}}}", False
        elif kind == "rehearsed":
            text = (f"I must end with \\boxed{{YES}} or \\boxed{{NO}}. {reasoning}"
                    f"</think>\n\n\\boxed{{{ans}}}")
            trunc = False
        elif kind == "unknown":
            text, trunc = f"{reasoning}The values look equal.\n\n\\boxed{{UNKNOWN}}", False
        elif kind == "no_box":
            text, trunc = f"{reasoning}The answer is {ans}.", False
        elif kind == "multiple":
            text = f"{reasoning}</think>\n\\boxed{{YES}} wait, \\boxed{{{ans}}}"
            trunc = False
        elif kind == "trailing":
            text = f"{reasoning}\n\n\\boxed{{{ans}}}\n\nHope that helps!"
            trunc = False
        elif kind == "invalid":
            text, trunc = f"{reasoning}\n\n\\boxed{{probably {ans.lower()}}}", False
        else:  # truncated
            text, trunc = f"{reasoning}Now the second value is roughly", True

        results.append((text, trunc, len(text.split())))
    return results
