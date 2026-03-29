"""Needle-in-a-haystack preview helpers without prompt leakage."""

from __future__ import annotations

import random
import re

from turbomlx.mlx_runtime.generation import generate_with_backend

from .runtime import decode_tokens, encode_text, load_model_and_tokenizer


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def insert_needle_into_context(
    context: str,
    needle: str,
    *,
    insertion_depth_pct: int = 50,
    seed: int = 0,
) -> str:
    if not 0 <= insertion_depth_pct <= 100:
        raise ValueError("insertion_depth_pct must be between 0 and 100")
    if not context.strip():
        return needle

    rng = random.Random(seed)
    lines = context.splitlines()
    if len(lines) > 1:
        base_index = int(round((len(lines) - 1) * (insertion_depth_pct / 100.0)))
        jitter_window = max(len(lines) // 20, 1)
        insert_at = max(0, min(len(lines), base_index + rng.randint(-jitter_window, jitter_window)))
        lines.insert(insert_at, needle)
        return "\n".join(lines)

    words = context.split()
    if not words:
        return needle
    base_index = int(round(len(words) * (insertion_depth_pct / 100.0)))
    jitter_window = max(len(words) // 20, 1)
    insert_at = max(0, min(len(words), base_index + rng.randint(-jitter_window, jitter_window)))
    merged = words[:insert_at] + [needle] + words[insert_at:]
    return " ".join(merged)


def build_needle_prompt(context: str, query: str) -> str:
    return (
        "You are given a long context.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer using only the context. Return only the answer."
    )


def score_needle_answer(answer: str, needle: str) -> dict[str, float]:
    normalized_answer = _normalize_text(answer)
    normalized_needle = _normalize_text(needle)
    exact_match = 1.0 if normalized_answer == normalized_needle else 0.0
    substring_match = (
        1.0
        if normalized_answer
        and normalized_needle
        and (normalized_needle in normalized_answer or normalized_answer in normalized_needle)
        else 0.0
    )
    return {"exact_match": exact_match, "substring_match": substring_match}


def run_needle_backend_eval(
    model_id: str,
    context: str,
    needle: str,
    query: str,
    *,
    backend: str,
    config,
    insertion_depth_pct: int = 50,
    seed: int = 0,
    max_tokens: int = 64,
) -> dict[str, float | int | str]:
    model, tokenizer = load_model_and_tokenizer(model_id)
    haystack_context = insert_needle_into_context(
        context,
        needle,
        insertion_depth_pct=insertion_depth_pct,
        seed=seed,
    )
    prompt = build_needle_prompt(haystack_context, query)
    tokens = encode_text(tokenizer, prompt)
    generated, _logprobs, stats = generate_with_backend(
        model,
        tokens,
        max_tokens=max_tokens,
        backend=backend,
        config=config,
    )
    answer = decode_tokens(tokenizer, generated)
    score = score_needle_answer(answer, needle)
    return {
        "model_id": model_id,
        "backend": backend,
        "prompt_characters": len(prompt),
        "answer": answer,
        "needle_score": score["exact_match"],
        "needle_substring_score": score["substring_match"],
        "insertion_depth_pct": insertion_depth_pct,
        "seed": seed,
        **stats.as_dict(),
    }
