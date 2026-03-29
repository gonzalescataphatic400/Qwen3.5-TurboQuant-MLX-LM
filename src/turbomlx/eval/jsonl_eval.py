"""Local JSONL backend evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from turbomlx.mlx_runtime.generation import generate_with_backend

from .runtime import decode_tokens, encode_text, load_model_and_tokenizer


def evaluate_jsonl_file(path: Path) -> dict[str, float | int]:
    total = 0
    exact = 0
    substring = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = str(row.get("prediction", "")).strip().lower()
            answers = [str(answer).strip().lower() for answer in row.get("answers", [])]
            total += 1
            if prediction in answers:
                exact += 1
            if any(answer in prediction or prediction in answer for answer in answers):
                substring += 1
    return {
        "rows": total,
        "exact_match": exact / total if total else 0.0,
        "substring_match": substring / total if total else 0.0,
    }


def _row_prompt(row: dict) -> str:
    for key in ("prompt", "input", "question", "context"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError("JSONL row must include one of: prompt, input, question, context")


def evaluate_jsonl_backend(
    model_id: str,
    path: Path,
    *,
    backend: str,
    config,
    max_tokens: int = 64,
    limit: int | None = None,
) -> dict[str, float | int | str]:
    model, tokenizer = load_model_and_tokenizer(model_id)
    total = 0
    exact = 0
    substring = 0

    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = _row_prompt(row)
            answers = [str(answer).strip().lower() for answer in row.get("answers", [])]
            tokens = encode_text(tokenizer, prompt)
            generated, _logprobs, _stats = generate_with_backend(
                model,
                tokens,
                max_tokens=max_tokens,
                backend=backend,
                config=config,
            )
            prediction = decode_tokens(tokenizer, generated).strip().lower()
            total += 1
            if prediction in answers:
                exact += 1
            if any(answer in prediction or prediction in answer for answer in answers):
                substring += 1
            if limit is not None and total >= limit:
                break

    return {
        "model_id": model_id,
        "backend": backend,
        "rows": total,
        "exact_match": exact / total if total else 0.0,
        "substring_match": substring / total if total else 0.0,
    }
