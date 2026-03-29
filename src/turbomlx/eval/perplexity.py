"""Backend-driven perplexity helpers."""

from __future__ import annotations

from collections import Counter
from math import exp, log
from pathlib import Path

import numpy as np

from turbomlx.mlx_runtime.generation import score_tokens_with_backend

from .runtime import encode_text, load_model_and_tokenizer


def unigram_perplexity_from_local_text(path: Path) -> dict[str, float | int]:
    text = path.read_text().split()
    if not text:
        return {"tokens": 0, "cross_entropy": 0.0, "perplexity": 1.0}
    counts = Counter(text)
    total = len(text)
    cross_entropy = -sum(log(counts[token] / total) for token in text) / total
    return {
        "tokens": total,
        "cross_entropy": cross_entropy,
        "perplexity": exp(cross_entropy),
    }


def perplexity_from_backend(model_id: str, text: str, *, backend: str, config) -> dict[str, float | int | str]:
    model, tokenizer = load_model_and_tokenizer(model_id)
    tokens = encode_text(tokenizer, text)
    token_logprobs, metrics = score_tokens_with_backend(model, tokens, backend=backend, config=config)
    logprobs_np = np.asarray(token_logprobs, dtype=np.float32)
    token_count = int(logprobs_np.size)
    if token_count == 0:
        cross_entropy = 0.0
        perplexity = 1.0
    else:
        cross_entropy = float(-np.mean(logprobs_np))
        perplexity = float(np.exp(cross_entropy))
    return {
        "model_id": model_id,
        "backend": backend,
        "tokens": token_count,
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        **metrics.as_dict(),
    }


def perplexity_from_text_file(model_id: str, path: Path, *, backend: str, config) -> dict[str, float | int | str]:
    return perplexity_from_backend(model_id, path.read_text(), backend=backend, config=config)
