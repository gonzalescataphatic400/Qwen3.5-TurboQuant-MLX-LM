"""Shared runtime-backed evaluation helpers."""

from __future__ import annotations

from turbomlx.exceptions import MissingDependencyError
from turbomlx.mlx_runtime.availability import mlx_runtime_available


def load_model_and_tokenizer(model_id: str):
    if not mlx_runtime_available():
        raise MissingDependencyError("MLX runtime dependencies are missing.")
    loader = __import__("mlx_lm", fromlist=["load"]).load
    return loader(model_id)


def encode_text(tokenizer, text: str):
    tokens = tokenizer.encode(text, return_tensors="mlx")
    if getattr(tokens, "ndim", 1) > 1:
        return tokens[0]
    return tokens


def decode_tokens(tokenizer, tokens) -> str:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens)
