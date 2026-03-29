"""Compatibility alias for legacy local JSONL evaluation helpers.

This module does not implement real LongBench-E scoring. Prefer
`turbomlx.eval.jsonl_eval` and the `eval-jsonl` CLI command for preview use.
"""

from __future__ import annotations

from .jsonl_eval import evaluate_jsonl_backend, evaluate_jsonl_file

evaluate_longbench_backend = evaluate_jsonl_backend
evaluate_longbench_file = evaluate_jsonl_file

__all__ = [
    "evaluate_jsonl_backend",
    "evaluate_jsonl_file",
    "evaluate_longbench_backend",
    "evaluate_longbench_file",
]
