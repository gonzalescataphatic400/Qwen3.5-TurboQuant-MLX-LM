from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from turbomlx.mlx_runtime import attention
from turbomlx.mlx_runtime.cache import TurboQuantKVCache
from turbomlx.mlx_runtime.config import ScorerMode, TurboQuantConfig


class _FakeFastNamespace:
    def __init__(self):
        self.calls = []

    def scaled_dot_product_attention(self, queries, keys, values, *, scale, mask):
        self.calls.append(
            {
                "queries_shape": tuple(int(dim) for dim in queries.shape),
                "keys_shape": tuple(int(dim) for dim in keys.shape),
                "values_shape": tuple(int(dim) for dim in values.shape),
                "scale": scale,
                "mask": mask,
            }
        )
        return np.zeros((queries.shape[0], queries.shape[1], queries.shape[2], values.shape[-1]), dtype=np.float32)


class _FakeMx:
    def __init__(self):
        self.fast = _FakeFastNamespace()


class _FakeTurboQuantKVCache:
    def __init__(self, dense_keys, *, scorer_mode=ScorerMode.ORACLE_PREVIEW, support_reason=None):
        self._dense_keys = dense_keys
        self.config = SimpleNamespace(
            values_mode_name="dense",
            value_bits=4,
            scorer_mode=scorer_mode,
        )
        self.value_group_size = None
        self.num_kv_heads = int(dense_keys.shape[1])
        self.head_dim = int(dense_keys.shape[-1])
        self.offset = int(dense_keys.shape[2])
        self.last_scorer_route = None
        self._support_reason = support_reason

    def materialize_keys(self, *, dtype=None):
        return self._dense_keys.astype(dtype or self._dense_keys.dtype, copy=False)

    def native_qwen_support_reason(self, _queries, _value_state):
        return self._support_reason

    def native_qwen_score_keys(self, *, dtype=None):
        return self._dense_keys.astype(dtype or self._dense_keys.dtype, copy=False)

    def score_qwen_grouped_queries_mlx(self, queries):
        batch, query_heads, query_len, _ = queries.shape
        return np.zeros((batch, query_heads, query_len, self.offset), dtype=np.float32)

    def score_queries(self, queries):
        batch, query_heads, query_len, _ = queries.shape
        return np.zeros((batch, query_heads, query_len, self.offset), dtype=np.float32)


def _require_mlx():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    return mx


def test_uses_qwen_grouped_attention_detects_qwen_like_head_layout():
    queries = np.zeros((1, 16, 2, 4), dtype=np.float32)
    dense_keys = np.zeros((1, 4, 5, 4), dtype=np.float32)
    assert attention.uses_qwen_grouped_attention(queries, dense_keys) is True


def test_uses_qwen_grouped_attention_rejects_non_divisible_head_layout():
    queries = np.zeros((1, 10, 2, 4), dtype=np.float32)
    dense_keys = np.zeros((1, 4, 5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        attention.uses_qwen_grouped_attention(queries, dense_keys)


def test_qwen_grouped_preview_attention_uses_native_sdpa(monkeypatch):
    fake_mx = _FakeMx()
    monkeypatch.setattr(attention, "_MX_RUNTIME_READY", True)
    monkeypatch.setattr(attention, "mx", fake_mx)
    queries = np.zeros((1, 16, 2, 4), dtype=np.float32)
    dense_keys = np.zeros((1, 4, 5, 4), dtype=np.float32)
    values = np.zeros((1, 4, 5, 4), dtype=np.float32)

    output = attention.qwen_grouped_preview_attention(
        queries,
        dense_keys,
        values,
        scale=0.5,
        mask=None,
    )

    assert output.shape == (1, 16, 2, 4)
    assert fake_mx.fast.calls == [
        {
            "queries_shape": (1, 16, 2, 4),
            "keys_shape": (1, 4, 5, 4),
            "values_shape": (1, 4, 5, 4),
            "scale": 0.5,
            "mask": None,
        }
    ]


def test_turboquant_attention_routes_supported_native_mlx_to_explicit_native_helper(monkeypatch):
    cache = _FakeTurboQuantKVCache(
        np.zeros((1, 4, 5, 4), dtype=np.float32),
        scorer_mode=ScorerMode.NATIVE_MLX,
        support_reason=None,
    )
    queries = np.zeros((1, 16, 2, 4), dtype=np.float32)
    values = np.zeros((1, 4, 5, 4), dtype=np.float32)
    called = {}

    monkeypatch.setattr(attention, "_MX_RUNTIME_READY", True)
    monkeypatch.setattr(attention, "mx", object())
    monkeypatch.setattr(attention, "TurboQuantKVCache", _FakeTurboQuantKVCache)

    def fake_native(q, v, c, *, scale, mask):
        called["args"] = (tuple(q.shape), tuple(v.shape), c, scale, mask)
        return np.zeros((1, 16, 2, 4), dtype=np.float32)

    monkeypatch.setattr(attention, "qwen_grouped_native_attention", fake_native)

    attention.turboquant_scaled_dot_product_attention(
        queries,
        None,
        values,
        cache,
        scale=0.5,
        mask=None,
    )

    assert called["args"] == ((1, 16, 2, 4), (1, 4, 5, 4), cache, 0.5, None)
    assert cache.last_scorer_route == "native_mlx"


def test_turboquant_attention_warns_once_and_falls_back_for_unsupported_native_mlx(monkeypatch):
    cache = _FakeTurboQuantKVCache(
        np.zeros((1, 4, 5, 4), dtype=np.float32),
        scorer_mode=ScorerMode.NATIVE_MLX,
        support_reason=("mode", "native scorer unsupported here"),
    )
    queries = np.zeros((1, 16, 2, 4), dtype=np.float32)
    values = np.zeros((1, 4, 5, 4), dtype=np.float32)
    warnings_seen = []
    preview_calls = []

    monkeypatch.setattr(attention, "_MX_RUNTIME_READY", True)
    monkeypatch.setattr(attention, "mx", object())
    monkeypatch.setattr(attention, "TurboQuantKVCache", _FakeTurboQuantKVCache)
    monkeypatch.setattr(attention, "_NATIVE_FALLBACK_WARNED_KEYS", set())
    monkeypatch.setattr(attention.warnings, "warn", lambda message, *args, **kwargs: warnings_seen.append(message))

    def fake_preview(q, k, v, *, scale, mask):
        preview_calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), scale, mask))
        return np.zeros((1, 16, 2, 4), dtype=np.float32)

    monkeypatch.setattr(attention, "qwen_grouped_preview_attention", fake_preview)

    attention.turboquant_scaled_dot_product_attention(queries, None, values, cache, scale=0.5, mask=None)
    attention.turboquant_scaled_dot_product_attention(queries, None, values, cache, scale=0.5, mask=None)

    assert len(warnings_seen) == 1
    assert "Falling back to preview grouped attention scoring." in warnings_seen[0]
    assert len(preview_calls) == 2
    assert cache.last_scorer_route == "native_mlx_fallback"


def test_qwen_grouped_native_attention_matches_dense_preview_reference_within_tolerance():
    mx = _require_mlx()
    rng = np.random.default_rng(2)
    config = TurboQuantConfig(bits_total=4, scorer_mode=ScorerMode.NATIVE_MLX)
    cache = TurboQuantKVCache(config)
    keys = mx.array(rng.standard_normal((1, 4, 5, 8), dtype=np.float32))
    values = mx.array(rng.standard_normal((1, 4, 5, 8), dtype=np.float32))
    queries = mx.array(rng.standard_normal((1, 16, 2, 8), dtype=np.float32))
    cache.update_and_fetch(keys, values)

    native_output = attention.qwen_grouped_native_attention(
        queries,
        values,
        cache,
        scale=0.5,
        mask=None,
    )
    preview_output = attention.qwen_grouped_preview_attention(
        queries,
        cache.materialize_keys(dtype=queries.dtype),
        values,
        scale=0.5,
        mask=None,
    )

    np.testing.assert_allclose(np.array(native_output), np.array(preview_output), atol=2e-2)
