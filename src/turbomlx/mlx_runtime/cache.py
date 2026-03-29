"""TurboMLX KV cache runtime implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from turbomlx.core_ref.mixed_precision import MixedPrecisionKeyPathRef, MixedPrecisionProfile
from turbomlx.core_ref.quantizers import TurboQuantMSERef, TurboQuantProdRef
from turbomlx.exceptions import MissingDependencyError
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime, mlx_runtime_available
from turbomlx.mlx_runtime.config import (
    MixedPrecisionProfileConfig,
    ScorerMode,
    TurboQuantConfig,
    TurboQuantMode,
    ValuesMode,
)
from turbomlx.mlx_runtime.metrics import MemoryMetrics, recursive_nbytes
from turbomlx.mlx_runtime.qwen_native import (
    build_qwen_mse_score_keys_mlx,
    gather_mse_centroids_mlx,
    native_qwen_support_reason,
    qwen_grouped_mse_scores_mlx,
    supports_qwen_native_mlx,
    unpack_packed_indices_4bit_mlx,
)

_META_STATE_SCHEMA_VERSION = 1


def _effective_group_size(dim: int, preferred: int = 64) -> int:
    for candidate in (128, 64, 32, 16, 8, 4, 2, 1):
        if candidate <= preferred and candidate <= dim and dim % candidate == 0:
            return candidate
    return 1


def _none_if_empty(value):
    if value is None:
        return None
    if getattr(value, "size", None) == 0:
        return None
    return value


def _slice_token_axis(value, tokens: int, *, axis: int = 2):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_slice_token_axis(item, tokens, axis=axis) for item in value)
    if isinstance(value, list):
        return [_slice_token_axis(item, tokens, axis=axis) for item in value]
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) <= axis:
        return value
    length = max(int(tokens), 0)
    slicer = [slice(None)] * len(shape)
    slicer[axis] = slice(0, length)
    return value[tuple(slicer)]


def _token_axis_length(value, *, axis: int = 2) -> int:
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        for item in value:
            length = _token_axis_length(item, axis=axis)
            if length:
                return length
        return 0
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) <= axis:
        return 0
    return int(shape[axis])


def _mixed_precision_meta(config: MixedPrecisionProfileConfig | None):
    return config.as_dict() if config is not None else None


def _serialize_meta_state(
    *,
    config: TurboQuantConfig,
    head_dim: int | None,
    num_kv_heads: int | None,
    offset: int,
    value_group_size: int | None,
) -> tuple[str]:
    payload = {
        "schema_version": _META_STATE_SCHEMA_VERSION,
        "quantizer_version": config.quantizer_version,
        "mode": config.mode.value,
        "bits_total": config.bits_total,
        "bits_mse": config.bits_mse,
        "value_bits": config.value_bits,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "rotation_kind": config.rotation_kind,
        "rotation_seed": config.rotation_seed,
        "qjl_seed": config.qjl_seed,
        "codebook_version": config.codebook_version,
        "codebook_id": config.codebook_id(head_dim) if head_dim is not None else None,
        "packing_layout": config.packing_layout,
        "values_mode": config.values_mode_name,
        "value_group_size": value_group_size,
        "quantized_kv_start": config.quantized_kv_start,
        "mixed_precision_profile": config.mixed_precision_profile,
        "mixed_precision": _mixed_precision_meta(config.mixed_precision),
        "scorer_mode": config.scorer_mode_name,
        "offset": offset,
    }
    return (json.dumps(payload, sort_keys=True),)


def _deserialize_meta_state(value) -> dict[str, Any]:
    payload = json.loads(value[0] if isinstance(value, (list, tuple)) else value)
    version = int(payload.get("schema_version", 0))
    if version != _META_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported TurboMLX meta_state schema_version={version}; "
            f"expected {_META_STATE_SCHEMA_VERSION}"
        )
    return payload


def _safe_numpy_array(value) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception:
        if hasattr(value, "tolist"):
            return np.asarray(value.tolist())
        if hasattr(value, "item"):
            return np.asarray(value.item())
        raise


def _materialize_dense_keys_numpy(
    *,
    quantizer,
    regular_payload,
    outlier_payload=None,
    outlier_mask=None,
    calibration_keys=None,
    calibration_tokens_seen: int = 0,
    head_dim: int | None = None,
    mixed_precision_profile: str | None = None,
) -> np.ndarray:
    quantized_keys = None
    if regular_payload is not None:
        if isinstance(quantizer, MixedPrecisionKeyPathRef):
            from turbomlx.core_ref.mixed_precision import MixedPrecisionPayload

            if outlier_payload is None or outlier_mask is None:
                raise ValueError("Mixed-precision dense key reconstruction requires outlier payload and mask.")
            payload = MixedPrecisionPayload(
                outlier_mask=np.asarray(outlier_mask, dtype=bool),
                regular_payload=regular_payload,
                outlier_payload=outlier_payload,
                head_dim=head_dim if head_dim is not None else quantizer.head_dim,
                profile_name=mixed_precision_profile or "mixed_precision_preview",
            )
            quantized_keys = quantizer.dequantize(payload)
        else:
            quantized_keys = quantizer.dequantize(regular_payload)

    if calibration_keys is not None:
        calibration_np = _safe_numpy_array(_slice_token_axis(calibration_keys, calibration_tokens_seen))
        if quantized_keys is None:
            quantized_keys = calibration_np
        else:
            quantized_keys = np.concatenate([quantized_keys, calibration_np], axis=2)

    if quantized_keys is None:
        raise ValueError("Cannot materialize keys from an empty TurboQuantKVCache.")

    return np.asarray(quantized_keys, dtype=np.float32)


if not mlx_runtime_available():

    class TurboQuantKVCache:  # pragma: no cover - exercised only when mlx is absent
        PROMPT_CACHE_TYPE_ID = "turbomlx.cache.turboquant_kv.v1"

        def __init__(self, *_args, **_kwargs):
            raise MissingDependencyError(
                "TurboQuantKVCache requires `mlx` and `mlx-lm`. Install `turbomlx[mlx]`."
            )

        @classmethod
        def from_state(cls, *_args, **_kwargs):
            raise MissingDependencyError(
                "TurboQuantKVCache restore requires `mlx` and `mlx-lm`. Install `turbomlx[mlx]`."
            )

else:  # pragma: no cover - local environment lacks mlx, so runtime stays lightly verified
    mx, _base_mod, _cache_mod = ensure_mlx_runtime()
    _BaseCache = _cache_mod._BaseCache
    KVCache = _cache_mod.KVCache
    create_attention_mask = _cache_mod.create_attention_mask

    @dataclass(slots=True)
    class _RuntimePayload:
        packed_indices: Any
        norms: Any
        packed_signs: Any
        residual_norms: Any

    class TurboQuantKVCache(_BaseCache):
        """Versioned prompt-cache-compatible TurboQuant KV cache."""

        PROMPT_CACHE_TYPE_ID = "turbomlx.cache.turboquant_kv.v1"

        def __init__(self, config: TurboQuantConfig):
            self.config = config
            self.offset = 0
            self.head_dim: int | None = None
            self.num_kv_heads: int | None = None
            self.quantizer = None
            self.regular_payload = _RuntimePayload(None, None, None, None)
            self.outlier_payload = _RuntimePayload(None, None, None, None)
            self.outlier_mask = None
            self.values_main = None
            self.value_group_size = None
            self.rotation_matrix = None
            self.qjl_matrix = None
            self.centroids = None
            self.regular_index_map = None
            self.outlier_index_map = None
            self.calibration_tokens_seen = 0
            self.calibration_keys = None
            self.calibration_values = None
            self.preview_dense_keys = None
            self.native_rotated_values = None
            self.last_scorer_route = None

        def _append(self, current, update, axis: int):
            if update is None:
                return current
            if current is None:
                return update
            return mx.concatenate([current, update], axis=axis)

        def _append_quantized(self, current, update):
            if current is None:
                return update
            return tuple(mx.concatenate([c, u], axis=2) for c, u in zip(current, update))

        def _append_preview_dense_keys(self, keys):
            if keys is None:
                return
            dense_update = keys if getattr(keys, "dtype", None) == mx.float32 else keys.astype(mx.float32)
            self.preview_dense_keys = self._append(self.preview_dense_keys, dense_update, axis=2)

        def _invalidate_preview_dense_keys(self):
            self.preview_dense_keys = None

        def _invalidate_native_rotated_values(self):
            self.native_rotated_values = None

        def _tracks_preview_dense_keys_eagerly(self) -> bool:
            return self.config.scorer_mode != ScorerMode.NATIVE_MLX

        def _uses_native_qwen_payload_cache(self) -> bool:
            return (
                self.config.scorer_mode == ScorerMode.NATIVE_MLX
                and self.config.mode == TurboQuantMode.MSE
                and self.config.bits_total == 4
                and self.config.values_mode == ValuesMode.DENSE
                and self.config.mixed_precision is None
                and self.centroids is not None
                and self.rotation_matrix is not None
            )

        def _append_native_rotated_values(self, payload: _RuntimePayload):
            if not self._uses_native_qwen_payload_cache() or payload.packed_indices is None:
                return
            indices = unpack_packed_indices_4bit_mlx(payload.packed_indices, head_dim=self.head_dim)
            rotated_update = gather_mse_centroids_mlx(indices, self.centroids)
            self.native_rotated_values = self._append(self.native_rotated_values, rotated_update.astype(mx.float32), axis=2)

        def _empty_uint8(self):
            return mx.zeros((0,), dtype=mx.uint8)

        def _empty_float32(self):
            return mx.zeros((0,), dtype=mx.float32)

        def _empty_bool(self):
            return mx.zeros((0,), dtype=mx.bool_)

        def _to_mx(self, array, *, dtype=None):
            if array is None:
                return None
            return mx.array(array, dtype=dtype)

        def _to_mx_state(self, value):
            if value is None:
                return None
            if isinstance(value, tuple):
                return tuple(self._to_mx_state(item) for item in value)
            if isinstance(value, list):
                return [self._to_mx_state(item) for item in value]
            if isinstance(value, dict):
                return {key: self._to_mx_state(item) for key, item in value.items()}
            if hasattr(value, "shape") or hasattr(value, "__array__"):
                return mx.array(np.asarray(value))
            return value

        def _quantizer_kind(self):
            return TurboQuantProdRef if self.config.mode == TurboQuantMode.PROD else TurboQuantMSERef

        def _build_quantizer(self, head_dim: int):
            if self.config.mixed_precision is not None:
                profile = MixedPrecisionProfile(
                    outlier_channels=self.config.mixed_precision.outlier_channels,
                    outlier_high_bits=self.config.mixed_precision.outlier_high_bits,
                    regular_bits=self.config.mixed_precision.regular_bits,
                    selection_policy=self.config.mixed_precision.outlier_selection_policy.value,
                    mode=self.config.mode.value,
                    dim=head_dim,
                )
                return MixedPrecisionKeyPathRef(
                    head_dim,
                    profile,
                    codebook_version=self.config.codebook_version,
                    rotation_kind=self.config.rotation_kind,
                    rotation_seed=self.config.rotation_seed,
                    qjl_seed=self.config.qjl_seed,
                )
            quantizer_cls = self._quantizer_kind()
            quantizer_kwargs = {
                "codebook_version": self.config.codebook_version,
                "rotation_kind": self.config.rotation_kind,
                "rotation_seed": self.config.rotation_seed,
            }
            if quantizer_cls is TurboQuantProdRef:
                quantizer_kwargs["qjl_seed"] = self.config.qjl_seed
            return quantizer_cls(head_dim, self.config.bits_total, **quantizer_kwargs)

        def _sync_outlier_mask_state(self):
            if self.outlier_mask is None:
                self.regular_index_map = None
                self.outlier_index_map = None
                return
            mask_np = self._to_numpy(self.outlier_mask).astype(bool)
            if isinstance(self.quantizer, MixedPrecisionKeyPathRef):
                self.quantizer.freeze(outlier_mask=mask_np)
            self.outlier_mask = self._to_mx(mask_np.astype(np.bool_), dtype=mx.bool_)
            self.regular_index_map = self._to_mx(np.flatnonzero(~mask_np).astype(np.int32), dtype=mx.int32)
            self.outlier_index_map = self._to_mx(np.flatnonzero(mask_np).astype(np.int32), dtype=mx.int32)

        def _slice_runtime_payload(self, payload: _RuntimePayload, tokens: int) -> _RuntimePayload:
            return _RuntimePayload(
                _none_if_empty(_slice_token_axis(payload.packed_indices, tokens)),
                _none_if_empty(_slice_token_axis(payload.norms, tokens)),
                _none_if_empty(_slice_token_axis(payload.packed_signs, tokens)),
                _none_if_empty(_slice_token_axis(payload.residual_norms, tokens)),
            )

        def _slice_calibration_buffers(self, tokens: int):
            self.calibration_keys = _none_if_empty(_slice_token_axis(self.calibration_keys, tokens))
            self.calibration_values = _none_if_empty(_slice_token_axis(self.calibration_values, tokens))
            self.calibration_tokens_seen = min(self.calibration_tokens_seen, _token_axis_length(self.calibration_keys))

        def _compacted_backing_to_offset(self):
            backing_tokens = max(
                _token_axis_length(self.regular_payload.packed_indices),
                _token_axis_length(self.values_main),
                _token_axis_length(self.calibration_keys),
                _token_axis_length(self.calibration_values),
            )
            if self.offset >= backing_tokens:
                return
            self.regular_payload = self._slice_runtime_payload(self.regular_payload, self.offset)
            self.outlier_payload = self._slice_runtime_payload(self.outlier_payload, self.offset)
            self.values_main = _none_if_empty(_slice_token_axis(self.values_main, self.offset))
            self._slice_calibration_buffers(self.offset)
            self.preview_dense_keys = _none_if_empty(_slice_token_axis(self.preview_dense_keys, self.offset))
            self.native_rotated_values = _none_if_empty(_slice_token_axis(self.native_rotated_values, self.offset))

        def _ensure_initialized(self, keys):
            if self.head_dim is not None:
                return

            self.head_dim = int(keys.shape[-1])
            self.num_kv_heads = int(keys.shape[1])
            self.quantizer = self._build_quantizer(self.head_dim)
            self._refresh_runtime_matrices()

        def _refresh_runtime_matrices(self):
            if self.quantizer is None:
                return
            if isinstance(self.quantizer, MixedPrecisionKeyPathRef):
                self.rotation_matrix = None
                self.centroids = None
                self.qjl_matrix = None
                return
            base_quantizer = self.quantizer.mse if hasattr(self.quantizer, "mse") else self.quantizer
            self.rotation_matrix = self._to_mx(base_quantizer.rotation.matrix, dtype=mx.float32)
            self.centroids = self._to_mx(base_quantizer.centroids, dtype=mx.float32)
            self.qjl_matrix = (
                self._to_mx(self.quantizer.qjl.matrix, dtype=mx.float32)
                if hasattr(self.quantizer, "qjl")
                else None
            )

        def _to_numpy(self, array) -> np.ndarray:
            return _safe_numpy_array(array)

        def _materialize_payload(self, payload):
            def optional_mx(value, *, dtype):
                if value is None:
                    return None
                size = getattr(value, "size", None)
                if size == 0:
                    return None
                return self._to_mx(value, dtype=dtype)

            if hasattr(payload, "regular_payload"):
                payload_mask = payload.outlier_mask.astype(bool)
                existing_mask = (
                    self._to_numpy(self.outlier_mask).astype(bool) if self.outlier_mask is not None else None
                )
                if existing_mask is not None and not np.array_equal(existing_mask, payload_mask):
                    raise ValueError("Mixed-precision outlier mask changed after cache initialization.")
                self.outlier_mask = self._to_mx(payload_mask.astype(np.bool_), dtype=mx.bool_)
                self._sync_outlier_mask_state()
                regular = payload.regular_payload
                outlier = payload.outlier_payload
                return (
                    _RuntimePayload(
                        self._to_mx(regular.packed_indices, dtype=mx.uint8),
                        self._to_mx(regular.norms, dtype=mx.float32),
                        optional_mx(getattr(regular, "packed_signs", None), dtype=mx.uint8),
                        optional_mx(getattr(regular, "residual_norms", None), dtype=mx.float32),
                    ),
                    _RuntimePayload(
                        self._to_mx(outlier.packed_indices, dtype=mx.uint8),
                        self._to_mx(outlier.norms, dtype=mx.float32),
                        optional_mx(getattr(outlier, "packed_signs", None), dtype=mx.uint8),
                        optional_mx(getattr(outlier, "residual_norms", None), dtype=mx.float32),
                    ),
                )

            return (
                _RuntimePayload(
                    self._to_mx(payload.packed_indices, dtype=mx.uint8),
                    self._to_mx(payload.norms, dtype=mx.float32),
                    optional_mx(getattr(payload, "packed_signs", None), dtype=mx.uint8),
                    optional_mx(getattr(payload, "residual_norms", None), dtype=mx.float32),
                ),
                _RuntimePayload(None, None, None, None),
            )

        def _quantize_values(self, values):
            if self.config.values_mode == ValuesMode.DENSE:
                return values
            self.value_group_size = _effective_group_size(values.shape[-1])
            return mx.quantize(
                values.astype(mx.bfloat16),
                group_size=self.value_group_size,
                bits=self.config.value_bits,
                mode="affine",
            )

        def _append_payload(self, current: _RuntimePayload, update: _RuntimePayload) -> _RuntimePayload:
            return _RuntimePayload(
                self._append(current.packed_indices, update.packed_indices, axis=2),
                self._append(current.norms, update.norms, axis=2),
                self._append(current.packed_signs, update.packed_signs, axis=2),
                self._append(current.residual_norms, update.residual_norms, axis=2),
            )

        def _mixed_precision_prefix_target(self) -> int:
            if self.config.mixed_precision is None:
                return 0
            return max(int(self.config.mixed_precision.calibration_prefix_tokens), 0)

        def _uses_calibration_prefix(self) -> bool:
            return (
                isinstance(self.quantizer, MixedPrecisionKeyPathRef)
                and self.config.mixed_precision is not None
                and self.outlier_mask is None
                and self._mixed_precision_prefix_target() > 0
            )

        def _append_calibration_buffer(self, keys, values):
            if (
                self.quantizer.profile.selection_policy == "rolling_variance"
                and self.outlier_mask is None
            ):
                self.quantizer.calibrate(self._to_numpy(keys).reshape(-1, self.head_dim))
            self.calibration_keys = self._append(self.calibration_keys, keys, axis=2)
            self.calibration_values = self._append(self.calibration_values, values, axis=2)
            self.calibration_tokens_seen += int(keys.shape[2])

        def _finalize_calibration_prefix(self):
            if self.calibration_keys is None:
                return
            if self.quantizer.profile.selection_policy == "rolling_variance":
                mask = self.quantizer.freeze()
            else:
                mask = self.quantizer.freeze(
                    calibration_vectors=self._to_numpy(self.calibration_keys).reshape(-1, self.head_dim)
                )
            self.outlier_mask = self._to_mx(mask.astype(np.bool_), dtype=mx.bool_)
            self._sync_outlier_mask_state()
            payload = self.quantizer.quantize(self._to_numpy(self.calibration_keys), outlier_mask=mask)
            regular, outlier = self._materialize_payload(payload)
            self.regular_payload = self._append_payload(self.regular_payload, regular)
            if outlier.packed_indices is not None:
                self.outlier_payload = self._append_payload(self.outlier_payload, outlier)
            quantized_values = self._quantize_values(self.calibration_values)
            self.values_main = quantized_values
            self.calibration_keys = None
            self.calibration_values = None

        def _dense_keys_from_payloads(self, regular_payload, outlier_payload=None):
            regular_ref = None
            outlier_ref = None
            outlier_mask = None
            if regular_payload is not None and regular_payload.packed_indices is not None:
                if isinstance(self.quantizer, MixedPrecisionKeyPathRef):
                    regular_ref = self._payload_to_ref(regular_payload, quantizer=self.quantizer.regular)
                    outlier_ref = self._payload_to_ref(outlier_payload, quantizer=self.quantizer.outlier)
                    outlier_mask = self._to_numpy(self.outlier_mask).astype(bool)
                else:
                    regular_ref = self._payload_to_ref(regular_payload, quantizer=self.quantizer)
            return _materialize_dense_keys_numpy(
                quantizer=self.quantizer,
                regular_payload=regular_ref,
                outlier_payload=outlier_ref,
                outlier_mask=outlier_mask,
                head_dim=self.head_dim,
                mixed_precision_profile=self.config.mixed_precision_profile,
            )

        def _rebuild_preview_dense_keys(self):
            dense_keys = _materialize_dense_keys_numpy(
                quantizer=self.quantizer,
                regular_payload=(
                    self._payload_to_ref(
                        self._slice_runtime_payload(self.regular_payload, self.offset),
                        quantizer=self.quantizer.regular,
                    )
                    if isinstance(self.quantizer, MixedPrecisionKeyPathRef)
                    and self.regular_payload.packed_indices is not None
                    else (
                        self._payload_to_ref(
                            self._slice_runtime_payload(self.regular_payload, self.offset),
                            quantizer=self.quantizer,
                        )
                        if self.regular_payload.packed_indices is not None
                        else None
                    )
                ),
                outlier_payload=(
                    self._payload_to_ref(
                        self._slice_runtime_payload(self.outlier_payload, self.offset),
                        quantizer=self.quantizer.outlier,
                    )
                    if isinstance(self.quantizer, MixedPrecisionKeyPathRef)
                    and self.outlier_payload.packed_indices is not None
                    else None
                ),
                outlier_mask=(
                    self._to_numpy(self.outlier_mask).astype(bool)
                    if isinstance(self.quantizer, MixedPrecisionKeyPathRef) and self.outlier_mask is not None
                    else None
                ),
                calibration_keys=self.calibration_keys,
                calibration_tokens_seen=self.calibration_tokens_seen,
                head_dim=self.head_dim,
                mixed_precision_profile=self.config.mixed_precision_profile,
            )
            self.preview_dense_keys = self._to_mx(dense_keys, dtype=mx.float32)

        def _rebuild_native_rotated_values(self):
            if not self._uses_native_qwen_payload_cache():
                self.native_rotated_values = None
                return
            regular_payload = self._slice_runtime_payload(self.regular_payload, self.offset)
            if regular_payload.packed_indices is None:
                self.native_rotated_values = None
                return
            indices = unpack_packed_indices_4bit_mlx(regular_payload.packed_indices, head_dim=self.head_dim)
            self.native_rotated_values = gather_mse_centroids_mlx(indices, self.centroids).astype(mx.float32)

        def update_and_fetch(self, keys, values):
            self._ensure_initialized(keys)
            self._compacted_backing_to_offset()
            if self._uses_calibration_prefix():
                if self._tracks_preview_dense_keys_eagerly():
                    self._append_preview_dense_keys(keys.astype(mx.float32))
                self._append_calibration_buffer(keys, values)
                self.offset += int(keys.shape[2])
                if self.calibration_tokens_seen >= self._mixed_precision_prefix_target():
                    self._finalize_calibration_prefix()
                return self.key_state, self.value_state
            payload = self.quantizer.quantize(self._to_numpy(keys))
            regular, outlier = self._materialize_payload(payload)
            self.regular_payload = self._append_payload(self.regular_payload, regular)
            if outlier.packed_indices is not None:
                self.outlier_payload = self._append_payload(self.outlier_payload, outlier)
            self._append_native_rotated_values(regular)
            if self._tracks_preview_dense_keys_eagerly():
                self._append_preview_dense_keys(
                    self._to_mx(self._dense_keys_from_payloads(regular, outlier), dtype=mx.float32)
                )
            quantized_values = self._quantize_values(values)
            self.values_main = (
                self._append(self.values_main, quantized_values, axis=2)
                if self.config.values_mode == ValuesMode.DENSE
                else self._append_quantized(self.values_main, quantized_values)
            )
            self.offset += int(keys.shape[2])
            return self.key_state, self.value_state

        @property
        def key_state(self):
            regular = self._slice_runtime_payload(self.regular_payload, self.offset)
            outlier = self._slice_runtime_payload(self.outlier_payload, self.offset)
            return (
                regular.packed_indices,
                regular.norms,
                regular.packed_signs,
                regular.residual_norms,
                outlier.packed_indices,
                outlier.norms,
                outlier.packed_signs,
                outlier.residual_norms,
                self.outlier_mask if self.outlier_mask is not None else self._empty_bool(),
            )

        @property
        def value_state(self):
            if self.calibration_values is not None:
                return _slice_token_axis(self.calibration_values, self.offset)
            return _slice_token_axis(self.values_main, self.offset)

        @property
        def state(self):
            regular = self._slice_runtime_payload(self.regular_payload, self.offset)
            outlier = self._slice_runtime_payload(self.outlier_payload, self.offset)
            values = self.value_state
            return (
                regular.packed_indices if regular.packed_indices is not None else self._empty_uint8(),
                regular.norms if regular.norms is not None else self._empty_float32(),
                regular.packed_signs if regular.packed_signs is not None else self._empty_uint8(),
                regular.residual_norms if regular.residual_norms is not None else self._empty_float32(),
                outlier.packed_indices if outlier.packed_indices is not None else self._empty_uint8(),
                outlier.norms if outlier.norms is not None else self._empty_float32(),
                outlier.packed_signs if outlier.packed_signs is not None else self._empty_uint8(),
                outlier.residual_norms if outlier.residual_norms is not None else self._empty_float32(),
                self.outlier_mask if self.outlier_mask is not None else self._empty_bool(),
                values if values is not None else self._empty_float32(),
            )

        @state.setter
        def state(self, value):
            (
                reg_idx,
                reg_norms,
                reg_signs,
                reg_residuals,
                out_idx,
                out_norms,
                out_signs,
                out_residuals,
                outlier_mask,
                values_main,
            ) = value
            self.regular_payload = _RuntimePayload(
                _none_if_empty(self._to_mx_state(reg_idx)),
                _none_if_empty(self._to_mx_state(reg_norms)),
                _none_if_empty(self._to_mx_state(reg_signs)),
                _none_if_empty(self._to_mx_state(reg_residuals)),
            )
            self.outlier_payload = _RuntimePayload(
                _none_if_empty(self._to_mx_state(out_idx)),
                _none_if_empty(self._to_mx_state(out_norms)),
                _none_if_empty(self._to_mx_state(out_signs)),
                _none_if_empty(self._to_mx_state(out_residuals)),
            )
            self.outlier_mask = _none_if_empty(self._to_mx_state(outlier_mask))
            self.values_main = _none_if_empty(self._to_mx_state(values_main))
            self._invalidate_preview_dense_keys()
            self._invalidate_native_rotated_values()
            if self.offset == 0:
                self.offset = max(
                    _token_axis_length(self.regular_payload.packed_indices),
                    _token_axis_length(self.values_main),
                )
            self._sync_outlier_mask_state()

        @property
        def meta_state(self):
            return _serialize_meta_state(
                config=self.config,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                offset=self.offset,
                value_group_size=self.value_group_size,
            )

        @meta_state.setter
        def meta_state(self, value):
            payload = _deserialize_meta_state(value)
            mixed_precision = (
                MixedPrecisionProfileConfig.from_dict(payload["mixed_precision"])
                if payload.get("mixed_precision") is not None
                else None
            )
            self.config = TurboQuantConfig(
                bits_total=int(payload["bits_total"]),
                mode=TurboQuantMode(payload["mode"]),
                values_mode=ValuesMode(payload["values_mode"]),
                value_bits=int(payload.get("value_bits", 4)),
                scorer_mode=ScorerMode(payload.get("scorer_mode", ScorerMode.ORACLE_PREVIEW.value)),
                rotation_kind=payload["rotation_kind"],
                rotation_seed=int(payload["rotation_seed"]),
                qjl_seed=int(payload["qjl_seed"]),
                quantizer_version=payload["quantizer_version"],
                codebook_version=payload.get("codebook_version", self.config.codebook_version),
                quantized_kv_start=int(payload.get("quantized_kv_start", 0)),
                packing_layout=payload.get("packing_layout", "little_endian_compact"),
                mixed_precision=mixed_precision,
            )
            self.head_dim = int(payload["head_dim"]) if payload["head_dim"] is not None else None
            self.num_kv_heads = int(payload["num_kv_heads"]) if payload["num_kv_heads"] is not None else None
            self.offset = int(payload["offset"])
            self.value_group_size = (
                int(payload["value_group_size"]) if payload.get("value_group_size") is not None else None
            )
            self._invalidate_preview_dense_keys()
            self._invalidate_native_rotated_values()
            if payload.get("mixed_precision_profile", "disabled") != self.config.mixed_precision_profile:
                raise ValueError("mixed_precision_profile metadata does not match serialized config.")
            if self.head_dim is not None:
                self.quantizer = self._build_quantizer(self.head_dim)
                expected_codebook_id = self.config.codebook_id(self.head_dim)
                if payload.get("codebook_id") not in (None, expected_codebook_id):
                    raise ValueError(
                        f"codebook_id mismatch during restore: {payload.get('codebook_id')} != {expected_codebook_id}"
                    )
                self._refresh_runtime_matrices()
                self._sync_outlier_mask_state()

        def size(self):
            return self.offset

        def empty(self):
            return self.offset == 0

        def is_trimmable(self):
            return True

        def trim(self, n):
            n = min(self.offset, int(n))
            self.offset -= n
            self._compacted_backing_to_offset()
            return n

        def make_mask(self, *args, **kwargs):
            return create_attention_mask(*args, offset=self.offset, **kwargs)

        @property
        def nbytes(self):
            allocated_state = (
                self.regular_payload.packed_indices,
                self.regular_payload.norms,
                self.regular_payload.packed_signs,
                self.regular_payload.residual_norms,
                self.outlier_payload.packed_indices,
                self.outlier_payload.norms,
                self.outlier_payload.packed_signs,
                self.outlier_payload.residual_norms,
                self.outlier_mask,
                self.values_main,
                self.calibration_keys,
                self.calibration_values,
                self.preview_dense_keys,
                self.native_rotated_values,
            )
            return recursive_nbytes(allocated_state)

        def memory_metrics(self) -> MemoryMetrics:
            used_state = self.state
            used_key_bytes = recursive_nbytes(used_state[:9])
            used_value_bytes = recursive_nbytes(used_state[9:])
            calibration_key_bytes = recursive_nbytes(self.calibration_keys)
            calibration_value_bytes = recursive_nbytes(self.calibration_values)
            preview_dense_key_bytes = recursive_nbytes(_slice_token_axis(self.preview_dense_keys, self.offset))
            native_rotated_key_bytes = recursive_nbytes(_slice_token_axis(self.native_rotated_values, self.offset))
            return MemoryMetrics(
                allocated_cache_bytes=self.nbytes,
                used_state_bytes=(
                    recursive_nbytes(used_state)
                    + calibration_key_bytes
                    + calibration_value_bytes
                    + preview_dense_key_bytes
                    + native_rotated_key_bytes
                ),
                key_path_bytes=used_key_bytes + calibration_key_bytes + preview_dense_key_bytes,
                value_path_bytes=used_value_bytes + calibration_value_bytes,
                total_kv_bytes=(
                    used_key_bytes
                    + used_value_bytes
                    + calibration_key_bytes
                    + calibration_value_bytes
                    + preview_dense_key_bytes
                ),
                native_working_set_bytes=native_rotated_key_bytes,
            )

        def _payload_to_ref(self, payload: _RuntimePayload, *, quantizer):
            if payload.packed_indices is None:
                return None
            if isinstance(quantizer, TurboQuantProdRef):
                from turbomlx.core_ref.quantizers import ProdPayload

                ref_payload = ProdPayload(
                    packed_indices=self._to_numpy(payload.packed_indices),
                    packed_signs=self._to_numpy(payload.packed_signs),
                    norms=self._to_numpy(payload.norms),
                    residual_norms=self._to_numpy(payload.residual_norms),
                    head_dim=quantizer.head_dim,
                    bits_total=quantizer.bits_total,
                    bits_mse=quantizer.bits_mse,
                    codebook_id=quantizer.codebook_id,
                )
                quantizer.validate_payload(ref_payload)
                return ref_payload
            from turbomlx.core_ref.quantizers import MSEPayload

            ref_payload = MSEPayload(
                packed_indices=self._to_numpy(payload.packed_indices),
                norms=self._to_numpy(payload.norms),
                head_dim=quantizer.head_dim,
                bits_total=quantizer.bits_total,
                codebook_id=quantizer.codebook_id,
            )
            quantizer.validate_payload(ref_payload)
            return ref_payload

        def materialize_keys(self, *, dtype=None):
            if self.offset == 0:
                if self.head_dim is None or self.num_kv_heads is None:
                    return mx.zeros((0,), dtype=dtype or mx.float32)
                return mx.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=dtype or mx.float32)
            if (
                self.preview_dense_keys is None
                or _token_axis_length(self.preview_dense_keys) != self.offset
            ):
                self._rebuild_preview_dense_keys()
            dense_keys = self.preview_dense_keys
            if dense_keys is None:
                self._rebuild_preview_dense_keys()
                dense_keys = self.preview_dense_keys
            if dtype is not None and getattr(dense_keys, "dtype", None) != dtype:
                return dense_keys.astype(dtype)
            return dense_keys

        def native_qwen_support_reason(self, queries, value_state):
            return native_qwen_support_reason(self.config, self, queries, value_state)

        def supports_qwen_native_mlx(self, queries, value_state) -> bool:
            return supports_qwen_native_mlx(self.config, self, queries, value_state)

        def native_qwen_score_keys(self, *, dtype=None):
            if self.offset == 0:
                if self.head_dim is None or self.num_kv_heads is None:
                    return mx.zeros((0,), dtype=dtype or mx.float32)
                return mx.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=dtype or mx.float32)
            if not self._uses_native_qwen_payload_cache():
                raise ValueError("TurboMLX native_mlx Qwen scorer is not configured for this cache.")
            if self.native_rotated_values is None or _token_axis_length(self.native_rotated_values) != self.offset:
                self._rebuild_native_rotated_values()
            regular = self._slice_runtime_payload(self.regular_payload, self.offset)
            return build_qwen_mse_score_keys_mlx(
                self.native_rotated_values,
                regular.norms,
                dtype=dtype or mx.float32,
            )

        def score_qwen_grouped_queries_mlx(self, queries):
            if self.offset == 0:
                shape = tuple(int(dim) for dim in queries.shape[:-1]) + (0,)
                return mx.zeros(shape, dtype=queries.dtype)
            return qwen_grouped_mse_scores_mlx(
                queries,
                packed_indices=None,
                norms=None,
                centroids=self.centroids,
                rotation_matrix=self.rotation_matrix,
                kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                score_keys=self.native_qwen_score_keys(dtype=mx.float32),
            ).astype(queries.dtype)

        def score_queries(self, queries):
            if self.offset == 0:
                shape = tuple(int(dim) for dim in queries.shape[:-1]) + (0,)
                return mx.zeros(shape, dtype=queries.dtype)
            dense_scores = None
            if self.calibration_keys is not None:
                dense_scores = mx.matmul(queries, mx.swapaxes(self.calibration_keys, -1, -2))
                if self.regular_payload.packed_indices is None:
                    return dense_scores.astype(queries.dtype)
            query_np = self._to_numpy(queries)
            if isinstance(self.quantizer, MixedPrecisionKeyPathRef):
                from turbomlx.core_ref.mixed_precision import MixedPrecisionPayload

                mask = self._to_numpy(self.outlier_mask).astype(bool)
                payload = MixedPrecisionPayload(
                    outlier_mask=mask,
                    regular_payload=self._payload_to_ref(
                        self._slice_runtime_payload(self.regular_payload, self.offset),
                        quantizer=self.quantizer.regular,
                    ),
                    outlier_payload=self._payload_to_ref(
                        self._slice_runtime_payload(self.outlier_payload, self.offset),
                        quantizer=self.quantizer.outlier,
                    ),
                    head_dim=self.head_dim,
                    profile_name=self.config.mixed_precision_profile,
                )
                scores = self.quantizer.score_matrix(query_np, payload)
            else:
                payload = self._payload_to_ref(
                    self._slice_runtime_payload(self.regular_payload, self.offset),
                    quantizer=self.quantizer,
                )
                scores = self.quantizer.score_matrix(query_np, payload)
            quantized_scores = mx.array(scores, dtype=queries.dtype)
            if dense_scores is not None:
                return mx.concatenate([quantized_scores, dense_scores.astype(queries.dtype)], axis=-1)
            return quantized_scores

        @classmethod
        def from_kvcache(cls, cache, config: TurboQuantConfig):
            instance = cls(config)
            if cache.keys is not None:
                instance.update_and_fetch(cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :])
            return instance

        @classmethod
        def from_state(cls, state, meta_state):
            payload = _deserialize_meta_state(meta_state)
            mixed_precision = (
                MixedPrecisionProfileConfig.from_dict(payload["mixed_precision"])
                if payload.get("mixed_precision") is not None
                else None
            )
            config = TurboQuantConfig(
                bits_total=int(payload["bits_total"]),
                mode=TurboQuantMode(payload["mode"]),
                values_mode=ValuesMode(payload["values_mode"]),
                value_bits=int(payload.get("value_bits", 4)),
                scorer_mode=ScorerMode(payload.get("scorer_mode", ScorerMode.ORACLE_PREVIEW.value)),
                rotation_kind=payload["rotation_kind"],
                rotation_seed=int(payload["rotation_seed"]),
                qjl_seed=int(payload["qjl_seed"]),
                quantizer_version=payload["quantizer_version"],
                codebook_version=payload.get("codebook_version", "v1-lloyd-max-beta"),
                quantized_kv_start=int(payload.get("quantized_kv_start", 0)),
                packing_layout=payload.get("packing_layout", "little_endian_compact"),
                mixed_precision=mixed_precision,
            )
            instance = cls(config)
            instance.meta_state = meta_state
            instance.state = state
            return instance

        def prompt_cache_extra_state(self) -> dict[str, Any]:
            return {
                "calibration_tokens_seen": int(self.calibration_tokens_seen),
                "outlier_mask_frozen": self.outlier_mask is not None,
                "calibration_keys": _slice_token_axis(self.calibration_keys, self.calibration_tokens_seen),
                "calibration_values": _slice_token_axis(self.calibration_values, self.calibration_tokens_seen),
            }

        def restore_prompt_cache_extra_state(self, payload: dict[str, Any]):
            self.calibration_tokens_seen = int(payload.get("calibration_tokens_seen", 0))
            self.calibration_keys = _none_if_empty(self._to_mx_state(payload.get("calibration_keys")))
            self.calibration_values = _none_if_empty(self._to_mx_state(payload.get("calibration_values")))
            if (
                self.calibration_keys is not None
                and isinstance(self.quantizer, MixedPrecisionKeyPathRef)
                and self.quantizer.profile.selection_policy == "rolling_variance"
                and self.outlier_mask is None
            ):
                self.quantizer.calibrate(self._to_numpy(self.calibration_keys).reshape(-1, self.head_dim))
            if payload.get("outlier_mask_frozen") and self.outlier_mask is None and self.calibration_keys is not None:
                self._finalize_calibration_prefix()
