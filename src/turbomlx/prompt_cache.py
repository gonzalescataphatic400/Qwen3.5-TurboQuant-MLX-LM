"""TurboMLX-owned prompt-cache save/load helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import pickle
from typing import Any
import warnings

import numpy as np

from turbomlx.exceptions import PromptCacheSerializationError


_PROMPT_CACHE_SCHEMA_VERSION = 2
_SUPPORTED_PROMPT_CACHE_SCHEMA_VERSIONS = {1, 2}
_TRUSTED_LOCAL_ONLY_WARNING = (
    "TurboMLX prompt-cache loading is trusted-local-only and pickle-backed. "
    "Do not load untrusted prompt-cache files."
)
_CLASS_PATH_FALLBACK_WARNING = (
    "TurboMLX prompt-cache restore fell back to deprecated class_path metadata. "
    "Re-save this cache to migrate it to schema v2 cache_type_id metadata."
)
_CACHE_TYPE_TO_CLASS_PATH = {
    "turbomlx.cache.turboquant_kv.v1": "turbomlx.mlx_runtime.cache.TurboQuantKVCache",
}


def _class_path(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _cache_type_id(value: type) -> str:
    return getattr(value, "PROMPT_CACHE_TYPE_ID", _class_path(value))


def _load_class(class_path: str) -> type:
    module_name, _, attr_path = class_path.rpartition(".")
    if not module_name or not attr_path:
        raise PromptCacheSerializationError(f"Invalid prompt-cache class path: {class_path}")
    module = import_module(module_name)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise PromptCacheSerializationError(f"Prompt-cache class path did not resolve to a type: {class_path}")
    return value


def _load_cache_type(cache_type_id: str) -> type:
    class_path = _CACHE_TYPE_TO_CLASS_PATH.get(cache_type_id, cache_type_id)
    return _load_class(class_path)


def _normalize_state(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_normalize_state(item) for item in value)
    if isinstance(value, list):
        return [_normalize_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_state(item) for key, item in value.items()}
    if isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return value
    if hasattr(value, "shape") or hasattr(value, "__array__"):
        return np.asarray(value)
    return value


def _cache_extra_state(cache) -> dict[str, Any]:
    if hasattr(cache, "prompt_cache_extra_state"):
        return _normalize_state(cache.prompt_cache_extra_state())
    return {}


def _serialize_prompt_cache_entry(cache) -> dict[str, Any]:
    if not hasattr(cache, "state") or not hasattr(cache, "meta_state"):
        raise PromptCacheSerializationError(
            f"Prompt-cache entry {type(cache).__name__} must expose `state` and `meta_state`."
        )
    cache_cls = type(cache)
    return {
        "cache_type_id": _cache_type_id(cache_cls),
        "class_path": _class_path(cache_cls),
        "state": _normalize_state(cache.state),
        "meta_state": _normalize_state(cache.meta_state),
        "extra_state": _cache_extra_state(cache),
    }


def save_prompt_cache(path: str | Path, prompt_cache) -> Path:
    payload = {
        "schema_version": _PROMPT_CACHE_SCHEMA_VERSION,
        "entries": [_serialize_prompt_cache_entry(cache) for cache in prompt_cache],
    }
    destination = Path(path)
    with destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def _resolve_prompt_cache_class(entry: dict[str, Any]):
    cache_type_id = entry.get("cache_type_id")
    if cache_type_id is not None:
        try:
            return _load_cache_type(cache_type_id)
        except (PromptCacheSerializationError, AttributeError, ImportError, ModuleNotFoundError):
            pass

    class_path = entry.get("class_path")
    if not class_path:
        raise PromptCacheSerializationError(
            "Prompt-cache entry must include `cache_type_id` or `class_path` metadata."
        )
    warnings.warn(_CLASS_PATH_FALLBACK_WARNING, DeprecationWarning, stacklevel=3)
    return _load_class(class_path)


def _restore_prompt_cache_entry(entry: dict[str, Any]):
    cls = _resolve_prompt_cache_class(entry)
    if hasattr(cls, "from_prompt_cache_entry"):
        return cls.from_prompt_cache_entry(entry)
    if hasattr(cls, "from_state"):
        cache = cls.from_state(entry["state"], entry["meta_state"])
    else:
        raise PromptCacheSerializationError(
            f"Prompt-cache class {entry.get('class_path', cls)} does not implement `from_state` "
            "or `from_prompt_cache_entry`."
        )

    extra_state = entry.get("extra_state") or {}
    if extra_state and hasattr(cache, "restore_prompt_cache_extra_state"):
        cache.restore_prompt_cache_extra_state(extra_state)
    elif extra_state:
        raise PromptCacheSerializationError(
            f"Prompt-cache class {entry.get('class_path', cls)} does not accept TurboMLX extra state."
        )
    return cache


def load_prompt_cache(path: str | Path):
    warnings.warn(_TRUSTED_LOCAL_ONLY_WARNING, UserWarning, stacklevel=2)

    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)

    version = int(payload.get("schema_version", 0))
    if version not in _SUPPORTED_PROMPT_CACHE_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(_SUPPORTED_PROMPT_CACHE_SCHEMA_VERSIONS))
        raise PromptCacheSerializationError(
            f"Unsupported TurboMLX prompt-cache schema_version={version}; "
            f"expected one of {{{supported}}}."
        )

    return [_restore_prompt_cache_entry(entry) for entry in payload.get("entries", [])]
