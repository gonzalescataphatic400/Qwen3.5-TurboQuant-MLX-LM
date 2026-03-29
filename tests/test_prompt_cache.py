from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pytest

from turbomlx import prompt_cache as prompt_cache_module
from turbomlx.prompt_cache import load_prompt_cache, save_prompt_cache


class FakeSerializableCache:
    def __init__(self, state, meta_state, *, extra_state=None):
        self._state = state
        self._meta_state = meta_state
        self._extra_state = extra_state or {}

    @property
    def state(self):
        return self._state

    @property
    def meta_state(self):
        return self._meta_state

    def prompt_cache_extra_state(self):
        return self._extra_state

    def restore_prompt_cache_extra_state(self, payload):
        self._extra_state = payload

    @classmethod
    def from_state(cls, state, meta_state):
        return cls(state, meta_state)


def _make_roundtrip_caches():
    dense_cache = FakeSerializableCache(
        state=(np.arange(6, dtype=np.float32).reshape(1, 1, 1, 6),),
        meta_state=(json.dumps({"kind": "dense"}),),
    )
    affine_cache = FakeSerializableCache(
        state=((np.arange(4, dtype=np.uint8).reshape(1, 1, 1, 4), np.ones((1, 1, 1, 4), dtype=np.float32)),),
        meta_state=(json.dumps({"kind": "affine"}),),
    )
    frozen_mixed_cache = FakeSerializableCache(
        state=(np.arange(4, dtype=np.uint8).reshape(1, 1, 1, 4),),
        meta_state=(json.dumps({"kind": "mixed-frozen"}),),
        extra_state={
            "calibration_tokens_seen": 256,
            "outlier_mask_frozen": True,
            "calibration_keys": None,
            "calibration_values": None,
        },
    )
    mid_calibration_cache = FakeSerializableCache(
        state=(np.zeros((0,), dtype=np.uint8),),
        meta_state=(json.dumps({"kind": "mixed-mid-calibration"}),),
        extra_state={
            "calibration_tokens_seen": 32,
            "outlier_mask_frozen": False,
            "calibration_keys": np.arange(24, dtype=np.float32).reshape(1, 1, 2, 12),
            "calibration_values": np.arange(24, dtype=np.float32).reshape(1, 1, 2, 12),
        },
    )
    return [dense_cache, affine_cache, frozen_mixed_cache, mid_calibration_cache]


def test_prompt_cache_roundtrip_preserves_state_meta_and_extra_and_writes_v2_metadata(tmp_path):
    path = tmp_path / "prompt-cache.tqcache"
    caches = _make_roundtrip_caches()

    save_prompt_cache(path, caches)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    assert payload["schema_version"] == 2
    assert all("cache_type_id" in entry for entry in payload["entries"])

    with pytest.warns(UserWarning, match="trusted-local-only"):
        restored = load_prompt_cache(path)

    assert len(restored) == 4
    assert np.array_equal(restored[0].state[0], caches[0].state[0])
    assert restored[1].meta_state == caches[1].meta_state
    assert restored[2].prompt_cache_extra_state()["outlier_mask_frozen"] is True
    assert restored[3].prompt_cache_extra_state()["calibration_tokens_seen"] == 32
    assert np.array_equal(
        restored[3].prompt_cache_extra_state()["calibration_keys"],
        caches[3].prompt_cache_extra_state()["calibration_keys"],
    )


def test_load_prompt_cache_supports_v1_class_path_fallback_with_warning(tmp_path):
    path = tmp_path / "prompt-cache-v1.tqcache"
    payload = {
        "schema_version": 1,
        "entries": [
            {
                "class_path": f"{__name__}.FakeSerializableCache",
                "state": (np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),),
                "meta_state": (json.dumps({"kind": "legacy"}),),
                "extra_state": {"flag": True},
            }
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = load_prompt_cache(path)

    messages = [str(item.message) for item in caught]
    assert any("trusted-local-only" in message for message in messages)
    assert any("deprecated class_path metadata" in message for message in messages)
    assert restored[0].meta_state == payload["entries"][0]["meta_state"]


def test_load_prompt_cache_prefers_cache_type_id_over_class_path(tmp_path, monkeypatch):
    path = tmp_path / "prompt-cache-v2.tqcache"
    monkeypatch.setitem(
        prompt_cache_module._CACHE_TYPE_TO_CLASS_PATH,
        "tests.fake.serializable.v1",
        f"{__name__}.FakeSerializableCache",
    )
    payload = {
        "schema_version": 2,
        "entries": [
            {
                "cache_type_id": "tests.fake.serializable.v1",
                "class_path": "broken.module.DoesNotExist",
                "state": (np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),),
                "meta_state": (json.dumps({"kind": "typed"}),),
                "extra_state": {},
            }
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.warns(UserWarning, match="trusted-local-only"):
        restored = load_prompt_cache(path)

    assert isinstance(restored[0], FakeSerializableCache)
    assert restored[0].meta_state == payload["entries"][0]["meta_state"]
