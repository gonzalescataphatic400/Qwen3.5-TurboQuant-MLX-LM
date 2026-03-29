"""Logit comparison utilities."""

from __future__ import annotations

import numpy as np


def logit_cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    denom = (np.linalg.norm(ref) * np.linalg.norm(cand)) + 1e-8
    return float(np.dot(ref, cand) / denom)
