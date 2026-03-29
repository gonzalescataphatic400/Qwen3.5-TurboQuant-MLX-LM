"""Shared random rotations keyed by head dimension."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codebooks import default_cache_dir


@dataclass(slots=True)
class RotationSpec:
    head_dim: int
    kind: str
    seed: int
    matrix: np.ndarray

    @property
    def identifier(self) -> str:
        return f"{self.kind}-d{self.head_dim}-s{self.seed}"


def _rotation_cache_dir(cache_dir: Path | None = None) -> Path:
    path = (cache_dir or default_cache_dir()) / "rotations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hadamard(order: int) -> np.ndarray:
    if order & (order - 1):
        raise ValueError("Walsh-Hadamard rotation requires a power-of-two head_dim")
    matrix = np.array([[1.0]], dtype=np.float32)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]]).astype(np.float32)
    return matrix


def generate_rotation(head_dim: int, *, kind: str = "qr", seed: int = 0) -> RotationSpec:
    rng = np.random.default_rng(seed)
    if kind == "qr":
        gaussian = rng.standard_normal((head_dim, head_dim), dtype=np.float64)
        q, r = np.linalg.qr(gaussian)
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        matrix = (q * signs).astype(np.float32)
    elif kind == "wht":
        had = _hadamard(head_dim)
        signs = rng.choice([-1.0, 1.0], size=(head_dim,)).astype(np.float32)
        matrix = (had * signs[None, :]) / np.sqrt(float(head_dim))
    else:
        raise ValueError(f"Unsupported rotation kind: {kind}")
    return RotationSpec(head_dim=head_dim, kind=kind, seed=seed, matrix=matrix)


def load_or_create_rotation(
    head_dim: int,
    *,
    kind: str = "qr",
    seed: int = 0,
    cache_dir: Path | None = None,
) -> RotationSpec:
    path = _rotation_cache_dir(cache_dir) / f"{kind}-d{head_dim}-s{seed}.npz"
    if path.exists():
        payload = np.load(path)
        return RotationSpec(
            head_dim=int(payload["head_dim"]),
            kind=str(payload["kind"]),
            seed=int(payload["seed"]),
            matrix=payload["matrix"].astype(np.float32),
        )
    spec = generate_rotation(head_dim, kind=kind, seed=seed)
    np.savez_compressed(
        path,
        head_dim=np.array(spec.head_dim, dtype=np.int32),
        kind=np.array(spec.kind),
        seed=np.array(spec.seed, dtype=np.int32),
        matrix=spec.matrix.astype(np.float32),
    )
    return spec


def apply_rotation(x: np.ndarray, spec: RotationSpec) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ spec.matrix.T


def apply_inverse_rotation(y: np.ndarray, spec: RotationSpec) -> np.ndarray:
    return np.asarray(y, dtype=np.float32) @ spec.matrix
