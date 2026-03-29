"""Reference packing utilities for packed payload serialization."""

from __future__ import annotations

from math import ceil

import numpy as np


def packed_nbytes(last_dim: int, bits: int) -> int:
    return ceil(last_dim * bits / 8)


def pack_bits(indices: np.ndarray, bits: int) -> np.ndarray:
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    values = np.asarray(indices, dtype=np.uint16)
    last_dim = values.shape[-1]
    packed_dim = packed_nbytes(last_dim, bits)
    out = np.zeros(values.shape[:-1] + (packed_dim,), dtype=np.uint8)
    mask = (1 << bits) - 1

    flat_values = values.reshape(-1, last_dim)
    flat_out = out.reshape(-1, packed_dim)
    for row_index, row in enumerate(flat_values):
        bit_cursor = 0
        for value in row:
            remaining = int(value) & mask
            byte_index = bit_cursor // 8
            offset = bit_cursor % 8
            flat_out[row_index, byte_index] |= (remaining << offset) & 0xFF
            spill = offset + bits - 8
            if spill > 0:
                flat_out[row_index, byte_index + 1] |= remaining >> (bits - spill)
            bit_cursor += bits
    return out


def unpack_bits(packed: np.ndarray, bits: int, original_dim: int) -> np.ndarray:
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    values = np.asarray(packed, dtype=np.uint8)
    out = np.zeros(values.shape[:-1] + (original_dim,), dtype=np.uint8)
    mask = (1 << bits) - 1

    flat_values = values.reshape(-1, values.shape[-1])
    flat_out = out.reshape(-1, original_dim)
    for row_index, row in enumerate(flat_values):
        bit_cursor = 0
        for coord in range(original_dim):
            byte_index = bit_cursor // 8
            offset = bit_cursor % 8
            raw = int(row[byte_index]) >> offset
            spill = offset + bits - 8
            if spill > 0 and byte_index + 1 < row.shape[0]:
                raw |= int(row[byte_index + 1]) << (8 - offset)
            flat_out[row_index, coord] = raw & mask
            bit_cursor += bits
    return out


def pack_sign_bits(signs: np.ndarray) -> np.ndarray:
    sign_bits = (np.asarray(signs) > 0).astype(np.uint8)
    return np.packbits(sign_bits, axis=-1, bitorder="little")


def unpack_sign_bits(packed: np.ndarray, original_dim: int) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, bitorder="little")
    bits = bits[..., :original_dim]
    return (bits.astype(np.int8) * 2) - 1
