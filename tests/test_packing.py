import numpy as np

from turbomlx.core_ref.packing import pack_bits, pack_sign_bits, unpack_bits, unpack_sign_bits


def test_pack_unpack_roundtrip_for_multiple_bitwidths():
    rng = np.random.default_rng(0)
    for bits in (1, 2, 3, 4):
        levels = 1 << bits
        indices = rng.integers(0, levels, size=(3, 5, 17), dtype=np.uint8)
        packed = pack_bits(indices, bits)
        unpacked = unpack_bits(packed, bits, 17)
        assert np.array_equal(indices, unpacked)


def test_pack_unpack_sign_bits_roundtrip():
    signs = np.array([[1, -1, 1, 1, -1, -1, 1, -1, 1]], dtype=np.int8)
    packed = pack_sign_bits(signs)
    unpacked = unpack_sign_bits(packed, signs.shape[-1])
    assert np.array_equal(signs, unpacked)
