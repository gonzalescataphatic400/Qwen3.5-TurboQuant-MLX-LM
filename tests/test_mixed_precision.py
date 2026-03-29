import numpy as np

from turbomlx.core_ref.mixed_precision import MixedPrecisionKeyPathRef, MixedPrecisionProfile


def test_outlier_selection_respects_channel_budget():
    rng = np.random.default_rng(0)
    calibration = rng.standard_normal((128, 12), dtype=np.float32)
    calibration[:, 3] *= 8
    calibration[:, 7] *= 6
    ref = MixedPrecisionKeyPathRef(
        12,
        MixedPrecisionProfile(
            outlier_channels=2,
            outlier_high_bits=3,
            regular_bits=2,
            mode="mse",
            dim=12,
        ),
    )
    mask = ref.select_outliers(calibration)
    assert mask.sum() == 2
    assert mask[3] or mask[7]


def test_mixed_precision_score_matrix_runs():
    rng = np.random.default_rng(1)
    keys = rng.standard_normal((1, 1, 10, 12), dtype=np.float32)
    queries = rng.standard_normal((1, 1, 4, 12), dtype=np.float32)
    ref = MixedPrecisionKeyPathRef(
        12,
        MixedPrecisionProfile(
            outlier_channels=3,
            outlier_high_bits=4,
            regular_bits=2,
            mode="mse",
            dim=12,
        ),
    )
    payload = ref.quantize(keys)
    scores = ref.score_matrix(queries, payload)
    assert scores.shape == (1, 1, 4, 10)


def test_mixed_precision_freezes_mask_after_first_quantize():
    rng = np.random.default_rng(2)
    first = rng.standard_normal((1, 1, 8, 10), dtype=np.float32)
    second = rng.standard_normal((1, 1, 8, 10), dtype=np.float32)
    first[..., 2] *= 8
    second[..., 7] *= 12
    ref = MixedPrecisionKeyPathRef(
        10,
        MixedPrecisionProfile(
            outlier_channels=2,
            outlier_high_bits=4,
            regular_bits=2,
            selection_policy="rolling_variance",
            mode="mse",
            dim=10,
        ),
    )
    payload_first = ref.quantize(first)
    payload_second = ref.quantize(second)
    assert np.array_equal(payload_first.outlier_mask, payload_second.outlier_mask)
    assert np.array_equal(ref.frozen_outlier_mask, payload_first.outlier_mask)


def test_rolling_variance_policy_accumulates_before_freeze():
    batch_a = np.zeros((64, 8), dtype=np.float32)
    batch_b = np.zeros((8, 8), dtype=np.float32)
    batch_a[:, 1] = np.linspace(-10.0, 10.0, 64, dtype=np.float32)
    batch_b[:, 4] = np.linspace(-3.0, 3.0, 8, dtype=np.float32)
    ref = MixedPrecisionKeyPathRef(
        8,
        MixedPrecisionProfile(
            outlier_channels=2,
            outlier_high_bits=3,
            regular_bits=2,
            selection_policy="rolling_variance",
            mode="mse",
            dim=8,
        ),
    )
    ref.calibrate(batch_a)
    mask = ref.calibrate(batch_b)
    assert mask.sum() == 2
    assert mask[1]


def test_rolling_variance_can_freeze_after_restore_like_recalibration():
    batch = np.zeros((16, 8), dtype=np.float32)
    batch[:, 5] = np.linspace(-4.0, 4.0, 16, dtype=np.float32)
    ref = MixedPrecisionKeyPathRef(
        8,
        MixedPrecisionProfile(
            outlier_channels=2,
            outlier_high_bits=3,
            regular_bits=2,
            selection_policy="rolling_variance",
            mode="mse",
            dim=8,
        ),
    )
    ref.calibrate(batch)
    mask = ref.freeze()
    assert mask.sum() == 2
    assert mask[5]
