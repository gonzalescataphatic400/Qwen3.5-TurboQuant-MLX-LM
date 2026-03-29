import pytest

from turbomlx.mlx_runtime.config import (
    ScorerMode,
    TurboQuantConfig,
    TurboQuantMode,
    ValuesMode,
    default_mode_for_bits,
)


def test_default_mode_semantics_follow_plan():
    assert default_mode_for_bits(1) == TurboQuantMode.MSE
    assert default_mode_for_bits(2) == TurboQuantMode.PROD
    assert default_mode_for_bits(3) == TurboQuantMode.MSE


def test_prod_requires_at_least_two_bits():
    with pytest.raises(Exception):
        TurboQuantConfig(bits_total=1, mode=TurboQuantMode.PROD)


def test_values_mode_defaults_to_dense():
    config = TurboQuantConfig(bits_total=4)
    assert config.values_mode == ValuesMode.DENSE
    assert config.bits_mse == 4
    assert config.scorer_mode == ScorerMode.ORACLE_PREVIEW
