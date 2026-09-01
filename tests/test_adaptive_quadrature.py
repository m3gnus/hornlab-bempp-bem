"""Wavelength-adaptive regular quadrature selection (BEAT CPU port)."""
import numpy as np
import pytest

from hornlab_bempp_bem.bie import (
    _p90_element_length,
    _select_regular_quadrature,
)
from hornlab_bempp_bem.config import SolveConfig


def _config(**kwargs) -> SolveConfig:
    return SolveConfig(**kwargs)


class TestSelectRegularQuadrature:
    def test_disabled_returns_base_orders_and_no_diagnostics(self):
        config = _config()
        slp, hyp, diag = _select_regular_quadrature(config, 10.0, 0.01)
        assert slp == config.slp_dlp_quadrature == 4
        assert hyp == config.hyp_adlp_quadrature == 4
        assert diag == {}

    def test_below_threshold_selects_low_order(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_kh_max=2.0,
            adaptive_quadrature_low_order=2,
        )
        # k*h = 100 * 0.01 = 1.0 <= 2.0
        slp, hyp, diag = _select_regular_quadrature(config, 100.0, 0.01)
        assert slp == 2
        assert hyp == 2
        assert diag["kh"] == pytest.approx(1.0)
        assert diag["slp_dlp_order"] == 2
        assert diag["base_slp_dlp_order"] == 4

    def test_above_threshold_keeps_base_order(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_kh_max=2.0,
            adaptive_quadrature_low_order=2,
        )
        # k*h = 1000 * 0.01 = 10.0 > 2.0
        slp, hyp, diag = _select_regular_quadrature(config, 1000.0, 0.01)
        assert slp == 4
        assert hyp == 4
        assert diag["kh"] == pytest.approx(10.0)
        assert diag["slp_dlp_order"] == 4

    def test_exact_threshold_is_inclusive(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_kh_max=2.0,
            adaptive_quadrature_low_order=2,
        )
        slp, _, _ = _select_regular_quadrature(config, 200.0, 0.01)
        assert slp == 2

    def test_below_kh_min_keeps_base_order(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_kh_min=0.4,
            adaptive_quadrature_kh_max=2.0,
            adaptive_quadrature_low_order=2,
        )
        # k*h = 10 * 0.01 = 0.1 < 0.4: the near-pair error zone.
        slp, hyp, diag = _select_regular_quadrature(config, 10.0, 0.01)
        assert slp == 4
        assert hyp == 4
        assert diag["kh"] == pytest.approx(0.1)

    def test_kh_min_zero_recovers_pure_upper_threshold(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_kh_min=0.0,
            adaptive_quadrature_kh_max=2.0,
            adaptive_quadrature_low_order=2,
        )
        slp, _, _ = _select_regular_quadrature(config, 10.0, 0.01)
        assert slp == 2

    def test_low_order_never_raises_above_base(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_kh_max=2.0,
            adaptive_quadrature_low_order=6,
            slp_dlp_quadrature=4,
            hyp_adlp_quadrature=3,
        )
        slp, hyp, _ = _select_regular_quadrature(config, 100.0, 0.01)
        assert slp == 4
        assert hyp == 3

    def test_singular_quadrature_untouched(self):
        config = _config(
            adaptive_quadrature=True,
            adaptive_quadrature_low_order=1,
        )
        _select_regular_quadrature(config, 100.0, 0.01)
        assert config.slp_dlp_singular_quadrature == 4


class TestP90ElementLength:
    def test_matches_sqrt_of_p90_area(self):
        class _Grid:
            volumes = np.linspace(1e-6, 1e-4, 100)

        expected = float(np.sqrt(np.percentile(_Grid.volumes, 90.0)))
        assert _p90_element_length(_Grid()) == pytest.approx(expected)


class TestConfigValidation:
    def test_defaults_are_off_and_valid(self):
        config = _config()
        assert config.adaptive_quadrature is False
        assert config.adaptive_quadrature_kh_min == 0.4
        assert config.adaptive_quadrature_kh_max == 2.0
        assert config.adaptive_quadrature_low_order == 2

    def test_rejects_kh_min_above_kh_max(self):
        with pytest.raises(ValueError, match="must not exceed"):
            _config(
                adaptive_quadrature_kh_min=3.0,
                adaptive_quadrature_kh_max=2.0,
            )

    @pytest.mark.parametrize("bad", [float("nan"), -0.1, None])
    def test_rejects_bad_kh_min(self, bad):
        with pytest.raises(ValueError, match="adaptive_quadrature_kh_min"):
            _config(adaptive_quadrature_kh_min=bad)

    def test_rejects_non_bool_enable(self):
        with pytest.raises(ValueError, match="adaptive_quadrature must be"):
            _config(adaptive_quadrature=1)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, None])
    def test_rejects_bad_kh_max(self, bad):
        with pytest.raises(ValueError, match="adaptive_quadrature_kh_max"):
            _config(adaptive_quadrature_kh_max=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, None])
    def test_rejects_bad_low_order(self, bad):
        with pytest.raises(ValueError, match="adaptive_quadrature_low_order"):
            _config(adaptive_quadrature_low_order=bad)
