"""``vectorization_mode`` selects the OpenCL kernel vector width.

Forcing a wider vector than the device's native one is worth about 1.08x on
assemblies with a large trial space and nothing on small ones, so the default
stays at bempp-cl's own ``auto``. What must hold unconditionally is that the
setting is validated, reaches bempp-cl, and never silently changes results.
"""
from __future__ import annotations

import sys
import types

import pytest

from hornlab_bempp_bem.config import VECTORIZATION_MODES, SolveConfig


def test_the_default_is_bempps_own_behaviour():
    assert SolveConfig().vectorization_mode == "auto"


@pytest.mark.parametrize("mode", VECTORIZATION_MODES)
def test_every_documented_mode_is_accepted(mode):
    assert SolveConfig(vectorization_mode=mode).vectorization_mode == mode


@pytest.mark.parametrize("mode", ["vec32", "VEC16", "", None, 8, "fastest"])
def test_an_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="vectorization_mode"):
        SolveConfig(vectorization_mode=mode)


@pytest.fixture
def fake_bempp(monkeypatch):
    module = types.ModuleType("bempp_cl.api")
    module.VECTORIZATION_MODE = "auto"
    root = types.ModuleType("bempp_cl")
    root.api = module
    monkeypatch.setitem(sys.modules, "bempp_cl", root)
    monkeypatch.setitem(sys.modules, "bempp_cl.api", module)
    return module


@pytest.mark.parametrize("mode", VECTORIZATION_MODES)
def test_the_mode_reaches_bempp(fake_bempp, mode):
    from hornlab_bempp_bem.device import set_vectorization_mode

    assert set_vectorization_mode(mode) == mode
    assert fake_bempp.VECTORIZATION_MODE == mode


def test_the_mode_is_normalised(fake_bempp):
    from hornlab_bempp_bem.device import set_vectorization_mode

    assert set_vectorization_mode("  VEC16 ") == "vec16"
    assert fake_bempp.VECTORIZATION_MODE == "vec16"


def test_an_unknown_mode_never_reaches_bempp(fake_bempp):
    from hornlab_bempp_bem.device import OpenCLError, set_vectorization_mode

    with pytest.raises(OpenCLError, match="vectorization_mode"):
        set_vectorization_mode("vec32")

    assert fake_bempp.VECTORIZATION_MODE == "auto", "must be left untouched"


def test_operator_kwargs_defaults_to_auto_for_an_old_config(fake_bempp):
    """`getattr(config, ...)` keeps a config without the field working."""
    from hornlab_bempp_bem.bie import _operator_kwargs

    kwargs = _operator_kwargs("numba", "single")

    assert "device_interface" in kwargs
    # numba path must not touch the OpenCL global at all
    assert fake_bempp.VECTORIZATION_MODE == "auto"
