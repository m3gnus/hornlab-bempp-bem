from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hornlab_bempp_bem
from hornlab_bempp_bem.bie import solve_single_frequency
from hornlab_bempp_bem.config import SolveConfig


def test_solve_rejects_native_symmetry_before_mesh_load(monkeypatch):
    calls = []

    def fake_load_mesh(mesh, *, scale=1.0):
        calls.append((mesh, scale))
        raise AssertionError("load_mesh should not be called")

    monkeypatch.setattr(hornlab_bempp_bem, "load_mesh", fake_load_mesh)

    with pytest.raises(NotImplementedError, match="does not mirror reduced meshes"):
        hornlab_bempp_bem.solve(
            "quarter.msh",
            SolveConfig(native_symmetry_plane="yz+xz"),
        )

    assert calls == []


def test_solve_single_frequency_rejects_native_symmetry_before_assembly():
    grid = SimpleNamespace(number_of_elements=1)

    with pytest.raises(NotImplementedError, match="hornlab-metal-bem"):
        solve_single_frequency(
            grid,
            np.array([2], dtype=np.int32),
            1000.0,
            SolveConfig(native_symmetry_plane="yz"),
        )
