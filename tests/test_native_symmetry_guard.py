from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hornlab_bempp_bem
from hornlab_bempp_bem.bie import solve_single_frequency
from hornlab_bempp_bem.config import SolveConfig
from hornlab_bempp_bem.mesh import MeshError


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


def test_solve_single_frequency_require_closed_mesh_rejects_open_grid():
    grid = SimpleNamespace(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ).T,
        elements=np.asarray(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
            ],
            dtype=np.int32,
        ).T,
        number_of_elements=3,
    )

    with pytest.raises(MeshError, match="open boundary edges"):
        solve_single_frequency(
            grid,
            np.array([1, 1, 2], dtype=np.int32),
            1000.0,
            SolveConfig(require_closed_mesh=True),
        )


def test_solve_single_frequency_skips_recheck_for_validated_sweep_mesh(monkeypatch):
    import hornlab_bempp_bem.bie as bie

    class StopAfterMeshValidation(Exception):
        pass

    grid = SimpleNamespace(number_of_elements=1)

    def fail_if_rechecked(*_args, **_kwargs):
        raise AssertionError("closed mesh was revalidated")

    def stop_before_assembly(_grid):
        raise StopAfterMeshValidation

    monkeypatch.setattr(bie, "_require_closed_surface", fail_if_rechecked)
    monkeypatch.setattr(bie, "_setup_function_spaces", stop_before_assembly)

    with pytest.raises(StopAfterMeshValidation):
        solve_single_frequency(
            grid,
            np.array([2], dtype=np.int32),
            1000.0,
            SolveConfig(require_closed_mesh=True),
            closed_mesh_validated=True,
        )
