from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hornlab_bempp_bem
from hornlab_bempp_bem.bie import solve_single_frequency
from hornlab_bempp_bem.config import BIEFormulation, SolveConfig
from hornlab_bempp_bem.mesh import MeshError
from hornlab_bempp_bem.symmetry import assemble_and_solve_symmetry


def test_solve_passes_native_symmetry_to_mesh_load(monkeypatch):
    calls = []

    class StopAfterMeshLoad(Exception):
        pass

    def fake_load_mesh(
        mesh,
        *,
        scale=1.0,
        require_closed=False,
        native_symmetry_plane=None,
    ):
        calls.append(
            (mesh, scale, require_closed, native_symmetry_plane),
        )
        raise StopAfterMeshLoad

    monkeypatch.setattr(hornlab_bempp_bem, "load_mesh", fake_load_mesh)

    with pytest.raises(StopAfterMeshLoad):
        hornlab_bempp_bem.solve(
            "quarter.msh",
            SolveConfig(native_symmetry_plane="yz+xz"),
        )

    assert calls == [("quarter.msh", 1.0, False, "yz+xz")]


def test_solve_single_frequency_builds_native_symmetry_context(monkeypatch):
    import hornlab_bempp_bem.symmetry as symmetry

    class StopAfterContextBuild(Exception):
        pass

    grid = SimpleNamespace(number_of_elements=1)
    calls = []

    def stop_after_context(grid_arg, tags_arg, plane_arg, *, ground_plane=None):
        calls.append((grid_arg, tags_arg.copy(), plane_arg, ground_plane))
        raise StopAfterContextBuild

    monkeypatch.setattr(
        symmetry, "build_symmetry_context", stop_after_context,
    )

    with pytest.raises(StopAfterContextBuild):
        solve_single_frequency(
            grid,
            np.array([2], dtype=np.int32),
            1000.0,
            SolveConfig(native_symmetry_plane="yz"),
        )
    assert calls[0][0] is grid
    np.testing.assert_array_equal(calls[0][1], [2])
    assert calls[0][2] == "yz"
    assert calls[0][3] is None

    # A ground plane reaches the same context builder, carried separately so
    # the expansion can tell a cut plane from an infinite rigid boundary.
    calls.clear()
    with pytest.raises(StopAfterContextBuild):
        solve_single_frequency(
            grid,
            np.array([2], dtype=np.int32),
            1000.0,
            SolveConfig(ground_plane="xy"),
        )
    assert calls[0][2] is None
    assert calls[0][3] == "xy"


def test_native_symmetry_rejects_robin_before_context_build():
    grid = SimpleNamespace(number_of_elements=1)
    with pytest.raises(NotImplementedError, match="Robin impedance"):
        solve_single_frequency(
            grid,
            np.array([2], dtype=np.int32),
            1000.0,
            SolveConfig(
                native_symmetry_plane="yz",
                impedance_sources={1: 0.1},
            ),
        )


def test_native_symmetry_rejects_burton_miller_explicitly():
    with pytest.raises(NotImplementedError, match="STANDARD and COMPLEX_K"):
        assemble_and_solve_symmetry(
            SimpleNamespace(),
            None,
            1.0,
            SolveConfig(formulation=BIEFormulation.BURTON_MILLER),
            {},
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
