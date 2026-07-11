"""Unit tests for hornlab_bempp_bem.config — pure dataclass tests, no bempp needed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_bempp_bem.backends import resolve_assembly_backend
from hornlab_bempp_bem.config import (
    BIEFormulation,
    LinearSolver,
    ObservationConfig,
    SolveConfig,
    VelocityMode,
)


def test_observation_config_custom_points_defaults_none():
    cfg = ObservationConfig()
    assert cfg.custom_points is None


def test_solve_config_frame_override_defaults_none():
    cfg = SolveConfig()
    assert cfg.frame_override is None


def test_solve_config_air_density_default():
    cfg = SolveConfig()
    assert cfg.air_density == 1.2041


def test_solve_config_air_density_custom():
    cfg = SolveConfig(air_density=1.18)
    assert cfg.air_density == 1.18


def test_solve_config_progress_callback_defaults_none():
    cfg = SolveConfig()
    assert cfg.progress_callback is None


def test_solve_config_on_frequency_result_defaults_none():
    cfg = SolveConfig()
    assert cfg.on_frequency_result is None


def test_solve_config_default_backend_stays_opencl_cpu():
    cfg = SolveConfig()
    assert cfg.assembly_backend == "opencl"
    assert cfg.opencl_device == "cpu"
    assert cfg.native_symmetry_plane is None


def test_solve_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="assembly_backend"):
        SolveConfig(assembly_backend="cuda")  # type: ignore[arg-type]


def test_solve_config_rejects_metal_backend():
    with pytest.raises(ValueError, match="assembly_backend"):
        SolveConfig(assembly_backend="metal")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("formulation", "standard", BIEFormulation.STANDARD),
        ("formulation", "complex_k", BIEFormulation.COMPLEX_K),
        ("formulation", "burton_miller", BIEFormulation.BURTON_MILLER),
        ("solver", "auto", LinearSolver.AUTO),
        ("solver", "lu", LinearSolver.LU),
        ("solver", "gmres", LinearSolver.GMRES),
        ("velocity_mode", "velocity", VelocityMode.VELOCITY),
        ("velocity_mode", "acceleration", VelocityMode.ACCELERATION),
    ],
)
def test_solve_config_coerces_string_enum_values(field, value, expected):
    config = SolveConfig(**{field: value})

    assert getattr(config, field) is expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"formulation": "combined"},
            "formulation must be 'standard', 'complex_k', or 'burton_miller'",
        ),
        ({"solver": "dense"}, "solver must be 'auto', 'lu', or 'gmres'"),
        (
            {"velocity_mode": "displacement"},
            "velocity_mode must be 'velocity' or 'acceleration'",
        ),
        ({"precision": "half"}, "precision must be 'single' or 'double'"),
    ],
)
def test_solve_config_rejects_invalid_solve_modes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SolveConfig(**kwargs)


@pytest.mark.parametrize("profile", ["dome", "ring"])
def test_solve_config_rejects_unimplemented_velocity_profiles(profile):
    with pytest.raises(
        NotImplementedError,
        match=r"only supports velocity_profile='piston'",
    ):
        SolveConfig(velocity_profile=profile)


def test_solve_config_accepts_compatibility_piston_profile():
    assert SolveConfig(velocity_profile="piston").velocity_profile == "piston"


def test_solve_config_rejects_unknown_native_symmetry_plane():
    with pytest.raises(ValueError, match="native_symmetry_plane"):
        SolveConfig(native_symmetry_plane="zx")  # type: ignore[arg-type]


def test_solve_config_accepts_native_symmetry_planes():
    assert SolveConfig(native_symmetry_plane="yz").native_symmetry_plane == "yz"
    assert SolveConfig(native_symmetry_plane="xz").native_symmetry_plane == "xz"
    assert SolveConfig(native_symmetry_plane="xy").native_symmetry_plane == "xy"
    assert SolveConfig(native_symmetry_plane="yz+xz").native_symmetry_plane == "yz+xz"


def test_auto_backend_resolves_to_opencl():
    resolution = resolve_assembly_backend(SolveConfig(assembly_backend="auto"))
    assert resolution.effective_backend == "opencl"
    assert resolution.fallback_used is False


def test_opencl_backend_reports_optional_dependency_install(monkeypatch):
    import sys

    from hornlab_bempp_bem.device import OpenCLError, configure_opencl

    configure_opencl.cache_clear()
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    with pytest.raises(OpenCLError, match=r"hornlab-bempp-bem\[opencl\]"):
        configure_opencl("cpu")
    configure_opencl.cache_clear()


def test_solve_config_callbacks_accept_callables():
    calls = []
    cfg = SolveConfig(
        progress_callback=lambda i, n, f: calls.append(("progress", i)),
        on_frequency_result=lambda i, f, log: True,
    )
    cfg.progress_callback(0, 5, 1000.0)
    assert calls == [("progress", 0)]
    assert cfg.on_frequency_result(0, 1000.0, {}) is True


def test_require_closed_mesh_defaults_off_and_forwards():
    """Closed-mode callers set require_closed_mesh; it must reach load_mesh."""
    import inspect

    from hornlab_bempp_bem import _resolve_mesh
    from hornlab_bempp_bem.config import SolveConfig
    from hornlab_bempp_bem.mesh import load_mesh

    assert SolveConfig().require_closed_mesh is False
    # The loader accepts the flag and the resolver forwards it.
    assert "require_closed" in inspect.signature(load_mesh).parameters
    assert "require_closed" in inspect.signature(_resolve_mesh).parameters


@pytest.mark.parametrize("entrypoint", ["solve", "solve_frequencies"])
def test_public_solve_rejects_velocity_source_tags_missing_from_mesh(entrypoint):
    import hornlab_bempp_bem
    from hornlab_bempp_bem.mesh import LoadedMesh

    loaded = LoadedMesh(
        grid=SimpleNamespace(),
        physical_tags=np.array([2, 1, 2, 1], dtype=np.int32),
        info=SimpleNamespace(),
    )
    config = SolveConfig(velocity_sources={4: 1.0, 3: 0.5})

    with pytest.raises(
        ValueError,
        match=(
            r"velocity_sources tags \[3, 4\] are not present in the mesh; "
            r"available physical tags: \[1, 2\]"
        ),
    ):
        if entrypoint == "solve":
            hornlab_bempp_bem.solve(loaded, config)
        else:
            hornlab_bempp_bem.solve_frequencies(loaded, [1000.0], config)
