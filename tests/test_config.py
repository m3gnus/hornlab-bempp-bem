"""Unit tests for hornlab_bempp_bem.config — pure dataclass tests, no bempp needed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_bempp_bem.backends import (
    AssemblyBackendUnavailable,
    resolve_assembly_backend,
    resolve_fastest_backend,
)
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
    assert cfg.sphere_grid is None
    assert cfg.sphere_theta_max_deg == 180.0


@pytest.mark.parametrize("grid", [(1, 8), (5, 2), (2.5, 8), (500, 500), 5])
def test_observation_config_rejects_invalid_sphere_grid(grid):
    with pytest.raises(ValueError, match="sphere_grid"):
        ObservationConfig(sphere_grid=grid)


def test_observation_config_normalizes_integral_sphere_grid():
    config = ObservationConfig(sphere_grid=(5.0, 8.0))

    assert config.sphere_grid == (5, 8)


@pytest.mark.parametrize("distance_m", [np.nan, 0.0, -1.0])
def test_observation_config_rejects_invalid_distance(distance_m):
    with pytest.raises(ValueError, match="distance_m"):
        ObservationConfig(distance_m=distance_m)


def test_observation_config_rejects_unknown_origin():
    with pytest.raises(ValueError, match="origin"):
        ObservationConfig(origin="speaker")  # type: ignore[arg-type]


def test_observation_config_rejects_zero_angle_count():
    with pytest.raises(ValueError, match="angle_count"):
        ObservationConfig(angle_count=0)


@pytest.mark.parametrize("field", ["angle_min_deg", "angle_max_deg"])
def test_observation_config_rejects_nonfinite_angle_bounds(field):
    with pytest.raises(ValueError, match=field):
        ObservationConfig(**{field: np.nan})


@pytest.mark.parametrize(
    "planes",
    [
        [],
        "horizontal",
        ["horizontal", 5],
        ["horizontal", "horizontal"],
        ["foobar"],
    ],
)
def test_observation_config_rejects_malformed_planes(planes):
    with pytest.raises(ValueError, match="planes"):
        ObservationConfig(planes=planes)  # type: ignore[arg-type]


def test_observation_config_allows_no_planes_with_sphere_grid():
    config = ObservationConfig(planes=[], sphere_grid=(3, 4))

    assert config.planes == []


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


def test_solve_config_rejects_unknown_frequency_spacing():
    with pytest.raises(ValueError, match="freq_spacing"):
        SolveConfig(freq_spacing="quadratic")  # type: ignore[arg-type]


@pytest.mark.parametrize("freq_count", [0, -1, 1.5, True, "2"])
def test_solve_config_rejects_invalid_frequency_count(freq_count):
    with pytest.raises(ValueError, match="freq_count must be at least 1"):
        SolveConfig(freq_count=freq_count)  # type: ignore[arg-type]


def test_solve_config_normalizes_integral_frequency_count():
    config = SolveConfig(freq_count=2.0)  # type: ignore[arg-type]

    assert config.freq_count == 2
    assert isinstance(config.freq_count, int)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("freq_min_hz", 0.0),
        ("freq_min_hz", -1.0),
        ("freq_min_hz", np.nan),
        ("freq_max_hz", np.inf),
    ],
)
def test_solve_config_rejects_invalid_frequency_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        SolveConfig(**{field: value})


def test_solve_config_rejects_reversed_frequency_range():
    with pytest.raises(ValueError, match="freq_min_hz must not exceed"):
        SolveConfig(freq_min_hz=2000.0, freq_max_hz=1000.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slp_dlp_quadrature": 0},
        {"slp_dlp_quadrature": 2.5},
        {"slp_dlp_singular_quadrature": -2},
        {"hyp_adlp_quadrature": True},
    ],
)
def test_solve_config_rejects_invalid_quadrature_orders(kwargs):
    field = next(iter(kwargs))
    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        SolveConfig(**kwargs)


@pytest.mark.parametrize("order", [3, 5])
def test_numba_accepts_singular_orders_that_opencl_mishandles(order):
    config = SolveConfig(
        assembly_backend="numba",
        slp_dlp_singular_quadrature=order,
    )

    resolution = resolve_assembly_backend(config)

    assert resolution.effective_backend == "numba"


@pytest.mark.parametrize("order", [3, 5])
def test_opencl_rejects_singular_orders_that_drop_points(order):
    config = SolveConfig(
        assembly_backend="opencl",
        slp_dlp_singular_quadrature=order,
    )

    with pytest.raises(
        ValueError,
        match=rf"singular_quadrature orders 3 and 5 are unsupported.*use.*4",
    ):
        resolve_assembly_backend(config)


@pytest.mark.parametrize("order", [3, 5])
def test_auto_accepts_odd_singular_order_after_numba_fallback(monkeypatch, order):
    from hornlab_bempp_bem.device import OpenCLError

    def fail_probe(_device):
        raise OpenCLError("no usable OpenCL device")

    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        fail_probe,
    )
    config = SolveConfig(
        assembly_backend="auto",
        slp_dlp_singular_quadrature=order,
    )

    resolution = resolve_assembly_backend(config)

    assert resolution.effective_backend == "numba"
    assert resolution.fallback_used is True


def test_solve_config_normalizes_integral_quadrature_orders():
    config = SolveConfig(
        slp_dlp_quadrature=2.0,  # type: ignore[arg-type]
        slp_dlp_singular_quadrature=4.0,  # type: ignore[arg-type]
        hyp_adlp_quadrature=6.0,  # type: ignore[arg-type]
    )

    assert config.slp_dlp_quadrature == 2
    assert config.slp_dlp_singular_quadrature == 4
    assert config.hyp_adlp_quadrature == 6


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


def test_gmres_defaults_and_restart_validation():
    assert SolveConfig().gmres_tol == 1e-6
    assert SolveConfig(gmres_restart=100).gmres_restart == 100

    with pytest.raises(ValueError, match="gmres_restart"):
        SolveConfig(gmres_restart=0)
    with pytest.raises(ValueError, match="gmres_restart"):
        SolveConfig(gmres_restart=1.5)
    with pytest.raises(ValueError, match="gmres_tol"):
        SolveConfig(gmres_tol=0.0)


@pytest.mark.parametrize("lu_threshold", [0, -1, 1.5, True, "6000"])
def test_solve_config_rejects_invalid_lu_threshold(lu_threshold):
    with pytest.raises(ValueError, match="lu_threshold must be a positive integer"):
        SolveConfig(lu_threshold=lu_threshold)  # type: ignore[arg-type]


def test_solve_config_normalizes_integral_lu_threshold():
    config = SolveConfig(lu_threshold=100.0)  # type: ignore[arg-type]

    assert config.lu_threshold == 100
    assert isinstance(config.lu_threshold, int)


def test_auto_backend_probes_and_resolves_to_opencl(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda device: calls.append(device),
    )

    resolution = resolve_assembly_backend(SolveConfig(assembly_backend="auto"))

    assert resolution.effective_backend == "opencl"
    assert resolution.fallback_used is False
    assert calls == ["cpu"]


def test_auto_backend_falls_back_to_numba_when_opencl_probe_fails(monkeypatch):
    from hornlab_bempp_bem.device import OpenCLError

    def fail_probe(_device):
        raise OpenCLError("no usable OpenCL device")

    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        fail_probe,
    )

    resolution = resolve_assembly_backend(SolveConfig(assembly_backend="auto"))

    assert resolution.effective_backend == "numba"
    assert resolution.fallback_used is True
    assert resolution.reason == "no usable OpenCL device"


def test_auto_double_precision_falls_back_from_fp32_only_opencl(monkeypatch):
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda _device: "Mock FP32 GPU",
    )
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.opencl_devices_without_fp64",
        lambda _device: ("Mock FP32 GPU",),
    )

    resolution = resolve_assembly_backend(
        SolveConfig(assembly_backend="auto", precision="single"),
        required_precision="double",
    )

    assert resolution.effective_backend == "numba"
    assert resolution.fallback_used is True
    assert "Mock FP32 GPU" in (resolution.reason or "")
    assert "fp64" in (resolution.reason or "")


def test_explicit_opencl_refuses_double_on_fp32_only_device(monkeypatch):
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda _device: "Mock FP32 Accelerator",
    )
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.opencl_devices_without_fp64",
        lambda _device: ("Mock FP32 Accelerator",),
    )

    with pytest.raises(
        AssemblyBackendUnavailable,
        match="Mock FP32 Accelerator.*fp64.*Numba",
    ):
        resolve_assembly_backend(
            SolveConfig(assembly_backend="opencl", precision="single"),
            required_precision="double",
        )


def test_auto_double_precision_keeps_fp64_opencl(monkeypatch):
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda _device: "Mock FP64 GPU",
    )
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.opencl_devices_without_fp64",
        lambda _device: (),
    )

    resolution = resolve_assembly_backend(
        SolveConfig(assembly_backend="auto", precision="single"),
        required_precision="double",
    )

    assert resolution.effective_backend == "opencl"
    assert resolution.fallback_used is False


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (SimpleNamespace(double_fp_config=1, extensions=""), True),
        (SimpleNamespace(double_fp_config=0, extensions="cl_khr_fp64"), True),
        (SimpleNamespace(double_fp_config=0, extensions="cl_amd_fp64"), True),
        (SimpleNamespace(double_fp_config=0, extensions=""), False),
    ],
)
def test_opencl_fp64_capability_uses_device_features(device, expected):
    from hornlab_bempp_bem.device import _opencl_device_supports_fp64

    assert _opencl_device_supports_fp64(device) is expected


def test_explicit_solve_backend_does_not_probe_opencl(monkeypatch):
    def fail_if_called(_device):
        raise AssertionError("explicit backend resolution must not probe")

    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        fail_if_called,
    )

    resolution = resolve_assembly_backend(SolveConfig(assembly_backend="opencl"))

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


def test_fastest_backend_prefers_opencl_and_honours_an_explicit_name(monkeypatch):
    # This is a resolution-policy test, independent of whether the CI host has
    # an actual OpenCL ICD/device installed.
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda _device: SimpleNamespace(),
    )
    resolution = resolve_fastest_backend("auto")
    assert resolution.effective_backend == "opencl"
    assert resolution.fallback_used is False

    # Naming a backend outright is taken as given, including the slow one.
    assert resolve_fastest_backend("numba").effective_backend == "numba"
    assert resolve_fastest_backend("opencl").effective_backend == "opencl"
    with pytest.raises(ValueError, match="assembly_backend"):
        resolve_fastest_backend("cuda")


def test_fastest_backend_falls_back_to_numba_without_an_opencl_device(monkeypatch):
    """Field evaluation must still run where the solve's hard failure would not."""

    import sys

    from hornlab_bempp_bem.device import configure_opencl

    configure_opencl.cache_clear()
    monkeypatch.setitem(sys.modules, "pyopencl", None)
    resolution = resolve_fastest_backend("auto")
    configure_opencl.cache_clear()

    assert resolution.effective_backend == "numba"
    assert resolution.fallback_used is True
    assert "PyOpenCL" in (resolution.reason or "")


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


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        (1000.0, "one-dimensional"),
        ([[500.0, 1000.0]], "one-dimensional"),
        ([500.0, np.nan], "finite"),
        ([500.0, np.inf], "finite"),
        ([500.0, 0.0], "positive"),
        ([500.0, -1000.0], "positive"),
    ],
)
def test_solve_frequencies_rejects_invalid_values_before_loading_mesh(
    frequencies, message
):
    import hornlab_bempp_bem

    with pytest.raises(ValueError, match=message):
        hornlab_bempp_bem.solve_frequencies(object(), frequencies)
