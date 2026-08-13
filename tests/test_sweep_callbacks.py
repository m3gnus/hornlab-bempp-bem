"""Unit tests for sweep.py — on-axis normalisation, early stopping, callbacks.

These tests mock out bempp-cl internals to test the sweep logic in isolation.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hornlab_bempp_bem.config import ObservationConfig, SolveConfig, SourceMotion
from hornlab_bempp_bem.observation import ObservationFrame
from hornlab_bempp_bem.result import MeshInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame() -> ObservationFrame:
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.array([0.0, 0.0, 1.0]),
        source_center=np.array([0.0, 0.0, 0.0]),
    )


def _make_mesh():
    mesh = MagicMock()
    mesh.info = MeshInfo(
        n_vertices=100, n_triangles=200,
        physical_groups={1: "body", 2: "source"},
        bounding_box_m=(np.zeros(3), np.ones(3)),
    )
    mesh.grid = MagicMock()
    mesh.physical_tags = np.array([1] * 180 + [2] * 20, dtype=np.int32)
    return mesh


def _fake_frequency_result(freq_hz: float):
    fr = MagicMock()
    fr.frequency_hz = freq_hz
    fr.iterations = 10
    fr.converged = True
    fr.timing_s = 0.5
    fr.phase_timings = {"total_s": 0.5, "linear_solve_s": 0.1}
    fr.impedance = 400.0 + 50j
    fr.pressure_on_surface = MagicMock()
    fr.pressure_on_surface.space = MagicMock()
    fr.neumann_data = MagicMock()
    fr.neumann_data.space = MagicMock()
    return fr


# Patch targets within hornlab_bempp_bem.sweep
_SWEEP = "hornlab_bempp_bem.sweep"


def _standard_sweep_patches():
    """Return a dict of patches that isolate run_sweep_serial from bempp."""
    return {
        "spaces": patch(f"{_SWEEP}._setup_function_spaces",
                        return_value=(MagicMock(), MagicMock())),
        "solve": patch(f"{_SWEEP}.solve_single_frequency",
                       side_effect=lambda *a, **kw: _fake_frequency_result(
                           a[2] if len(a) > 2 else 1000.0)),
        "pavg": patch(f"{_SWEEP}.compute_surface_pressure_avg",
                      return_value={2: 1.0 + 0.1j}),
        "ff": patch(f"{_SWEEP}._evaluate_far_field",
                    return_value=np.ones(5, dtype=np.complex128)),
        "op_kw": patch(f"{_SWEEP}._operator_kwargs", return_value={}),
        "dir": patch(f"{_SWEEP}._evaluate_directivity",
                     side_effect=lambda fr, obs, ang, cfg, sphere=None: (
                         np.ones((len(fr), obs.shape[0], obs.shape[1]),
                                 dtype=np.complex128),
                         np.zeros((len(fr), obs.shape[0], obs.shape[1]),
                                  dtype=np.float64),
                         None,
                     )),
    }


# ---------------------------------------------------------------------------
# On-axis index at non-zero angle_min
# ---------------------------------------------------------------------------

class TestOnAxisIndex:

    def test_spl_normalisation_handles_silent_points_without_log_warning(self):
        from hornlab_bempp_bem._constants import REFERENCE_PRESSURE
        from hornlab_bempp_bem.sweep import _normalized_spl_db

        pressure = np.array(
            [0.0, REFERENCE_PRESSURE, 10.0 * REFERENCE_PRESSURE],
            dtype=np.complex128,
        )
        with np.errstate(divide="raise", invalid="raise"):
            spl = _normalized_spl_db(pressure, on_axis_idx=1)

        np.testing.assert_allclose(spl, [-120.0, 0.0, 20.0])

    def test_spl_normalisation_handles_multiple_planes(self):
        from hornlab_bempp_bem.sweep import _normalized_spl_db

        pressure = np.array([
            [1.0, 2.0, 4.0],
            [8.0, 4.0, 2.0],
        ], dtype=np.complex128)

        spl = _normalized_spl_db(pressure, on_axis_idx=1)

        np.testing.assert_allclose(
            spl,
            [
                [-20.0 * np.log10(2.0), 0.0, 20.0 * np.log10(2.0)],
                [20.0 * np.log10(2.0), 0.0, -20.0 * np.log10(2.0)],
            ],
        )

    def test_on_axis_at_nonzero_angle_min(self):
        angles = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert on_axis_idx == 2

    def test_on_axis_offset_from_zero(self):
        angles = np.linspace(5.0, 180.0, 36)
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert on_axis_idx == 0

    def test_on_axis_negative_start(self):
        angles = np.linspace(-180.0, 180.0, 73)
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert angles[on_axis_idx] == 0.0

    def test_spl_normalisation_uses_on_axis_idx(self):
        from hornlab_bempp_bem._constants import REFERENCE_PRESSURE

        angles = np.linspace(-90.0, 90.0, 37)
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert on_axis_idx == 18

        amplitudes = np.linspace(0.5, 1.0, 37)
        amplitudes[on_axis_idx] = 2.0
        spl_raw = 20.0 * np.log10(amplitudes / REFERENCE_PRESSURE)
        spl_norm = spl_raw - spl_raw[on_axis_idx]

        assert spl_norm[on_axis_idx] == 0.0
        assert np.all(spl_norm[:on_axis_idx] < 0.0)
        assert np.all(spl_norm[on_axis_idx + 1:] < 0.0)


# ---------------------------------------------------------------------------
# Batched observation-plane evaluation
# ---------------------------------------------------------------------------

class TestBatchedObservationEvaluation:

    def test_all_planes_share_one_far_field_evaluation(self):
        from hornlab_bempp_bem.sweep import _evaluate_observation_planes

        obs_points = np.arange(3 * 4 * 3, dtype=np.float64).reshape(3, 4, 3)
        flat_pressure = np.arange(1, 13, dtype=np.float64).astype(np.complex128)

        with patch(
            f"{_SWEEP}._evaluate_far_field",
            return_value=flat_pressure,
        ) as evaluate_far_field:
            pressure, spl, sphere_pressure = _evaluate_observation_planes(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                2.5,
                obs_points,
                {},
                on_axis_idx=1,
            )

        assert evaluate_far_field.call_count == 1
        np.testing.assert_array_equal(
            evaluate_far_field.call_args.args[5],
            obs_points.reshape(-1, 3),
        )
        np.testing.assert_array_equal(pressure, flat_pressure.reshape(3, 4))
        np.testing.assert_allclose(spl[:, 1], 0.0)
        assert sphere_pressure is None

    def test_sphere_and_display_planes_share_one_far_field_evaluation(self):
        from hornlab_bempp_bem.sweep import _evaluate_observation_planes

        obs_points = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
        sphere_points = 100.0 + np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
        evaluated = np.arange(1, 11, dtype=np.float64).astype(np.complex128)

        with patch(
            f"{_SWEEP}._evaluate_far_field",
            return_value=evaluated,
        ) as evaluate_far_field:
            pressure, _spl, sphere_pressure = _evaluate_observation_planes(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                2.5,
                obs_points,
                {},
                on_axis_idx=0,
                sphere_points=sphere_points,
            )

        expected_points = np.concatenate([obs_points.reshape(-1, 3), sphere_points])
        np.testing.assert_array_equal(evaluate_far_field.call_args.args[5], expected_points)
        np.testing.assert_array_equal(pressure, evaluated[:6].reshape(2, 3))
        np.testing.assert_array_equal(sphere_pressure, evaluated[6:])


# ---------------------------------------------------------------------------
# Early stopping via on_frequency_result
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_early_stop_after_two_frequencies(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        stop_after = 2
        call_count = [0]
        callback_logs = []

        def stopper(freq_idx, freq_hz, log_entry):
            call_count[0] += 1
            callback_logs.append(log_entry)
            return freq_idx < stop_after - 1

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            on_frequency_result=stopper,
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            result = run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0, 2000.0, 4000.0]),
                _make_frame(), config,
            )

        assert len(result.frequencies_hz) == 2
        assert call_count[0] == 2
        assert all(log["converged"] is True for log in callback_logs)
        assert all(log["converged"] is True for log in result.solver_log)
        np.testing.assert_allclose(result.frequencies_hz, [500.0, 1000.0])

    def test_no_early_stop_all_frequencies_solved(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            on_frequency_result=lambda i, f, log: True,
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"] as dir_mock:
            result = run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0, 2000.0]),
                _make_frame(), config,
            )

        assert len(result.frequencies_hz) == 3
        assert dir_mock.call_count == 0

    def test_side_effect_callback_returning_none_does_not_stop(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        seen = []

        def observer(i, frequency_hz, _log):
            seen.append((i, frequency_hz))

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            on_frequency_result=observer,
        )
        frequencies = np.array([500.0, 1000.0, 2000.0])

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            result = run_sweep_serial(
                _make_mesh(), frequencies, _make_frame(), config,
            )

        np.testing.assert_allclose(result.frequencies_hz, frequencies)
        assert seen == [(0, 500.0), (1, 1000.0), (2, 2000.0)]


def test_serial_result_publishes_spherical_pressure_and_coordinates():
    from hornlab_bempp_bem.sweep import run_sweep_serial

    patches = _standard_sweep_patches()
    frequencies = np.array([500.0, 1000.0])
    sphere_pressure = np.arange(24, dtype=np.float64).reshape(2, 12).astype(np.complex128)
    patches["dir"] = patch(
        f"{_SWEEP}._evaluate_directivity",
        return_value=(
            np.ones((2, 1, 5), dtype=np.complex128),
            np.zeros((2, 1, 5), dtype=np.float64),
            sphere_pressure,
        ),
    )
    config = SolveConfig(
        observation=ObservationConfig(
            planes=["horizontal"],
            angle_count=5,
            sphere_grid=(3, 4),
        ),
    )

    with patches["spaces"], patches["solve"], patches["pavg"], \
         patches["ff"], patches["op_kw"], patches["dir"]:
        result = run_sweep_serial(_make_mesh(), frequencies, _make_frame(), config)

    np.testing.assert_array_equal(result.sphere_pressure_complex, sphere_pressure)
    np.testing.assert_allclose(result.sphere_theta_deg, np.repeat([0.0, 90.0, 180.0], 4))
    np.testing.assert_allclose(result.sphere_phi_deg, np.tile([0.0, 90.0, 180.0, 270.0], 3))


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_progress_callback_called_per_frequency(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        progress_calls = []

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            progress_callback=lambda i, n, f: progress_calls.append((i, n, f)),
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            run_sweep_serial(
                _make_mesh(), np.array([100.0, 200.0, 300.0]),
                _make_frame(), config,
            )

        assert len(progress_calls) == 3
        assert progress_calls[0] == (0, 3, 100.0)
        assert progress_calls[1] == (1, 3, 200.0)
        assert progress_calls[2] == (2, 3, 300.0)


# ---------------------------------------------------------------------------
# Parallel mode rejects callbacks
# ---------------------------------------------------------------------------

class TestParallelRejectsCallbacks:

    def test_progress_callback_is_supported(self):
        """It stays in the parent and is fed by a queue the workers publish to.

        ``on_frequency_result`` stays rejected because a worker cannot cancel
        frequencies already running in its siblings.
        """
        from hornlab_bempp_bem.sweep import run_sweep_parallel

        seen = []
        config = SolveConfig(
            progress_callback=lambda i, n, f: seen.append((i, n, f)),
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        )
        frequencies = np.array([500.0, 1000.0])

        class ImmediateFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class InlineExecutor:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, fn, **kwargs):
                return ImmediateFuture(fn(**kwargs))

        def fake_worker(**kwargs):
            freqs = np.asarray(kwargs["frequencies"], dtype=np.float64)
            n_planes, n_angles, _ = kwargs["obs_points"].shape
            for offset, freq in enumerate(freqs):
                kwargs["progress_queue"].put_nowait(
                    (int(kwargs["global_indices"][offset]), float(freq)),
                )
            return (
                np.ones((len(freqs), n_planes, n_angles), dtype=np.complex128),
                np.zeros((len(freqs), n_planes, n_angles), dtype=np.float64),
                np.full(len(freqs), 400.0 + 50j, dtype=np.complex128),
                [{"frequency_hz": float(freq)} for freq in freqs],
                {2: freqs.astype(np.complex128)},
                None,
            )

        with patch(f"{_SWEEP}.ProcessPoolExecutor", InlineExecutor), \
             patch(f"{_SWEEP}.wait", side_effect=lambda fs, timeout=None: (set(fs), set())), \
             patch(f"{_SWEEP}._worker_solve_chunk", side_effect=fake_worker):
            run_sweep_parallel(
                _make_mesh(), frequencies, _make_frame(), config, 2,
            )

        assert [index for index, _, _ in seen] == [0, 1], "must be monotonic"
        assert {total for _, total, _ in seen} == {2}
        assert sorted(freq for _, _, freq in seen) == [500.0, 1000.0]

    def test_on_frequency_result_rejected(self):
        from hornlab_bempp_bem.sweep import run_sweep_parallel

        config = SolveConfig(on_frequency_result=lambda i, f, log: True)
        with pytest.raises(ValueError, match="not supported in parallel mode"):
            run_sweep_parallel(
                _make_mesh(), np.array([100.0]), _make_frame(), config, 2,
            )


# ---------------------------------------------------------------------------
# Empty sweeps
# ---------------------------------------------------------------------------

class TestEmptySweeps:

    def test_serial_rejects_empty_frequencies_before_setup(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        with patch(f"{_SWEEP}._setup_function_spaces") as setup_spaces, \
             pytest.raises(ValueError, match="frequencies must contain at least one value"):
            run_sweep_serial(
                _make_mesh(), np.array([]), _make_frame(), SolveConfig(),
            )

        setup_spaces.assert_not_called()

    def test_parallel_rejects_empty_frequencies_before_spawning(self):
        from hornlab_bempp_bem.sweep import run_sweep_parallel

        with patch(f"{_SWEEP}.ProcessPoolExecutor") as executor, \
             pytest.raises(ValueError, match="frequencies must contain at least one value"):
            run_sweep_parallel(
                _make_mesh(), np.array([]), _make_frame(), SolveConfig(), 2,
            )

        executor.assert_not_called()


# ---------------------------------------------------------------------------
# surface_pressure_avg populated in result
# ---------------------------------------------------------------------------

class TestSurfacePressureAvg:

    def test_serial_sweep_threads_frame_axis_to_axial_source(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        frame = _make_frame()
        config = SolveConfig(
            source_motion=SourceMotion.AXIAL,
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        )
        axial_element_scale = np.arange(
            len(_make_mesh().physical_tags), dtype=np.float64,
        )

        with patches["spaces"], patches["solve"] as solve_mock, patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"], \
             patch(
                 f"{_SWEEP}._build_axial_element_scale",
                 return_value=axial_element_scale,
             ) as build_scale:
            run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0]), frame, config,
            )

        build_scale.assert_called_once()
        assert solve_mock.call_count == 2
        for call in solve_mock.call_args_list:
            np.testing.assert_array_equal(call.kwargs["source_axis"], frame.axis)
            assert call.kwargs["axial_element_scale"] is axial_element_scale

    def test_serial_sweep_validates_closed_mesh_once(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        mesh = _make_mesh()
        mesh.grid.vertices = np.zeros((3, 4), dtype=np.float64)
        mesh.grid.elements = np.zeros((3, 2), dtype=np.int32)
        config = SolveConfig(
            require_closed_mesh=True,
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        )

        with patches["spaces"], patches["solve"] as solve_mock, patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"], \
             patch(f"{_SWEEP}._require_closed_surface") as require_closed:
            run_sweep_serial(
                mesh, np.array([500.0, 1000.0, 2000.0]), _make_frame(), config,
            )

        assert require_closed.call_count == 1
        assert solve_mock.call_count == 3
        assert all(
            call.kwargs["closed_mesh_validated"] is True
            for call in solve_mock.call_args_list
        )

    def test_surface_pressure_avg_in_result(self):
        from hornlab_bempp_bem.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        # Override pavg to return a known value
        patches["pavg"] = patch(
            f"{_SWEEP}.compute_surface_pressure_avg",
            return_value={2: 100.0 + 50j},
        )

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            result = run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0]),
                _make_frame(), config,
            )

        assert result.surface_pressure_avg is not None
        assert 2 in result.surface_pressure_avg
        assert len(result.surface_pressure_avg[2]) == 2
        np.testing.assert_allclose(
            result.surface_pressure_avg[2],
            [100.0 + 50j, 100.0 + 50j],
        )

    def test_parallel_worker_returns_surface_pressure_averages(self):
        from hornlab_bempp_bem.sweep import _worker_solve_chunk

        frequencies = np.array([500.0, 1000.0])
        source_axis = np.array([0.0, 1.0, 0.0])
        config = SolveConfig(
            assembly_backend="numba",
            source_motion=SourceMotion.AXIAL,
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        )
        axial_element_scale = np.array([0.25, 1.0])
        with patch("bempp_cl.api.Grid", return_value=MagicMock()), \
             patch(f"{_SWEEP}._setup_function_spaces", return_value=(MagicMock(), MagicMock())), \
             patch(
                 f"{_SWEEP}._build_axial_element_scale",
                 return_value=axial_element_scale,
             ) as build_scale, \
             patch(f"{_SWEEP}.solve_single_frequency", side_effect=lambda *a, **k: _fake_frequency_result(float(a[2]))) as solve_mock, \
             patch(f"{_SWEEP}.compute_surface_pressure_avg", return_value={2: 100.0 + 50j}), \
             patch(f"{_SWEEP}._evaluate_far_field", return_value=np.ones(5, dtype=np.complex128)), \
             patch(f"{_SWEEP}._operator_kwargs", return_value={}), \
             patch(f"{_SWEEP}.resolve_assembly_backend", return_value=SimpleNamespace(effective_backend="numba")):
            worker_result = _worker_solve_chunk(
                mesh_grid_verts=np.zeros((3, 4)),
                mesh_grid_elems=np.zeros((3, 2), dtype=np.int32),
                physical_tags=np.array([1, 2], dtype=np.int32),
                frequencies=frequencies,
                obs_points=np.zeros((1, 5, 3)),
                angles_deg=np.linspace(0.0, 180.0, 5),
                config=config,
                source_axis=source_axis,
            )

        surface_pressure = worker_result[4]
        np.testing.assert_allclose(
            surface_pressure[2], [100.0 + 50j, 100.0 + 50j]
        )
        build_scale.assert_called_once()
        assert all(
            call.kwargs["closed_mesh_validated"] is True
            for call in solve_mock.call_args_list
        )
        for call in solve_mock.call_args_list:
            np.testing.assert_array_equal(
                call.kwargs["source_axis"], source_axis,
            )
            assert call.kwargs["axial_element_scale"] is axial_element_scale

    def test_parallel_result_preserves_worker_surface_pressure_order(self):
        from hornlab_bempp_bem.sweep import run_sweep_parallel

        class ImmediateFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class InlineExecutor:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, fn, **kwargs):
                return ImmediateFuture(fn(**kwargs))

        def fake_worker(**kwargs):
            np.testing.assert_array_equal(kwargs["source_axis"], _make_frame().axis)
            freqs = np.asarray(kwargs["frequencies"], dtype=np.float64)
            n_planes, n_angles, _ = kwargs["obs_points"].shape
            sphere_points = kwargs["sphere_points"]
            return (
                np.ones((len(freqs), n_planes, n_angles), dtype=np.complex128),
                np.zeros((len(freqs), n_planes, n_angles), dtype=np.float64),
                np.full(len(freqs), 400.0 + 50j, dtype=np.complex128),
                [{"frequency_hz": float(freq)} for freq in freqs],
                {2: freqs.astype(np.complex128) + 10j},
                np.repeat(freqs[:, None], sphere_points.shape[0], axis=1).astype(
                    np.complex128
                ),
            )

        frequencies = np.array([500.0, 1000.0, 2000.0])
        config = SolveConfig(
            observation=ObservationConfig(
                planes=["horizontal"], angle_count=5, sphere_grid=(2, 3)
            )
        )
        with patch(f"{_SWEEP}.ProcessPoolExecutor", InlineExecutor), \
             patch(f"{_SWEEP}.wait", side_effect=lambda fs, timeout=None: (set(fs), set())), \
             patch(f"{_SWEEP}._worker_solve_chunk", side_effect=fake_worker):
            result = run_sweep_parallel(
                _make_mesh(), frequencies, _make_frame(), config, worker_count=2
            )

        assert result.surface_pressure_avg is not None
        np.testing.assert_allclose(
            result.surface_pressure_avg[2], frequencies + 10j
        )
        assert result.sphere_pressure_complex is not None
        np.testing.assert_allclose(
            result.sphere_pressure_complex,
            np.repeat(frequencies[:, None], 6, axis=1),
        )
        np.testing.assert_allclose(result.sphere_theta_deg, np.repeat([0.0, 180.0], 3))
        np.testing.assert_allclose(result.sphere_phi_deg, np.tile([0.0, 120.0, 240.0], 2))
