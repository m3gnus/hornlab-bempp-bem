"""Streaming releases solve objects without changing rows or diagnostics."""

from __future__ import annotations

import sys
import weakref
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hornlab_bempp_bem import sweep, symmetry
from hornlab_bempp_bem.config import ObservationConfig, SolveConfig


class _Trace:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.space = None


class SweepHarness:
    """Run the real sweep/evaluation loop with small deterministic fake traces."""

    def __init__(self, *, expanded=False, dofs=64):
        self.expanded = expanded
        self.dofs = dofs
        self.trace_refs = []
        self.array_refs = []
        self.live_at_callback = []
        self.live_before_solve = []
        self.live_bytes_at_callback = []
        self.callback_rows = []
        self.progress = []

    def _trace(self, count, value):
        array = np.full(count, value, dtype=np.complex128)
        trace = _Trace(array)
        self.trace_refs.append(weakref.ref(trace))
        self.array_refs.append(weakref.ref(array))
        return trace

    def solve(self, grid, tags, frequency, config, **kwargs):
        self.live_before_solve.append(sum(ref() is not None for ref in self.trace_refs))
        pressure = self._trace(self.dofs, frequency + 1j)
        neumann = self._trace(self.dofs, 2j * frequency)
        return sweep.FrequencyResult(
            frequency,
            pressure,
            neumann,
            frequency + 3j,
            7,
            True,
            0.25,
            phase_timings={"linear_solve_s": 0.125},
            requested_backend="numba",
            effective_backend="numba",
            requested_precision="single",
            effective_precision="double",
            field_pressure_on_surface=(
                self._trace(2 * self.dofs, 3 * frequency + 1j)
                if self.expanded
                else None
            ),
            field_neumann_data=(
                self._trace(2 * self.dofs, 4j * frequency) if self.expanded else None
            ),
            regular_quadrature={"order": 4},
        )

    @staticmethod
    def far_field(p1, dp0, pressure, neumann, k, points, kwargs, restrict):
        value = pressure.coefficients[0] + neumann.coefficients[0] + k
        return value * np.arange(1, len(points) + 1, dtype=float)

    def run(
        self,
        *,
        module=sweep,
        stream=True,
        retain=False,
        stop_after=None,
        frequencies=None,
        mutate_callback=False,
        worker=False,
    ):
        frequencies = (
            np.array([100.0, 300.0, 200.0, 400.0])
            if frequencies is None
            else frequencies
        )

        def callback(index, frequency, entry):
            self.live_at_callback.append(
                sum(ref() is not None for ref in self.trace_refs)
            )
            self.live_bytes_at_callback.append(
                sum(
                    array.nbytes
                    for ref in self.array_refs
                    if (array := ref()) is not None
                )
            )
            self.callback_rows.append((index, frequency, deepcopy(entry)))
            if mutate_callback:
                entry["iterations"] = -1
                entry["phase_timings"].clear()
            return stop_after is None or index + 1 < stop_after

        config = SolveConfig(
            assembly_backend="numba",
            return_surface_traces=retain,
            native_symmetry_plane="yz" if self.expanded else None,
            observation=ObservationConfig(angle_count=3, sphere_grid=(3, 4)),
            on_frequency_result=callback if stream else None,
            progress_callback=lambda *args: self.progress.append(args),
        )
        mesh = SimpleNamespace(grid=None, physical_tags=np.array([2]), info=None)
        frame = SimpleNamespace(axis=np.array([0.0, 0.0, 1.0]))
        with ExitStack() as stack:
            stack.enter_context(
                patch.multiple(
                    module,
                    solve_single_frequency=self.solve,
                    _setup_function_spaces=lambda grid: (None, None),
                    compute_surface_pressure_avg=lambda grid, pressure, *args: {
                        2: complex(pressure.coefficients[0]),
                    },
                    build_observation_points=lambda *args: (
                        np.zeros((2, 3, 3)),
                        np.array([-90.0, 0.0, 90.0]),
                    ),
                    build_sphere_grid_points=lambda *args: (
                        np.zeros((12, 3)),
                        np.repeat([0.0, 90.0, 180.0], 4),
                        np.tile([0.0, 90.0, 180.0, 270.0], 3),
                    ),
                    _evaluate_far_field=self.far_field,
                )
            )
            stack.enter_context(
                patch.object(
                    symmetry,
                    "build_symmetry_context",
                    lambda *args, **kwargs: SimpleNamespace(
                        reduced_p1=None,
                        reduced_dp0=None,
                    ),
                )
            )
            if worker:
                api = SimpleNamespace(Grid=lambda *args: None)
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "bempp_cl": SimpleNamespace(api=api),
                            "bempp_cl.api": api,
                        },
                    )
                )
                return module._worker_solve_chunk_inner(
                    None,
                    None,
                    mesh.physical_tags,
                    frequencies,
                    np.zeros((2, 3, 3)),
                    np.array([-90.0, 0.0, 90.0]),
                    config,
                    sphere_points=np.zeros((12, 3)),
                    source_axis=frame.axis,
                )
            return module.run_sweep_serial(
                mesh,
                frequencies,
                frame,
                config,
                mesh_contracts_validated=True,
            )


RESULT_ARRAYS = (
    "frequencies_hz",
    "pressure_complex",
    "spl_db",
    "impedance",
    "observation_angles_deg",
    "observation_points",
    "sphere_pressure_complex",
    "sphere_theta_deg",
    "sphere_phi_deg",
    "surface_pressure_complex",
    "surface_neumann_complex",
)


def assert_identical_results(actual, expected):
    for name in RESULT_ARRAYS:
        left, right = getattr(actual, name), getattr(expected, name)
        if left is None or right is None:
            assert left is right, name
        else:
            assert left.shape == right.shape, name
            assert left.dtype == right.dtype, name
            assert left.tobytes() == right.tobytes(), name
    assert actual.observation_planes == expected.observation_planes
    assert actual.solver_log == expected.solver_log
    assert actual.surface_pressure_avg.keys() == expected.surface_pressure_avg.keys()
    for tag, values in actual.surface_pressure_avg.items():
        assert values.tobytes() == expected.surface_pressure_avg[tag].tobytes()


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("stop_after", [None, 2])
def test_streaming_releases_grid_functions_between_frequencies(
    expanded, retain, stop_after
):
    harness = SweepHarness(expanded=expanded)
    result = harness.run(retain=retain, stop_after=stop_after)
    per_frequency = 4 if expanded else 2
    solved = stop_after or 4
    assert harness.live_at_callback == [per_frequency] * solved
    assert harness.live_before_solve == [0] * solved
    assert all(ref() is None for ref in harness.trace_refs)
    if not retain:
        assert all(ref() is None for ref in harness.array_refs)
        assert result.surface_pressure_complex is None
        assert result.surface_neumann_complex is None


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("stop_after", [None, 2])
def test_streamed_results_are_byte_identical_to_deferred(expanded, retain, stop_after):
    streamed, deferred = (
        SweepHarness(expanded=expanded),
        SweepHarness(expanded=expanded),
    )
    frequencies = np.array([100.0, 300.0, 200.0, 400.0])
    result = streamed.run(retain=retain, stop_after=stop_after, frequencies=frequencies)
    reference = deferred.run(
        stream=False,
        retain=retain,
        frequencies=frequencies[:stop_after],
    )
    assert_identical_results(result, reference)
    assert streamed.progress == [
        (i, len(frequencies), float(freq))
        for i, freq in enumerate(frequencies[:stop_after])
    ]
    for i, freq, entry in streamed.callback_rows:
        assert freq == frequencies[i]
        assert entry["observation_spl_db"].tobytes() == result.spl_db[i].tobytes()
        assert (
            entry["observation_sphere_pressure_complex"].tobytes()
            == result.sphere_pressure_complex[i].tobytes()
        )


def test_callback_diagnostic_mutations_do_not_change_final_log():
    result = SweepHarness().run(mutate_callback=True)
    assert all(entry["iterations"] == 7 for entry in result.solver_log)
    assert all(
        entry["phase_timings"] == {"linear_solve_s": 0.125}
        for entry in result.solver_log
    )


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("retain", [False, True])
def test_worker_and_serial_keep_identical_rows_and_solve_space_traces(expanded, retain):
    worker = SweepHarness(expanded=expanded).run(worker=True, retain=retain)
    serial = SweepHarness(expanded=expanded).run(stream=False, retain=retain)
    assert len(worker) == (8 if retain else 6)
    assert worker[3] == serial.solver_log
    assert worker[4][2].tobytes() == serial.surface_pressure_avg[2].tobytes()
    arrays = [
        (0, "pressure_complex"),
        (1, "spl_db"),
        (2, "impedance"),
        (5, "sphere_pressure_complex"),
    ]
    if retain:
        arrays.extend([(6, "surface_pressure_complex"), (7, "surface_neumann_complex")])
    for index, name in arrays:
        actual, expected = worker[index], getattr(serial, name)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
