"""Frequency sweep — serial and parallel execution."""
from __future__ import annotations

import logging
import queue
import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
from .backends import resolve_assembly_backend
from .bie import (
    FrequencyResult,
    _build_axial_element_scale,
    _evaluate_far_field,
    _operator_kwargs,
    _setup_function_spaces,
    compute_surface_pressure_avg,
    solve_single_frequency,
)
from .config import SolveConfig, SourceMotion
from .mesh import (
    LoadedMesh,
    _require_closed_surface,
    _validate_velocity_source_tags,
)
from .observation import (
    ObservationFrame,
    build_observation_points,
    build_sphere_grid_points,
)
from .result import SolveResult

logger = logging.getLogger(__name__)


def _solver_log_entry(fr: FrequencyResult) -> dict:
    """Build the diagnostic fields shared by every sweep execution path."""
    return {
        "frequency_hz": fr.frequency_hz,
        "iterations": fr.iterations,
        "converged": fr.converged,
        "timing_s": fr.timing_s,
        "phase_timings": dict(fr.phase_timings),
    }


def _retained_surface_traces(
    freq_results: list[FrequencyResult],
    enabled: bool,
) -> tuple[
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
]:
    """Stack solve-space P1 pressure and total DP0 Neumann coefficients."""
    if not enabled:
        return None, None
    pressure = np.stack(
        [
            np.asarray(fr.pressure_on_surface.coefficients, dtype=np.complex128)
            for fr in freq_results
        ],
        axis=0,
    )
    neumann = np.stack(
        [
            np.asarray(fr.neumann_data.coefficients, dtype=np.complex128)
            for fr in freq_results
        ],
        axis=0,
    )
    return pressure, neumann


def _build_frequency_grid(config: SolveConfig) -> NDArray[np.float64]:
    if config.freq_spacing == "log":
        return np.geomspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)
    return np.linspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)


def _normalized_spl_db(
    pressure: NDArray[np.complex128],
    on_axis_idx: int,
) -> NDArray[np.float64]:
    """Convert pressure to SPL and normalize its last axis to zero on-axis."""
    amplitudes = np.abs(pressure)
    spl_raw = np.full(amplitudes.shape, -120.0, dtype=np.float64)
    audible = amplitudes > 1e-15
    spl_raw[audible] = 20.0 * np.log10(
        amplitudes[audible] / REFERENCE_PRESSURE
    )
    return spl_raw - spl_raw[..., on_axis_idx, None]


def _evaluate_observation_planes(
    p1_space,
    dp0_space,
    pressure_on_surface,
    neumann_data,
    k_real: float,
    obs_points: NDArray[np.float64],
    op_kwargs: dict,
    on_axis_idx: int,
    restrict_neumann_space: bool = True,
    sphere_points: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.complex128] | None,
]:
    """Evaluate display planes and an optional sphere in one potential operation."""
    n_planes, n_angles, _ = obs_points.shape
    arc_points = obs_points.reshape(n_planes * n_angles, 3)
    evaluation_points = (
        arc_points
        if sphere_points is None
        else np.concatenate([arc_points, sphere_points], axis=0)
    )
    evaluated = _evaluate_far_field(
        p1_space,
        dp0_space,
        pressure_on_surface,
        neumann_data,
        k_real,
        evaluation_points,
        op_kwargs,
        restrict_neumann_space,
    )
    arc_count = n_planes * n_angles
    pressure = evaluated[:arc_count].reshape(n_planes, n_angles)
    sphere_pressure = evaluated[arc_count:] if sphere_points is not None else None
    return pressure, _normalized_spl_db(pressure, on_axis_idx), sphere_pressure


def _evaluate_directivity(
    freq_results: list[FrequencyResult],
    obs_points: NDArray[np.float64],
    angles_deg: NDArray[np.float64],
    config: SolveConfig,
    sphere_points: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.complex128] | None,
]:
    """Evaluate far-field pressure at observation points for all frequencies.

    Returns:
        pressure_complex: (F, P, N_angles)
        spl_db: (F, P, N_angles) — normalised on-axis = 0 dB
    """
    n_freq = len(freq_results)
    n_planes, n_angles, _ = obs_points.shape

    pressure = np.zeros((n_freq, n_planes, n_angles), dtype=np.complex128)
    spl = np.full((n_freq, n_planes, n_angles), -120.0, dtype=np.float64)
    sphere_pressure = (
        np.zeros((n_freq, sphere_points.shape[0]), dtype=np.complex128)
        if sphere_points is not None
        else None
    )

    # On-axis index: the angle closest to 0 degrees
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs = _operator_kwargs(
        backend, config.precision, config.opencl_device,
        vectorization_mode=getattr(config, "vectorization_mode", "auto"),
    )

    for fi, fr in enumerate(freq_results):
        k_real = 2.0 * np.pi * fr.frequency_hz / SPEED_OF_SOUND

        field_pressure = (
            fr.field_pressure_on_surface
            if fr.field_pressure_on_surface is not None
            else fr.pressure_on_surface
        )
        field_neumann = (
            fr.field_neumann_data
            if fr.field_neumann_data is not None
            else fr.neumann_data
        )
        p1 = field_pressure.space
        dp0 = field_neumann.space

        arc_pressure, arc_spl, sphere_row = _evaluate_observation_planes(
            p1,
            dp0,
            field_pressure,
            field_neumann,
            k_real,
            obs_points,
            op_kwargs,
            on_axis_idx,
            config.restrict_neumann_space,
            sphere_points,
        )
        pressure[fi], spl[fi] = arc_pressure, arc_spl
        if sphere_pressure is not None and sphere_row is not None:
            sphere_pressure[fi] = sphere_row

    return pressure, spl, sphere_pressure


def run_sweep_serial(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
    *,
    mesh_contracts_validated: bool = False,
) -> SolveResult:
    """Run frequency sweep in a single process."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("frequencies must contain at least one value")

    t_total = time.time()

    if not mesh_contracts_validated:
        _validate_velocity_source_tags(
            mesh.physical_tags, config.velocity_sources,
        )
        if config.require_closed_mesh:
            _require_closed_surface(
                np.asarray(mesh.grid.vertices, dtype=np.float64).T,
                np.asarray(mesh.grid.elements, dtype=np.int32).T,
            )

    obs_points, angles_deg = build_observation_points(frame, config.observation)
    sphere_points = None
    sphere_theta_deg = None
    sphere_phi_deg = None
    if config.observation.sphere_grid is not None:
        sphere_points, sphere_theta_deg, sphere_phi_deg = build_sphere_grid_points(
            frame, config.observation,
        )

    symmetry_context = None
    if config.native_symmetry_plane is None:
        p1_space, dp0_space = _setup_function_spaces(mesh.grid)
    else:
        from .symmetry import build_symmetry_context

        symmetry_context = build_symmetry_context(
            mesh.grid,
            mesh.physical_tags,
            config.native_symmetry_plane,
        )
        if config.require_closed_mesh:
            _require_closed_surface(
                np.asarray(
                    symmetry_context.expanded_grid.vertices,
                    dtype=np.float64,
                ).T,
                np.asarray(
                    symmetry_context.expanded_grid.elements,
                    dtype=np.int32,
                ).T,
            )
        p1_space = symmetry_context.reduced_p1
        dp0_space = symmetry_context.reduced_dp0

    source_tags = list(config.velocity_sources.keys())
    axial_element_scale = None
    if config.source_motion == SourceMotion.AXIAL:
        axial_element_scale = _build_axial_element_scale(
            mesh.grid,
            mesh.physical_tags,
            source_tags,
            frame.axis,
        )
    freq_results: list[FrequencyResult] = []
    surface_pavg: dict[int, list[complex]] = {tag: [] for tag in source_tags}

    # Pre-compute op_kwargs for per-frequency far-field evaluation
    # (only used when on_frequency_result is set)
    has_callback = config.on_frequency_result is not None
    callback_pressure_rows: list[NDArray[np.complex128]] = []
    callback_spl_rows: list[NDArray[np.float64]] = []
    callback_sphere_rows: list[NDArray[np.complex128]] = []
    if has_callback:
        _backend = resolve_assembly_backend(config).effective_backend
        _ff_op_kwargs = _operator_kwargs(
            _backend, config.precision, config.opencl_device,
            vectorization_mode=getattr(config, "vectorization_mode", "auto"),
        )
        on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    for i, freq in enumerate(frequencies):
        logger.info("[%d/%d] %.1f Hz", i + 1, len(frequencies), freq)
        fr = solve_single_frequency(
            mesh.grid, mesh.physical_tags, freq, config,
            p1_space=p1_space,
            dp0_space=dp0_space,
            source_axis=frame.axis,
            axial_element_scale=axial_element_scale,
            closed_mesh_validated=True,
            symmetry_context=symmetry_context,
        )
        freq_results.append(fr)

        # Surface pressure average per source tag
        pavg = compute_surface_pressure_avg(
            mesh.grid, fr.pressure_on_surface,
            mesh.physical_tags, p1_space, source_tags,
        )
        for tag in source_tags:
            surface_pavg[tag].append(pavg[tag])

        log_entry = _solver_log_entry(fr)
        log_entry["impedance"] = fr.impedance

        # When the callback is set, evaluate per-frequency directivity
        # so the caller can act on partial results as they stream in.
        if has_callback:
            k_real = 2.0 * np.pi * fr.frequency_hz / SPEED_OF_SOUND
            field_pressure = (
                fr.field_pressure_on_surface
                if fr.field_pressure_on_surface is not None
                else fr.pressure_on_surface
            )
            field_neumann = (
                fr.field_neumann_data
                if fr.field_neumann_data is not None
                else fr.neumann_data
            )
            per_freq_pressure, per_freq_spl, per_freq_sphere = _evaluate_observation_planes(
                field_pressure.space,
                field_neumann.space,
                field_pressure,
                field_neumann,
                k_real,
                obs_points,
                _ff_op_kwargs,
                on_axis_idx,
                config.restrict_neumann_space,
                sphere_points,
            )
            callback_pressure_rows.append(per_freq_pressure)
            callback_spl_rows.append(per_freq_spl)
            log_entry["observation_spl_db"] = per_freq_spl
            log_entry["observation_angles_deg"] = angles_deg
            log_entry["observation_planes"] = config.observation.planes
            if per_freq_sphere is not None:
                callback_sphere_rows.append(per_freq_sphere)
                log_entry["observation_sphere_pressure_complex"] = per_freq_sphere

        # Progress callback
        if config.progress_callback is not None:
            config.progress_callback(i, len(frequencies), float(freq))

        # Early-stopping callback
        if has_callback:
            if config.on_frequency_result(i, float(freq), log_entry) is False:
                logger.info("Early stop requested after %.1f Hz", freq)
                break

    t_solve = time.time() - t_total

    if has_callback:
        logger.info("Reusing callback directivity rows for final result.")
        t_dir = 0.0
        pressure = np.stack(callback_pressure_rows, axis=0)
        spl = np.stack(callback_spl_rows, axis=0)
        sphere_pressure = (
            np.stack(callback_sphere_rows, axis=0)
            if sphere_points is not None
            else None
        )
    else:
        logger.info("Evaluating directivity at %d observation points...",
                    obs_points.shape[1] * obs_points.shape[0])
        t_dir = time.time()
        pressure, spl, sphere_pressure = _evaluate_directivity(
            freq_results, obs_points, angles_deg, config, sphere_points,
        )
        t_dir = time.time() - t_dir

    impedance = np.array(
        [fr.impedance for fr in freq_results], dtype=np.complex128,
    )

    solver_log = [_solver_log_entry(fr) for fr in freq_results]

    # Build surface_pressure_avg arrays
    sp_avg: dict[int, np.ndarray] = {}
    for tag in source_tags:
        sp_avg[tag] = np.array(surface_pavg[tag], dtype=np.complex128)

    surface_pressure_traces, surface_neumann_traces = _retained_surface_traces(
        freq_results, config.return_surface_traces,
    )

    return SolveResult(
        frequencies_hz=np.array(
            frequencies[:len(freq_results)], dtype=np.float64,
        ),
        pressure_complex=pressure,
        spl_db=spl,
        impedance=impedance,
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings={
            "solve_s": t_solve,
            "directivity_s": t_dir,
            "total_s": time.time() - t_total,
        },
        solver_log=solver_log,
        surface_pressure_avg=sp_avg if sp_avg else None,
        surface_pressure_complex=surface_pressure_traces,
        surface_neumann_complex=surface_neumann_traces,
        sphere_pressure_complex=sphere_pressure,
        sphere_theta_deg=sphere_theta_deg,
        sphere_phi_deg=sphere_phi_deg,
    )


def run_sweep_parallel(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
    worker_count: int,
    *,
    mesh_contracts_validated: bool = False,
) -> SolveResult:
    """Run frequency sweep across multiple processes.

    Each worker solves a chunk of frequencies, evaluates far-field pressure
    at observation points, and returns the results. This avoids shipping
    bempp GridFunction objects across process boundaries.

    ``progress_callback`` is supported: it stays in the parent process and is
    driven by a manager queue the workers publish to as each frequency lands.
    Progress therefore arrives out of order, which is inherent to chunking the
    sweep.

    ``on_frequency_result`` is not supported. It exists to abort a sweep early,
    and a worker cannot stop the chunks already running in its siblings, so
    honouring it here would silently mean something different from serial mode.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("frequencies must contain at least one value")

    if config.on_frequency_result is not None:
        raise ValueError(
            "on_frequency_result is not supported in parallel mode "
            "(workers > 1): early stopping cannot cancel frequencies already "
            "running in sibling workers. Use serial mode or set workers=1."
        )

    if not mesh_contracts_validated:
        _validate_velocity_source_tags(
            mesh.physical_tags, config.velocity_sources,
        )
        if config.require_closed_mesh:
            _require_closed_surface(
                np.asarray(mesh.grid.vertices, dtype=np.float64).T,
                np.asarray(mesh.grid.elements, dtype=np.int32).T,
            )

    t_total = time.time()
    obs_points, angles_deg = build_observation_points(frame, config.observation)
    sphere_points = None
    sphere_theta_deg = None
    sphere_phi_deg = None
    if config.observation.sphere_grid is not None:
        sphere_points, sphere_theta_deg, sphere_phi_deg = build_sphere_grid_points(
            frame, config.observation,
        )

    chunks = np.array_split(frequencies, min(worker_count, len(frequencies)))
    chunk_indices = np.array_split(
        np.arange(len(frequencies)), min(worker_count, len(frequencies)),
    )

    n_planes, n_angles, _ = obs_points.shape
    pressure_all = np.zeros(
        (len(frequencies), n_planes, n_angles), dtype=np.complex128,
    )
    spl_all = np.full(
        (len(frequencies), n_planes, n_angles), -120.0, dtype=np.float64,
    )
    sphere_pressure_all = (
        np.zeros((len(frequencies), sphere_points.shape[0]), dtype=np.complex128)
        if sphere_points is not None
        else None
    )
    impedance_all = np.zeros(len(frequencies), dtype=np.complex128)
    source_tags = list(config.velocity_sources.keys())
    surface_pressure_all = {
        tag: np.zeros(len(frequencies), dtype=np.complex128)
        for tag in source_tags
    }
    surface_pressure_traces_all = None
    surface_neumann_traces_all = None
    solver_log: list[dict] = [{}] * len(frequencies)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    # Callables cannot cross a spawn boundary. The callback stays here and is
    # fed by a queue the workers publish to.
    progress_callback = config.progress_callback
    worker_config = replace(
        config, progress_callback=None, on_frequency_result=None,
    )

    manager = ctx.Manager() if progress_callback is not None else None
    progress_queue = manager.Queue() if manager is not None else None
    completed = 0
    total = len(frequencies)

    def drain_progress() -> None:
        nonlocal completed
        if progress_queue is None:
            return
        while True:
            try:
                _global_i, frequency_hz = progress_queue.get_nowait()
            except queue.Empty:
                return
            # Report a monotonic count: chunks finish out of order, so the
            # frequency index itself would make the bar jump backwards.
            progress_callback(completed, total, float(frequency_hz))
            completed += 1

    try:
        with ProcessPoolExecutor(
            max_workers=len(chunks), mp_context=ctx,
        ) as executor:
            futures = {}
            for chunk_freqs, chunk_idx in zip(chunks, chunk_indices):
                fut = executor.submit(
                    _worker_solve_chunk,
                    mesh_grid_verts=np.array(mesh.grid.vertices),
                    mesh_grid_elems=np.array(mesh.grid.elements),
                    physical_tags=mesh.physical_tags,
                    frequencies=chunk_freqs,
                    obs_points=obs_points,
                    sphere_points=sphere_points,
                    angles_deg=angles_deg,
                    config=worker_config,
                    source_axis=np.asarray(frame.axis, dtype=np.float64),
                    progress_queue=progress_queue,
                    global_indices=np.asarray(chunk_idx, dtype=np.int64),
                )
                futures[fut] = chunk_idx

            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.2)
                drain_progress()
                for fut in done:
                    idx = futures[fut]
                    worker_result = fut.result()
                    if config.return_surface_traces:
                        (
                            chunk_pressure,
                            chunk_spl,
                            chunk_imp,
                            chunk_log,
                            chunk_surface_pressure,
                            chunk_sphere_pressure,
                            chunk_pressure_traces,
                            chunk_neumann_traces,
                        ) = worker_result
                        if surface_pressure_traces_all is None:
                            surface_pressure_traces_all = np.zeros(
                                (len(frequencies), chunk_pressure_traces.shape[1]),
                                dtype=np.complex128,
                            )
                            surface_neumann_traces_all = np.zeros(
                                (len(frequencies), chunk_neumann_traces.shape[1]),
                                dtype=np.complex128,
                            )
                    else:
                        (
                            chunk_pressure,
                            chunk_spl,
                            chunk_imp,
                            chunk_log,
                            chunk_surface_pressure,
                            chunk_sphere_pressure,
                        ) = worker_result
                        chunk_pressure_traces = None
                        chunk_neumann_traces = None
                    for local_i, global_i in enumerate(idx):
                        pressure_all[global_i] = chunk_pressure[local_i]
                        spl_all[global_i] = chunk_spl[local_i]
                        impedance_all[global_i] = chunk_imp[local_i]
                        solver_log[global_i] = chunk_log[local_i]
                        for tag in source_tags:
                            surface_pressure_all[tag][global_i] = (
                                chunk_surface_pressure[tag][local_i]
                            )
                        if (
                            surface_pressure_traces_all is not None
                            and surface_neumann_traces_all is not None
                            and chunk_pressure_traces is not None
                            and chunk_neumann_traces is not None
                        ):
                            surface_pressure_traces_all[global_i] = (
                                chunk_pressure_traces[local_i]
                            )
                            surface_neumann_traces_all[global_i] = (
                                chunk_neumann_traces[local_i]
                            )
                        if (
                            sphere_pressure_all is not None
                            and chunk_sphere_pressure is not None
                        ):
                            sphere_pressure_all[global_i] = chunk_sphere_pressure[local_i]
                    logger.info(
                        "Completed chunk: %d frequencies", len(idx),
                    )
            drain_progress()
    finally:
        if manager is not None:
            manager.shutdown()

    return SolveResult(
        frequencies_hz=frequencies,
        pressure_complex=pressure_all,
        spl_db=spl_all,
        impedance=impedance_all,
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings={"total_s": time.time() - t_total},
        solver_log=solver_log,
        surface_pressure_avg=surface_pressure_all if surface_pressure_all else None,
        surface_pressure_complex=surface_pressure_traces_all,
        surface_neumann_complex=surface_neumann_traces_all,
        sphere_pressure_complex=sphere_pressure_all,
        sphere_theta_deg=sphere_theta_deg,
        sphere_phi_deg=sphere_phi_deg,
    )


class WorkerSolveError(RuntimeError):
    """A parallel worker failed; carries the original type and traceback.

    Some exceptions raised deep in the OpenCL stack are not picklable (notably
    ``pyopencl._cl._ErrorRecord``), and an unpicklable exception surfaces in
    the parent as ``TypeError: cannot pickle ...``, destroying the real
    diagnostic exactly the way pyopencl's own cache handler does. Workers
    therefore report failures as this, which always survives the boundary.
    """


def _worker_solve_chunk(
    mesh_grid_verts,
    mesh_grid_elems,
    physical_tags,
    frequencies,
    obs_points,
    angles_deg,
    config,
    sphere_points=None,
    source_axis=None,
    progress_queue=None,
    global_indices=None,
):
    """Worker entry point; wraps failures so they cross the process boundary."""
    try:
        return _worker_solve_chunk_inner(
            mesh_grid_verts,
            mesh_grid_elems,
            physical_tags,
            frequencies,
            obs_points,
            angles_deg,
            config,
            sphere_points=sphere_points,
            source_axis=source_axis,
            progress_queue=progress_queue,
            global_indices=global_indices,
        )
    except BaseException as exc:
        import traceback

        raise WorkerSolveError(
            f"parallel worker failed solving "
            f"{float(np.min(frequencies)):.1f}-{float(np.max(frequencies)):.1f} Hz "
            f"with {type(exc).__name__}: {exc}\n"
            + "".join(traceback.format_exception(exc))
        ) from None


def _worker_solve_chunk_inner(
    mesh_grid_verts,
    mesh_grid_elems,
    physical_tags,
    frequencies,
    obs_points,
    angles_deg,
    config,
    sphere_points=None,
    source_axis=None,
    progress_queue=None,
    global_indices=None,
):
    """Reconstruct grid, solve, evaluate far-field, return arrays."""
    import bempp_cl.api as bempp_api

    grid = bempp_api.Grid(mesh_grid_verts, mesh_grid_elems)
    symmetry_context = None
    if config.native_symmetry_plane is None:
        p1_space, dp0_space = _setup_function_spaces(grid)
    else:
        from .symmetry import build_symmetry_context

        symmetry_context = build_symmetry_context(
            grid,
            physical_tags,
            config.native_symmetry_plane,
        )
        if config.require_closed_mesh:
            _require_closed_surface(
                np.asarray(
                    symmetry_context.expanded_grid.vertices,
                    dtype=np.float64,
                ).T,
                np.asarray(
                    symmetry_context.expanded_grid.elements,
                    dtype=np.int32,
                ).T,
            )
        p1_space = symmetry_context.reduced_p1
        dp0_space = symmetry_context.reduced_dp0

    n_planes, n_angles, _ = obs_points.shape
    pressure = np.zeros((len(frequencies), n_planes, n_angles), dtype=np.complex128)
    spl = np.full((len(frequencies), n_planes, n_angles), -120.0)
    sphere_pressure = (
        np.zeros((len(frequencies), sphere_points.shape[0]), dtype=np.complex128)
        if sphere_points is not None
        else None
    )
    impedance = np.zeros(len(frequencies), dtype=np.complex128)
    source_tags = list(config.velocity_sources.keys())
    axial_element_scale = None
    if config.source_motion == SourceMotion.AXIAL:
        axial_element_scale = _build_axial_element_scale(
            grid,
            physical_tags,
            source_tags,
            source_axis,
        )
    surface_pressure = {
        tag: np.zeros(len(frequencies), dtype=np.complex128)
        for tag in source_tags
    }
    log_entries = []
    pressure_trace_rows = [] if config.return_surface_traces else None
    neumann_trace_rows = [] if config.return_surface_traces else None

    # On-axis index: the angle closest to 0 degrees
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs = _operator_kwargs(
        backend, config.precision, config.opencl_device,
        vectorization_mode=getattr(config, "vectorization_mode", "auto"),
    )

    for i, freq in enumerate(frequencies):
        fr = solve_single_frequency(
            grid, physical_tags, freq, config,
            p1_space=p1_space,
            dp0_space=dp0_space,
            source_axis=source_axis,
            axial_element_scale=axial_element_scale,
            closed_mesh_validated=True,
            symmetry_context=symmetry_context,
        )
        impedance[i] = fr.impedance
        pavg = compute_surface_pressure_avg(
            grid,
            fr.pressure_on_surface,
            physical_tags,
            p1_space,
            source_tags,
        )
        for tag in source_tags:
            surface_pressure[tag][i] = pavg[tag]
        log_entries.append(_solver_log_entry(fr))
        if pressure_trace_rows is not None and neumann_trace_rows is not None:
            pressure_trace_rows.append(
                np.asarray(
                    fr.pressure_on_surface.coefficients,
                    dtype=np.complex128,
                )
            )
            neumann_trace_rows.append(
                np.asarray(fr.neumann_data.coefficients, dtype=np.complex128)
            )

        k_real = 2.0 * np.pi * freq / SPEED_OF_SOUND
        field_pressure = (
            fr.field_pressure_on_surface
            if fr.field_pressure_on_surface is not None
            else fr.pressure_on_surface
        )
        field_neumann = (
            fr.field_neumann_data
            if fr.field_neumann_data is not None
            else fr.neumann_data
        )
        arc_pressure, arc_spl, sphere_row = _evaluate_observation_planes(
            field_pressure.space,
            field_neumann.space,
            field_pressure,
            field_neumann,
            k_real,
            obs_points,
            op_kwargs,
            on_axis_idx,
            config.restrict_neumann_space,
            sphere_points,
        )
        pressure[i], spl[i] = arc_pressure, arc_spl
        if sphere_pressure is not None and sphere_row is not None:
            sphere_pressure[i] = sphere_row

        if progress_queue is not None:
            global_i = (
                int(global_indices[i]) if global_indices is not None else i
            )
            # Progress reporting must never take the sweep down with it.
            try:
                progress_queue.put_nowait((global_i, float(freq)))
            except Exception:  # pragma: no cover - queue full / manager gone
                logger.debug("dropped a progress update", exc_info=True)

    result = (
        pressure,
        spl,
        impedance,
        log_entries,
        surface_pressure,
        sphere_pressure,
    )
    if pressure_trace_rows is None or neumann_trace_rows is None:
        return result
    return (
        *result,
        np.stack(pressure_trace_rows, axis=0),
        np.stack(neumann_trace_rows, axis=0),
    )
