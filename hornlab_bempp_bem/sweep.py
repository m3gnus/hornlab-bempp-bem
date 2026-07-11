"""Frequency sweep — serial and parallel execution."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
from .backends import resolve_assembly_backend
from .bie import (
    FrequencyResult,
    _evaluate_far_field,
    _operator_kwargs,
    _setup_function_spaces,
    compute_surface_pressure_avg,
    solve_single_frequency,
)
from .config import SolveConfig
from .mesh import (
    LoadedMesh,
    _require_closed_surface,
    _validate_velocity_source_tags,
)
from .observation import ObservationFrame, build_observation_points
from .result import SolveResult

logger = logging.getLogger(__name__)


def _build_frequency_grid(config: SolveConfig) -> NDArray[np.float64]:
    if config.freq_spacing == "log":
        return np.geomspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)
    return np.linspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)


def _evaluate_directivity(
    freq_results: list[FrequencyResult],
    obs_points: NDArray[np.float64],
    angles_deg: NDArray[np.float64],
    config: SolveConfig,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Evaluate far-field pressure at observation points for all frequencies.

    Returns:
        pressure_complex: (F, P, N_angles)
        spl_db: (F, P, N_angles) — normalised on-axis = 0 dB
    """
    n_freq = len(freq_results)
    n_planes, n_angles, _ = obs_points.shape

    pressure = np.zeros((n_freq, n_planes, n_angles), dtype=np.complex128)
    spl = np.full((n_freq, n_planes, n_angles), -120.0, dtype=np.float64)

    # On-axis index: the angle closest to 0 degrees
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs = _operator_kwargs(backend, config.precision, config.opencl_device)

    for fi, fr in enumerate(freq_results):
        k_real = 2.0 * np.pi * fr.frequency_hz / SPEED_OF_SOUND

        p1 = fr.pressure_on_surface.space
        dp0 = fr.neumann_data.space

        for pi in range(n_planes):
            pts = obs_points[pi]  # (N_angles, 3)
            p_complex = _evaluate_far_field(
                p1, dp0,
                fr.pressure_on_surface,
                fr.neumann_data,
                k_real, pts, op_kwargs,
            )
            pressure[fi, pi, :] = p_complex

            amplitudes = np.abs(p_complex)
            spl_raw = np.where(
                amplitudes > 1e-15,
                20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                -120.0,
            )
            # Normalise: on-axis (0 deg) = 0 dB
            spl[fi, pi, :] = spl_raw - spl_raw[on_axis_idx]

    return pressure, spl


def run_sweep_serial(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
    *,
    mesh_contracts_validated: bool = False,
) -> SolveResult:
    """Run frequency sweep in a single process."""
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

    p1_space, dp0_space = _setup_function_spaces(mesh.grid)

    source_tags = list(config.velocity_sources.keys())
    freq_results: list[FrequencyResult] = []
    surface_pavg: dict[int, list[complex]] = {tag: [] for tag in source_tags}
    completed_freqs: list[float] = []

    # Pre-compute op_kwargs for per-frequency far-field evaluation
    # (only used when on_frequency_result is set)
    has_callback = config.on_frequency_result is not None
    callback_pressure_rows: list[NDArray[np.complex128]] = []
    callback_spl_rows: list[NDArray[np.float64]] = []
    if has_callback:
        _backend = resolve_assembly_backend(config).effective_backend
        _ff_op_kwargs = _operator_kwargs(
            _backend, config.precision, config.opencl_device,
        )
        on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    for i, freq in enumerate(frequencies):
        logger.info("[%d/%d] %.1f Hz", i + 1, len(frequencies), freq)
        fr = solve_single_frequency(
            mesh.grid, mesh.physical_tags, freq, config,
            p1_space=p1_space,
            dp0_space=dp0_space,
            source_axis=frame.axis,
            closed_mesh_validated=True,
        )
        freq_results.append(fr)
        completed_freqs.append(float(freq))

        # Surface pressure average per source tag
        pavg = compute_surface_pressure_avg(
            mesh.grid, fr.pressure_on_surface,
            mesh.physical_tags, p1_space, source_tags,
        )
        for tag in source_tags:
            surface_pavg[tag].append(pavg[tag])

        log_entry = {
            "frequency_hz": fr.frequency_hz,
            "iterations": fr.iterations,
            "converged": fr.converged,
            "timing_s": fr.timing_s,
            "impedance": fr.impedance,
        }

        # When the callback is set, evaluate per-frequency directivity
        # so the caller can act on partial results as they stream in.
        if has_callback:
            k_real = 2.0 * np.pi * fr.frequency_hz / SPEED_OF_SOUND
            n_planes = obs_points.shape[0]
            n_angles = obs_points.shape[1]
            per_freq_pressure = np.zeros(
                (n_planes, n_angles), dtype=np.complex128,
            )
            per_freq_spl = np.full((n_planes, n_angles), -120.0, dtype=np.float64)
            for pi in range(n_planes):
                p_complex = _evaluate_far_field(
                    p1_space, dp0_space,
                    fr.pressure_on_surface, fr.neumann_data,
                    k_real, obs_points[pi], _ff_op_kwargs,
                )
                per_freq_pressure[pi, :] = p_complex
                amplitudes = np.abs(p_complex)
                spl_raw = np.where(
                    amplitudes > 1e-15,
                    20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                    -120.0,
                )
                per_freq_spl[pi, :] = spl_raw - spl_raw[on_axis_idx]
            callback_pressure_rows.append(per_freq_pressure)
            callback_spl_rows.append(per_freq_spl)
            log_entry["observation_spl_db"] = per_freq_spl
            log_entry["observation_angles_deg"] = angles_deg
            log_entry["observation_planes"] = config.observation.planes

        # Progress callback
        if config.progress_callback is not None:
            config.progress_callback(i, len(frequencies), float(freq))

        # Early-stopping callback
        if has_callback:
            if config.on_frequency_result(i, float(freq), log_entry) is False:
                logger.info("Early stop requested after %.1f Hz", freq)
                break

    t_solve = time.time() - t_total

    # Trim frequencies to only those actually completed (for early stopping)
    actual_freqs = np.array(completed_freqs, dtype=np.float64)

    if has_callback and len(callback_pressure_rows) == len(freq_results):
        logger.info("Reusing callback directivity rows for final result.")
        t_dir = 0.0
        pressure = np.stack(callback_pressure_rows, axis=0)
        spl = np.stack(callback_spl_rows, axis=0)
    else:
        logger.info("Evaluating directivity at %d observation points...",
                    obs_points.shape[1] * obs_points.shape[0])
        t_dir = time.time()
        pressure, spl = _evaluate_directivity(
            freq_results, obs_points, angles_deg, config,
        )
        t_dir = time.time() - t_dir

    impedance = np.array(
        [fr.impedance for fr in freq_results], dtype=np.complex128,
    )

    solver_log = [
        {
            "frequency_hz": fr.frequency_hz,
            "iterations": fr.iterations,
            "converged": fr.converged,
            "timing_s": fr.timing_s,
        }
        for fr in freq_results
    ]

    # Build surface_pressure_avg arrays
    sp_avg: dict[int, np.ndarray] = {}
    for tag in source_tags:
        sp_avg[tag] = np.array(surface_pavg[tag], dtype=np.complex128)

    return SolveResult(
        frequencies_hz=actual_freqs,
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

    Callbacks (progress_callback, on_frequency_result) are not supported
    in parallel mode — they are not picklable across process boundaries.
    """
    if config.progress_callback is not None or config.on_frequency_result is not None:
        raise ValueError(
            "progress_callback and on_frequency_result are not supported in "
            "parallel mode (workers > 1). Use serial mode or set workers=1."
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
    impedance_all = np.zeros(len(frequencies), dtype=np.complex128)
    source_tags = list(config.velocity_sources.keys())
    surface_pressure_all = {
        tag: np.zeros(len(frequencies), dtype=np.complex128)
        for tag in source_tags
    }
    solver_log: list[dict] = [{}] * len(frequencies)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=ctx,
    ) as executor:
        futures = {}
        for ci, (chunk_freqs, chunk_idx) in enumerate(
            zip(chunks, chunk_indices),
        ):
            fut = executor.submit(
                _worker_solve_chunk,
                mesh_grid_verts=np.array(mesh.grid.vertices),
                mesh_grid_elems=np.array(mesh.grid.elements),
                physical_tags=mesh.physical_tags,
                frequencies=chunk_freqs,
                obs_points=obs_points,
                angles_deg=angles_deg,
                config=config,
                source_axis=np.asarray(frame.axis, dtype=np.float64),
            )
            futures[fut] = chunk_idx

        for fut in as_completed(futures):
            idx = futures[fut]
            (
                chunk_pressure,
                chunk_spl,
                chunk_imp,
                chunk_log,
                chunk_surface_pressure,
            ) = fut.result()
            for local_i, global_i in enumerate(idx):
                pressure_all[global_i] = chunk_pressure[local_i]
                spl_all[global_i] = chunk_spl[local_i]
                impedance_all[global_i] = chunk_imp[local_i]
                solver_log[global_i] = chunk_log[local_i]
                for tag in source_tags:
                    surface_pressure_all[tag][global_i] = chunk_surface_pressure[tag][
                        local_i
                    ]
            logger.info(
                "Completed chunk: %d frequencies", len(idx),
            )

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
    )


def _worker_solve_chunk(
    mesh_grid_verts,
    mesh_grid_elems,
    physical_tags,
    frequencies,
    obs_points,
    angles_deg,
    config,
    source_axis=None,
):
    """Worker function: reconstruct grid, solve, evaluate far-field, return arrays."""
    import bempp_cl.api as bempp_api

    from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
    grid = bempp_api.Grid(mesh_grid_verts, mesh_grid_elems)
    p1_space, dp0_space = _setup_function_spaces(grid)

    n_planes, n_angles, _ = obs_points.shape
    pressure = np.zeros((len(frequencies), n_planes, n_angles), dtype=np.complex128)
    spl = np.full((len(frequencies), n_planes, n_angles), -120.0)
    impedance = np.zeros(len(frequencies), dtype=np.complex128)
    source_tags = list(config.velocity_sources.keys())
    surface_pressure = {
        tag: np.zeros(len(frequencies), dtype=np.complex128)
        for tag in source_tags
    }
    log_entries = []

    # On-axis index: the angle closest to 0 degrees
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs = _operator_kwargs(backend, config.precision, config.opencl_device)

    for i, freq in enumerate(frequencies):
        fr = solve_single_frequency(
            grid, physical_tags, freq, config,
            p1_space=p1_space,
            dp0_space=dp0_space,
            source_axis=source_axis,
            closed_mesh_validated=True,
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
        log_entries.append({
            "frequency_hz": fr.frequency_hz,
            "iterations": fr.iterations,
            "converged": fr.converged,
            "timing_s": fr.timing_s,
        })

        k_real = 2.0 * np.pi * freq / SPEED_OF_SOUND
        for pi in range(n_planes):
            pts = obs_points[pi]
            p_complex = _evaluate_far_field(
                p1_space, dp0_space,
                fr.pressure_on_surface, fr.neumann_data,
                k_real, pts, op_kwargs,
            )
            pressure[i, pi, :] = p_complex
            amplitudes = np.abs(p_complex)
            spl_raw = np.where(
                amplitudes > 1e-15,
                20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                -120.0,
            )
            spl[i, pi, :] = spl_raw - spl_raw[on_axis_idx]

    return pressure, spl, impedance, log_entries, surface_pressure
