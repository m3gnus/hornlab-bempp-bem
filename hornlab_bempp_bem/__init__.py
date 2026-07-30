"""hornlab-bempp-bem — bempp-cl acoustic BEM solver for HornLab.

Public API:
    solve(mesh, config)            → SolveResult (full frequency sweep)
    solve_frequencies(mesh, freqs) → SolveResult (caller-ordered frequencies)
    load_mesh(path, scale)         → LoadedMesh
"""
from __future__ import annotations

import logging
import os

import numpy as np

from .backends import (
    AssemblyBackendResolution,
    AssemblyBackendUnavailable,
    resolve_assembly_backend,
)
from .config import (
    BIEFormulation,
    LinearSolver,
    ObservationConfig,
    SolveConfig,
    SourceMotion,
    VelocityMode,
    reject_unsupported_native_symmetry,
)
from .device import OpenCLError, configure_opencl
from .mesh import (
    LoadedMesh,
    MeshError,
    _require_closed_surface,
    _validate_velocity_source_tags,
    load_mesh,
)
from .observation import ObservationFrame, infer_frame
from .result import MeshInfo, SolveResult

__all__ = [
    "solve",
    "solve_frequencies",
    "load_mesh",
    "SolveConfig",
    "SolveResult",
    "ObservationConfig",
    "BIEFormulation",
    "LinearSolver",
    "VelocityMode",
    "SourceMotion",
    "LoadedMesh",
    "MeshInfo",
    "MeshError",
    "ObservationFrame",
    "OpenCLError",
    "configure_opencl",
    "AssemblyBackendResolution",
    "AssemblyBackendUnavailable",
    "resolve_assembly_backend",
]

logger = logging.getLogger(__name__)


def _coerce_frequencies(frequencies_hz) -> np.ndarray:
    """Return a one-dimensional array of finite, positive frequencies."""
    try:
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            "frequencies_hz must be a one-dimensional sequence of numbers"
        ) from None

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies_hz must be a one-dimensional sequence of numbers"
        )
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_hz must contain only finite values")
    if np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain only positive values")
    return frequencies


# A spawned worker re-imports bempp-cl and re-JITs its numba kernels before it
# can solve anything: about 5 s on the reference host (1.4 s import, 3.5 s JIT,
# measured 2026-07-30), against roughly 0.13 s per warm frequency. bempp-cl's
# hot kernels are declared without ``cache=True`` and are not cacheable even
# when caching is forced on their dispatchers, so this cost is per process and
# cannot be amortized on disk. A worker therefore has to solve about 40
# frequencies just to pay for its own start-up, and a short sweep is
# dramatically faster in one warm process -- 16 frequencies measured at 1.7 s
# serial against 16.8 s across two workers.
_MIN_FREQUENCIES_PER_WORKER = 40


def _detect_worker_count() -> int:
    """Auto-detect physical core count (not hyperthreads)."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        count = multiprocessing.cpu_count() or 1
    return max(1, count)


def _resolve_worker_count(requested: int, n_frequencies: int) -> int:
    """Pick a worker count that cannot make the sweep slower.

    ``workers=0`` means auto: split only when each worker gets enough
    frequencies to cover its start-up. An explicit count above 1 is honoured --
    the caller may know something we do not, such as a long sweep on a machine
    with idle cores -- but is warned about when the arithmetic says it will
    lose.
    """
    if requested == 0:
        return max(1, min(
            _detect_worker_count(),
            n_frequencies // _MIN_FREQUENCIES_PER_WORKER,
        ))
    if requested > 1 and n_frequencies < requested * _MIN_FREQUENCIES_PER_WORKER:
        logger.warning(
            "workers=%d requested for only %d frequencies; each worker pays "
            "about 5 s of start-up, so this is likely slower than workers=1. "
            "Use workers=0 to let the sweep decide.",
            requested, n_frequencies,
        )
    return requested


def _resolve_mesh(
    mesh,
    scale: float = 1.0,
    require_closed: bool = False,
    native_symmetry_plane: str | None = None,
) -> LoadedMesh:
    """Accept str, Path, or LoadedMesh."""
    if isinstance(mesh, LoadedMesh):
        if require_closed:
            _require_closed_surface(
                np.asarray(mesh.grid.vertices, dtype=np.float64).T,
                np.asarray(mesh.grid.elements, dtype=np.int32).T,
            )
        return mesh
    return load_mesh(
        mesh,
        scale=scale,
        require_closed=require_closed,
        native_symmetry_plane=native_symmetry_plane,
    )


def _resolve_frame(loaded: LoadedMesh, config: SolveConfig) -> ObservationFrame:
    """Resolve the observation frame from an override or mesh inference."""
    if config.frame_override is not None:
        return config.frame_override

    return infer_frame(
        loaded.grid,
        loaded.physical_tags,
        source_tag=min(config.velocity_sources.keys(), default=2),
        origin_at=config.observation.origin,
        symmetry_plane=config.native_symmetry_plane,
    )


def solve(
    mesh,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run a BEM acoustic solve across a frequency sweep.

    Parameters
    ----------
    mesh : str | Path | LoadedMesh
        Path to a .msh file (with ABEC-convention physical groups),
        or a pre-loaded LoadedMesh from load_mesh().

    config : SolveConfig, optional
        Full solve specification. Defaults to SolveConfig() if None.

    Returns
    -------
    SolveResult with complex pressure, SPL, impedance, and metadata.
    """
    if config is None:
        config = SolveConfig()

    reject_unsupported_native_symmetry(config)

    loaded = _resolve_mesh(
        mesh,
        scale=config.mesh_scale,
        require_closed=(
            config.require_closed_mesh
            and config.native_symmetry_plane is None
        ),
        native_symmetry_plane=config.native_symmetry_plane,
    )
    _validate_velocity_source_tags(loaded.physical_tags, config.velocity_sources)
    frame = _resolve_frame(loaded, config)

    from .sweep import (
        _build_frequency_grid,
        run_sweep_parallel,
        run_sweep_serial,
    )

    frequencies = _build_frequency_grid(config)

    workers = _resolve_worker_count(config.workers, len(frequencies))

    if workers <= 1:
        return run_sweep_serial(
            loaded,
            frequencies,
            frame,
            config,
            mesh_contracts_validated=True,
        )
    else:
        return run_sweep_parallel(
            loaded,
            frequencies,
            frame,
            config,
            workers,
            mesh_contracts_validated=True,
        )


def solve_frequencies(
    mesh,
    frequencies_hz: list[float] | np.ndarray,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Solve at specific frequencies in caller-specified order.

    Unlike solve(), this does not generate a frequency grid from config.
    The frequencies are solved in the exact order given -- useful for
    priority-ordered solves with external early-stopping logic.
    """
    if config is None:
        config = SolveConfig()

    reject_unsupported_native_symmetry(config)
    freqs = _coerce_frequencies(frequencies_hz)

    loaded = _resolve_mesh(
        mesh,
        scale=config.mesh_scale,
        require_closed=(
            config.require_closed_mesh
            and config.native_symmetry_plane is None
        ),
        native_symmetry_plane=config.native_symmetry_plane,
    )
    _validate_velocity_source_tags(loaded.physical_tags, config.velocity_sources)
    frame = _resolve_frame(loaded, config)

    from .sweep import run_sweep_serial

    # Always serial for caller-ordered frequencies (order matters)
    return run_sweep_serial(
        loaded,
        freqs,
        frame,
        config,
        mesh_contracts_validated=True,
    )
