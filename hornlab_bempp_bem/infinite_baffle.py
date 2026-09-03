"""Coupled interior-BEM / Rayleigh infinite-baffle solver.

The aperture closes one interior acoustic domain. Its normal points into the
cavity, matching HornLab's Metal/CircSym convention. The unknown aperture
Neumann trace is coupled to the exterior half-space through

    p_aperture = 2 V_R q_aperture,

and the radiated field is evaluated from that aperture only. No finite baffle
geometry or image approximation is involved.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._blas_threads import limited_blas_threads
from ._constants import SPEED_OF_SOUND
from .backends import resolve_assembly_backend
from .bie import (
    _build_axial_element_scale,
    _build_neumann_data,
    _build_p1_to_dp0_projection,
    _operator_kwargs,
    _restrict_neumann_to_nonzero_support,
    compute_surface_pressure_avg,
)
from .config import BIEFormulation, SolveConfig, SourceMotion
from .mesh import LoadedMesh, MeshError, _require_closed_surface
from .observation import (
    ObservationFrame,
    build_observation_points,
    build_sphere_grid_points,
)
from .result import SolveResult
from .sweep import _normalized_spl_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ApertureGeometry:
    element_indices: NDArray[np.int64]
    center: NDArray[np.float64]
    inward_normal: NDArray[np.float64]
    outward_normal: NDArray[np.float64]


def _validate_coupled_infinite_baffle(
    mesh: LoadedMesh,
    config: SolveConfig,
    frame: ObservationFrame,
) -> _ApertureGeometry:
    """Validate the full-domain coupled-IB topology and orientation."""
    if config.aperture_tag is None:
        raise ValueError("aperture_tag must be set for an infinite-baffle solve")
    if config.native_symmetry_plane is not None:
        raise NotImplementedError(
            "BEMPP coupled infinite-baffle currently requires the full-domain "
            "mesh; use Metal BEM for native half/quarter image symmetry"
        )
    if config.formulation is BIEFormulation.BURTON_MILLER:
        raise NotImplementedError(
            "BEMPP coupled infinite-baffle supports STANDARD and COMPLEX_K; "
            "Burton-Miller coupling is not implemented"
        )
    if config.impedance_sources:
        raise NotImplementedError(
            "BEMPP coupled infinite-baffle does not yet compose with Robin "
            "impedance boundaries"
        )
    if config.return_surface_traces:
        raise NotImplementedError(
            "return_surface_traces is unavailable for coupled infinite-baffle "
            "solves because generic free-space trace evaluation would use the "
            "wrong exterior representation"
        )

    aperture_tag = int(config.aperture_tag)
    source_tags = {int(tag) for tag in config.velocity_sources}
    if aperture_tag in source_tags:
        raise ValueError("aperture_tag must not also be listed in velocity_sources")

    tags = np.asarray(mesh.physical_tags, dtype=np.int32)
    aperture_indices = np.flatnonzero(tags == aperture_tag).astype(np.int64)
    if aperture_indices.size == 0:
        raise ValueError(
            f"aperture_tag {aperture_tag} is not present in the mesh; "
            f"available physical tags: {sorted(int(x) for x in np.unique(tags))}"
        )

    vertices = np.asarray(mesh.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(mesh.grid.elements, dtype=np.int32).T
    _require_closed_surface(vertices, triangles)

    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    raw_normals = np.cross(p1 - p0, p2 - p0)
    normal_lengths = np.linalg.norm(raw_normals, axis=1)
    aperture_lengths = normal_lengths[aperture_indices]
    if np.any(aperture_lengths <= 1.0e-15):
        raise MeshError("coupled infinite-baffle aperture has degenerate triangles")

    aperture_normals = (
        raw_normals[aperture_indices] / aperture_lengths[:, None]
    )
    weighted_normal = np.sum(raw_normals[aperture_indices], axis=0)
    weighted_norm = float(np.linalg.norm(weighted_normal))
    if weighted_norm <= 1.0e-15:
        raise MeshError("coupled infinite-baffle aperture normal is ambiguous")
    inward_normal = weighted_normal / weighted_norm

    aperture_vertices = np.unique(triangles[aperture_indices].reshape(-1))
    center = np.average(
        (p0[aperture_indices] + p1[aperture_indices] + p2[aperture_indices]) / 3.0,
        weights=0.5 * aperture_lengths,
        axis=0,
    )
    mesh_extent = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    tolerance = max(1.0e-8 * mesh_extent, 1.0e-10)

    plane_distance = (vertices[aperture_vertices] - center) @ inward_normal
    if float(np.max(np.abs(plane_distance))) > tolerance:
        raise MeshError(
            "aperture_tag triangles must form one planar mouth patch; maximum "
            f"distance from the fitted plane is {np.max(np.abs(plane_distance)):.6g} m"
        )
    alignment = aperture_normals @ inward_normal
    if np.any(alignment < 1.0 - 1.0e-6):
        raise MeshError(
            "aperture_tag triangles must have one consistent normal direction"
        )

    # Every non-aperture vertex must be inside/behind the aperture plane in the
    # direction selected by the aperture normal. This both identifies that
    # normal as cavity-inward and rejects geometry leaking in front of the
    # infinite baffle.
    signed_depth = (vertices - center) @ inward_normal
    used_non_aperture = np.setdiff1d(
        np.unique(triangles.reshape(-1)),
        aperture_vertices,
        assume_unique=False,
    )
    if used_non_aperture.size == 0 or not np.any(
        signed_depth[used_non_aperture] > tolerance
    ):
        raise MeshError(
            "coupled infinite-baffle cavity must extend behind the aperture plane"
        )
    if np.any(signed_depth[used_non_aperture] < -tolerance):
        bad_vertex = int(
            used_non_aperture[
                np.flatnonzero(signed_depth[used_non_aperture] < -tolerance)[0]
            ]
        )
        raise MeshError(
            "coupled infinite-baffle cavity geometry leaks in front of the "
            f"aperture plane at vertex {bad_vertex}"
        )

    # Require consistent manifold winding and a single edge-connected aperture
    # whose physical rim is shared with non-coplanar cavity wall triangles.
    edge_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for tri_index, (a, b, c) in enumerate(triangles):
        for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (start, end) if start < end else (end, start)
            edge_uses.setdefault(edge, []).append((tri_index, start, end))
    for edge, uses in edge_uses.items():
        if len(uses) != 2:
            raise MeshError(
                "coupled infinite-baffle full-domain mesh must be watertight "
                f"and edge-manifold; edge {edge} has {len(uses)} uses"
            )
        (_, start0, end0), (_, start1, end1) = uses
        if start0 != end1 or end0 != start1:
            raise MeshError(
                "coupled infinite-baffle mesh has inconsistent triangle "
                f"winding across edge {edge}"
            )

    aperture_set = {int(index) for index in aperture_indices}
    adjacency = {index: set() for index in aperture_set}
    rim_edges = 0
    for edge, uses in edge_uses.items():
        aperture_uses = [use for use in uses if use[0] in aperture_set]
        if len(aperture_uses) == 2:
            left, right = aperture_uses[0][0], aperture_uses[1][0]
            adjacency[left].add(right)
            adjacency[right].add(left)
            continue
        if len(aperture_uses) != 1:
            continue
        other = uses[0] if uses[1][0] in aperture_set else uses[1]
        other_index = other[0]
        if int(tags[other_index]) in source_tags:
            raise MeshError(
                "aperture rim must join cavity wall triangles, not a velocity "
                f"source triangle ({other_index})"
            )
        other_vertices = triangles[other_index]
        other_plane_distance = (
            vertices[other_vertices] - center
        ) @ inward_normal
        if np.all(np.abs(other_plane_distance) <= tolerance):
            raise MeshError(
                "aperture_tag selects only part of a coplanar mouth patch"
            )
        rim_edges += 1

    seed = next(iter(aperture_set))
    visited = {seed}
    pending = [seed]
    while pending:
        current = pending.pop()
        for adjacent in adjacency[current] - visited:
            visited.add(adjacent)
            pending.append(adjacent)
    if visited != aperture_set:
        raise MeshError(
            "aperture_tag triangles must form one edge-connected mouth patch"
        )
    if rim_edges == 0:
        raise MeshError("aperture patch has no rim shared with the cavity wall")

    signed_volume = float(np.sum(p0 * np.cross(p1, p2)))
    if signed_volume >= 0.0:
        raise MeshError(
            "coupled infinite-baffle mesh must use interior-domain winding "
            "(negative signed volume)"
        )

    outward_normal = -inward_normal
    frame_axis = np.asarray(frame.axis, dtype=np.float64)
    frame_axis /= np.linalg.norm(frame_axis)
    if float(frame_axis @ outward_normal) < 0.99:
        raise ValueError(
            "observation frame axis must point out of the aperture into the "
            "radiating half-space"
        )

    return _ApertureGeometry(
        element_indices=aperture_indices,
        center=np.asarray(center, dtype=np.float64),
        inward_normal=np.asarray(inward_normal, dtype=np.float64),
        outward_normal=np.asarray(outward_normal, dtype=np.float64),
    )


def _evaluate_rayleigh_aperture(
    aperture_space,
    aperture_neumann,
    points: NDArray[np.float64],
    k_real: float,
    op_kwargs: dict,
    geometry: _ApertureGeometry,
) -> NDArray[np.complex128]:
    """Evaluate ``2 V_R q_a`` in front; return exactly zero behind."""
    import bempp_cl.api as bempp_api

    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros(pts.shape[0], dtype=np.complex128)
    front = (pts - geometry.center) @ geometry.inward_normal <= 1.0e-10
    if not np.any(front):
        return out
    active_points = np.ascontiguousarray(pts[front].T, dtype=np.float64)
    potential = bempp_api.operators.potential.helmholtz.single_layer(
        aperture_space,
        active_points,
        k_real,
        **op_kwargs,
    )
    out[front] = 2.0 * np.asarray(potential * aperture_neumann).reshape(-1)
    return out


def run_coupled_infinite_baffle_sweep(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
) -> SolveResult:
    """Solve a full-domain BEMPP coupled infinite-baffle frequency sweep."""
    import bempp_cl.api as bempp_api
    import scipy.linalg

    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if frequencies.size == 0:
        raise ValueError("frequencies must contain at least one value")
    geometry = _validate_coupled_infinite_baffle(mesh, config, frame)
    if config.workers > 1:
        logger.warning(
            "BEMPP coupled infinite-baffle currently runs serially so its "
            "frequency-local aperture systems stay deterministic"
        )

    t_total = time.time()
    grid = mesh.grid
    tags = np.asarray(mesh.physical_tags, dtype=np.int32)
    aperture_indices = geometry.element_indices
    p1_space = bempp_api.function_space(grid, "P", 1)
    dp0_space = bempp_api.function_space(grid, "DP", 0)
    aperture_space = bempp_api.function_space(
        grid,
        "DP",
        0,
        support_elements=aperture_indices,
    )
    if aperture_space.global_dof_count != aperture_indices.size:
        raise RuntimeError(
            "BEMPP aperture DP0 space does not map one DOF per aperture triangle"
        )

    p1_to_dp0 = _build_p1_to_dp0_projection(p1_space, dp0_space)
    aperture_pressure_projection = p1_to_dp0[aperture_indices]
    aperture_areas = np.asarray(grid.volumes, dtype=np.float64)[aperture_indices]
    if np.any(aperture_areas <= 0.0):
        raise MeshError("coupled infinite-baffle aperture has zero-area triangles")

    obs_points, angles_deg = build_observation_points(frame, config.observation)
    sphere_points = None
    sphere_theta_deg = None
    sphere_phi_deg = None
    if config.observation.sphere_grid is not None:
        sphere_points, sphere_theta_deg, sphere_phi_deg = build_sphere_grid_points(
            frame,
            config.observation,
        )
    flat_arc_points = obs_points.reshape(-1, 3)
    all_field_points = (
        flat_arc_points
        if sphere_points is None
        else np.concatenate([flat_arc_points, sphere_points], axis=0)
    )
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend_resolution = resolve_assembly_backend(config)
    backend = backend_resolution.effective_backend
    if backend_resolution.fallback_used:
        logger.warning(
            "Assembly backend %s could not be used; falling back to %s: %s",
            backend_resolution.requested_backend,
            backend_resolution.effective_backend,
            backend_resolution.reason or "no reason reported",
        )
    op_kwargs = _operator_kwargs(
        backend,
        config.precision,
        config.opencl_device,
        config.slp_dlp_quadrature,
        config.slp_dlp_singular_quadrature,
        vectorization_mode=config.vectorization_mode,
    )
    field_kwargs = _operator_kwargs(
        backend,
        config.precision,
        config.opencl_device,
        vectorization_mode=config.vectorization_mode,
    )
    axial_element_scale = None
    if config.source_motion == SourceMotion.AXIAL:
        axial_element_scale = _build_axial_element_scale(
            grid,
            tags,
            config.velocity_sources.keys(),
            frame.axis,
        )

    pressure_rows: list[NDArray[np.complex128]] = []
    spl_rows: list[NDArray[np.float64]] = []
    sphere_rows: list[NDArray[np.complex128]] = []
    impedance_rows: list[complex] = []
    solver_log: list[dict] = []
    completed_frequencies: list[float] = []
    source_tags = [int(tag) for tag in config.velocity_sources]
    surface_pressure_avg: dict[int, list[complex]] = {
        tag: [] for tag in source_tags
    }

    for index, frequency in enumerate(frequencies):
        case_start = time.perf_counter()
        omega = 2.0 * np.pi * float(frequency)
        k_real = omega / SPEED_OF_SOUND
        k = (
            k_real * (1.0 + 1j * config.complex_k_shift)
            if config.formulation is BIEFormulation.COMPLEX_K
            else complex(k_real)
        )
        phase: dict[str, float] = {}

        start = time.perf_counter()
        identity = bempp_api.operators.boundary.sparse.identity(
            p1_space,
            p1_space,
            p1_space,
        )
        dlp = bempp_api.operators.boundary.helmholtz.double_layer(
            p1_space,
            p1_space,
            p1_space,
            k,
            **op_kwargs,
        )
        aperture_slp = bempp_api.operators.boundary.helmholtz.single_layer(
            aperture_space,
            p1_space,
            p1_space,
            k,
            **op_kwargs,
        )
        rayleigh_slp = bempp_api.operators.boundary.helmholtz.single_layer(
            aperture_space,
            aperture_space,
            aperture_space,
            k_real,
            **op_kwargs,
        )
        driver_neumann = _build_neumann_data(
            dp0_space,
            tags,
            omega,
            config,
            config.precision,
            grid=grid,
            source_axis=frame.axis,
            axial_element_scale=axial_element_scale,
        )
        driver_space, driver_restricted = _restrict_neumann_to_nonzero_support(
            grid,
            driver_neumann,
        )
        driver_slp = bempp_api.operators.boundary.helmholtz.single_layer(
            driver_space,
            p1_space,
            p1_space,
            k,
            **op_kwargs,
        )
        phase["operator_construction_s"] = time.perf_counter() - start

        start = time.perf_counter()
        top_left = np.asarray(
            bempp_api.as_matrix((dlp - 0.5 * identity).weak_form()),
            dtype=np.complex128,
        )
        top_right = -np.asarray(
            bempp_api.as_matrix(aperture_slp.weak_form()),
            dtype=np.complex128,
        )
        rayleigh_weak = np.asarray(
            bempp_api.as_matrix(rayleigh_slp.weak_form()),
            dtype=np.complex128,
        )
        rayleigh_average = rayleigh_weak / aperture_areas[:, None]
        driver_matrix = np.asarray(
            bempp_api.as_matrix(driver_slp.weak_form()),
            dtype=np.complex128,
        )
        rhs_top = driver_matrix @ np.asarray(
            driver_restricted.coefficients,
            dtype=np.complex128,
        )
        phase["assembly_s"] = time.perf_counter() - start

        start = time.perf_counter()
        n_pressure = p1_space.global_dof_count
        n_aperture = aperture_space.global_dof_count
        system = np.empty(
            (n_pressure + n_aperture, n_pressure + n_aperture),
            dtype=np.complex128,
        )
        system[:n_pressure, :n_pressure] = top_left
        system[:n_pressure, n_pressure:] = top_right
        system[n_pressure:, :n_pressure] = aperture_pressure_projection.toarray()
        system[n_pressure:, n_pressure:] = -2.0 * rayleigh_average
        rhs = np.zeros(n_pressure + n_aperture, dtype=np.complex128)
        rhs[:n_pressure] = rhs_top
        phase["system_build_s"] = time.perf_counter() - start

        start = time.perf_counter()
        with limited_blas_threads(1):
            solution = scipy.linalg.solve(system, rhs, assume_a="gen")
        phase["linear_solve_s"] = time.perf_counter() - start

        pressure_coefficients = solution[:n_pressure]
        aperture_coefficients = solution[n_pressure:]
        pressure_surface = bempp_api.GridFunction(
            p1_space,
            coefficients=pressure_coefficients,
        )
        aperture_neumann = bempp_api.GridFunction(
            aperture_space,
            coefficients=aperture_coefficients,
        )
        aperture_pressure = aperture_pressure_projection @ pressure_coefficients
        rayleigh_pressure = 2.0 * (rayleigh_average @ aperture_coefficients)
        continuity_rel = float(
            np.linalg.norm(aperture_pressure - rayleigh_pressure)
            / max(float(np.linalg.norm(aperture_pressure)), 1.0e-30)
        )

        start = time.perf_counter()
        field_flat = _evaluate_rayleigh_aperture(
            aperture_space,
            aperture_neumann,
            all_field_points,
            k_real,
            field_kwargs,
            geometry,
        )
        arc_count = flat_arc_points.shape[0]
        arc_pressure = field_flat[:arc_count].reshape(obs_points.shape[:2])
        arc_spl = _normalized_spl_db(arc_pressure, on_axis_idx)
        sphere_pressure = (
            field_flat[arc_count:] if sphere_points is not None else None
        )
        phase["field_s"] = time.perf_counter() - start

        pavg = compute_surface_pressure_avg(
            grid,
            pressure_surface,
            tags,
            p1_space,
            source_tags,
        )
        impedance = pavg[min(source_tags)] if source_tags else 0.0 + 0.0j
        for source_tag in source_tags:
            surface_pressure_avg[source_tag].append(pavg[source_tag])

        elapsed = time.perf_counter() - case_start
        phase["total_s"] = elapsed
        diagnostics = {
            "coupled_ib": True,
            "backend": "bempp",
            "field": "rayleigh_aperture_only",
            "aperture_tag": int(config.aperture_tag),
            "aperture_triangles": int(n_aperture),
            "aperture_velocity_basis": "DP0",
            "aperture_pressure_continuity_rel": continuity_rel,
            "complex_k": config.formulation is BIEFormulation.COMPLEX_K,
            "assembly_backend": backend,
            "requested_backend": backend_resolution.requested_backend,
            "effective_backend": backend_resolution.effective_backend,
            "fallback_used": backend_resolution.fallback_used,
            "reason": backend_resolution.reason,
        }
        log_entry = {
            "frequency_hz": float(frequency),
            "iterations": None,
            "converged": True,
            "timing_s": elapsed,
            "requested_precision": config.precision,
            "effective_precision": config.precision,
            "requested_solver": config.solver.value,
            "effective_solver": "lu",
            "requested_backend": backend_resolution.requested_backend,
            "effective_backend": backend_resolution.effective_backend,
            "fallback_used": backend_resolution.fallback_used,
            "reason": backend_resolution.reason,
            "phase_timings": phase,
            "impedance": impedance,
            "native_diagnostics": diagnostics,
        }
        if config.on_frequency_result is not None:
            log_entry.update(
                {
                    "observation_pressure_complex": arc_pressure,
                    "observation_spl_db": arc_spl,
                    "observation_angles_deg": angles_deg,
                    "observation_planes": config.observation.planes,
                    "observation_sphere_pressure_complex": sphere_pressure,
                }
            )

        pressure_rows.append(arc_pressure)
        spl_rows.append(arc_spl)
        if sphere_pressure is not None:
            sphere_rows.append(sphere_pressure)
        impedance_rows.append(impedance)
        solver_log.append(log_entry)
        completed_frequencies.append(float(frequency))

        if config.progress_callback is not None:
            config.progress_callback(index, len(frequencies), float(frequency))
        if (
            config.on_frequency_result is not None
            and config.on_frequency_result(index, float(frequency), log_entry) is False
        ):
            logger.info("Early stop requested after %.1f Hz", frequency)
            break

    pressure = np.stack(pressure_rows, axis=0)
    spl = np.stack(spl_rows, axis=0)
    sphere = np.stack(sphere_rows, axis=0) if sphere_points is not None else None
    return SolveResult(
        frequencies_hz=np.asarray(completed_frequencies, dtype=np.float64),
        pressure_complex=pressure,
        spl_db=spl,
        impedance=np.asarray(impedance_rows, dtype=np.complex128),
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings={
            "solve_s": time.time() - t_total,
            "directivity_s": sum(
                entry["phase_timings"]["field_s"] for entry in solver_log
            ),
            "total_s": time.time() - t_total,
        },
        solver_log=solver_log,
        surface_pressure_avg={
            tag: np.asarray(values, dtype=np.complex128)
            for tag, values in surface_pressure_avg.items()
        },
        sphere_pressure_complex=sphere,
        sphere_theta_deg=sphere_theta_deg,
        sphere_phi_deg=sphere_phi_deg,
    )
