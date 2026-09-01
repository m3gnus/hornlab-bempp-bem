"""BIE assembly, boundary conditions, and field evaluation.

This is the physics core. One formulation, one solve, one evaluation path.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from ._blas_threads import limited_blas_threads
from ._constants import SPEED_OF_SOUND
from .backends import resolve_assembly_backend
from .config import (
    BIEFormulation,
    LinearSolver,
    SolveConfig,
    SourceMotion,
    VelocityMode,
    reject_unsupported_native_symmetry,
    uses_image_assembly,
)
from .mesh import _require_closed_surface

logger = logging.getLogger(__name__)


@dataclass
class FrequencyResult:
    """Result from solving a single frequency."""
    frequency_hz: float
    pressure_on_surface: object       # bempp GridFunction (P1)
    neumann_data: object              # bempp GridFunction (DP0)
    impedance: complex
    iterations: int | None
    converged: bool
    timing_s: float
    phase_timings: dict[str, float] = field(default_factory=dict)
    field_pressure_on_surface: object | None = None
    field_neumann_data: object | None = None
    # Regular-quadrature selection diagnostics; populated when
    # config.adaptive_quadrature is enabled (see _select_regular_quadrature).
    regular_quadrature: dict = field(default_factory=dict)


def _choose_solver(config: SolveConfig, n_triangles: int) -> LinearSolver:
    if config.solver is not LinearSolver.AUTO:
        return config.solver
    return LinearSolver.LU if n_triangles <= config.lu_threshold else LinearSolver.GMRES


def _operator_kwargs(
    backend: str,
    precision: str,
    opencl_device: str = "cpu",
    quadrature_order: int | None = None,
    singular_quadrature_order: int | None = None,
    vectorization_mode: str = "auto",
) -> dict:
    """Build kwargs for bempp operator construction."""
    kwargs: dict = {}
    if backend != "auto":
        kwargs["assembler"] = "dense"
        kwargs["device_interface"] = backend
    if backend == "opencl":
        from .device import configure_opencl, set_vectorization_mode

        configure_opencl(opencl_device)
        set_vectorization_mode(vectorization_mode)
    if precision == "single":
        kwargs["precision"] = "single"
    if quadrature_order is not None or singular_quadrature_order is not None:
        from bempp_cl.api.utils.parameters import DefaultParameters

        params = DefaultParameters()
        if quadrature_order is not None:
            params.quadrature.regular = int(quadrature_order)
        if singular_quadrature_order is not None:
            params.quadrature.singular = int(singular_quadrature_order)
        kwargs["parameters"] = params
    return kwargs


def _p90_element_length(grid) -> float:
    """Characteristic mesh length h = sqrt(p90 element area), in metres."""
    areas = np.asarray(grid.volumes, dtype=np.float64)
    return float(np.sqrt(np.percentile(areas, 90.0)))


def _select_regular_quadrature(
    config: SolveConfig,
    k_real: float,
    element_length_m: float,
) -> tuple[int, int, dict]:
    """Per-frequency regular quadrature orders and diagnostics.

    Wavelength-adaptive policy ported from boundary-lab's BEAT Engine CPU:
    while ``adaptive_quadrature_kh_min <= k*h <= adaptive_quadrature_kh_max``
    the regular (non-adjacent pair) orders drop to
    ``adaptive_quadrature_low_order``; outside the window the configured
    base orders apply unchanged. The lower bound is a tuned departure from
    BEAT (see SolveConfig); 0.0 recovers their pure upper threshold.
    Singular quadrature is never touched. Returns
    ``(slp_dlp_order, hyp_adlp_order, diagnostics)``.
    """
    slp_order = config.slp_dlp_quadrature
    hyp_order = config.hyp_adlp_quadrature
    if not config.adaptive_quadrature:
        return slp_order, hyp_order, {}
    kh = float(k_real) * float(element_length_m)
    if config.adaptive_quadrature_kh_min <= kh <= config.adaptive_quadrature_kh_max:
        # Never raise an order above its configured base.
        slp_order = min(slp_order, config.adaptive_quadrature_low_order)
        hyp_order = min(hyp_order, config.adaptive_quadrature_low_order)
    diagnostics = {
        "mode": "wavelength",
        "slp_dlp_order": int(slp_order),
        "hyp_adlp_order": int(hyp_order),
        "base_slp_dlp_order": int(config.slp_dlp_quadrature),
        "base_hyp_adlp_order": int(config.hyp_adlp_quadrature),
        "element_length_m": float(element_length_m),
        "kh": kh,
        "kh_min": float(config.adaptive_quadrature_kh_min),
        "kh_max": float(config.adaptive_quadrature_kh_max),
    }
    return slp_order, hyp_order, diagnostics


def _setup_function_spaces(grid):
    """Create P1 and DP0 function spaces on the grid."""
    import bempp_cl.api as bempp_api

    p1 = bempp_api.function_space(grid, "P", 1)
    dp0 = bempp_api.function_space(grid, "DP", 0)
    return p1, dp0


def _build_axial_element_scale(
    grid,
    physical_tags: NDArray[np.int32],
    source_tags,
    axis: NDArray[np.float64],
) -> NDArray | None:
    """Per-element ``n_hat . axis`` projection for a rigid axial (piston) source.

    ``axis`` is the resolved observation-frame axis shared by every source tag.
    Each tag gets one area-weighted sign vote so its drive remains
    outward-positive, while the per-face projection retains the frame-relative
    cosine used by hornlab-metal-bem. Returns ``None`` for a degenerate axis or
    when no configured source face is present.
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis.shape[0] != 3:
        return None
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        return None
    axis = axis / axis_norm

    vertices = np.asarray(grid.vertices.T, dtype=np.float64)
    elements = np.asarray(grid.elements.T, dtype=np.int32)
    n_elem = elements.shape[0]

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    raw = np.cross(p1 - p0, p2 - p0)  # outward (canonical winding); |raw| = 2*area
    mags = np.linalg.norm(raw, axis=1)

    scale = np.zeros(n_elem, dtype=np.float64)
    any_source = False
    for tag in sorted({int(t) for t in source_tags}):
        idx = np.where(physical_tags == tag)[0]
        if idx.size == 0:
            continue
        tag_mags = mags[idx]
        safe_mags = np.where(tag_mags > 1e-15, tag_mags, 1.0)
        unit_normals = raw[idx] / safe_mags[:, None]
        projection = unit_normals @ axis
        if float(np.dot(projection, tag_mags)) < 0.0:
            projection = -projection
        scale[idx] = projection
        any_source = True

    return scale if any_source else None


def _build_neumann_data(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    precision: str = "single",
    grid=None,
    source_axis: NDArray[np.float64] | None = None,
    axial_element_scale: NDArray[np.float64] | None = None,
) -> object:
    """Construct Neumann boundary condition: dp/dn = i*rho*omega*v_n.

    Supports multiple velocity source tags with independent weights, and
    acceleration vs velocity driving modes. When ``config.source_motion ==
    "axial"`` (and ``grid`` is supplied) each source face is driven at
    ``weight * (n_hat . axis)`` -- a rigid piston -- instead of a uniform normal
    velocity; the default "normal" path is unchanged.
    """
    import bempp_cl.api as bempp_api

    dtype = np.complex64 if precision == "single" else np.complex128
    coeffs = _build_neumann_coefficients(
        dp0_space,
        physical_tags,
        omega,
        config,
        dtype,
        grid=grid,
        source_axis=source_axis,
        axial_element_scale=axial_element_scale,
    )
    return bempp_api.GridFunction(dp0_space, coefficients=coeffs)


def _restrict_neumann_to_nonzero_support(grid, neumann_fun):
    """Return a DP0 GridFunction containing only nonzero Neumann elements.

    The Helmholtz SLP is linear in the prescribed Neumann data. Assembling
    columns for rigid elements whose coefficient is exactly zero cannot affect
    the RHS, so a restricted DP0 domain is mathematically identical and avoids
    most SLP work for a small source patch.
    """
    import bempp_cl.api as bempp_api

    coefficients = np.asarray(neumann_fun.coefficients)
    support_elements = np.flatnonzero(coefficients != 0)
    if support_elements.size == 0 or support_elements.size == coefficients.size:
        return neumann_fun.space, neumann_fun

    restricted_space = bempp_api.function_space(
        grid,
        "DP",
        0,
        support_elements=support_elements,
    )
    restricted_fun = bempp_api.GridFunction(
        restricted_space,
        coefficients=coefficients[support_elements],
    )
    return restricted_space, restricted_fun


def _build_p1_to_dp0_projection(p1_space, dp0_space):
    """Sparse P1→DP0 coefficient projection: vertex pressures → per-element
    averages. Each output DP0 DOF is the integral average of the P1 function
    on that triangle. Local multipliers keep excluded boundary DOFs at zero on
    open surfaces.
    """
    import scipy.sparse as sp
    n_dp0 = dp0_space.global_dof_count
    n_p1 = p1_space.global_dof_count
    local2global = np.array(p1_space.local2global)  # (n_elements, 3)
    local_multipliers = np.asarray(p1_space.local_multipliers)
    if local2global.shape[0] != n_dp0:
        raise RuntimeError(
            f"local2global has {local2global.shape[0]} rows but DP0 space "
            f"has {n_dp0} DOFs"
        )
    rows = np.repeat(np.arange(n_dp0, dtype=np.int64), 3)
    cols = local2global.flatten().astype(np.int64)
    data = local_multipliers.flatten().astype(np.float64) / 3.0
    return sp.csr_matrix((data, (rows, cols)), shape=(n_dp0, n_p1))


def _build_robin_contribution(
    single_layer_matrix: NDArray,
    beta_vec: NDArray,
    p1_to_dp0,
    k: complex,
) -> NDArray:
    """Build ``i*k*V*diag(beta)*M`` without densifying sparse ``M``."""
    import scipy.sparse as sp

    weighted_projection = sp.diags(beta_vec) @ p1_to_dp0
    return (1j * k) * (single_layer_matrix @ weighted_projection)


def _build_neumann_coefficients(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    dtype: type,
    excluded_tags=(),
    grid=None,
    source_axis: NDArray[np.float64] | None = None,
    axial_element_scale: NDArray[np.float64] | None = None,
) -> NDArray:
    """Build source Neumann coefficients, optionally excluding physical tags.

    Honors ``config.source_motion == "axial"`` (rigid piston) when ``grid`` is
    supplied; the default normal path is unchanged.
    """
    coeffs = np.zeros(dp0_space.global_dof_count, dtype=dtype)
    air_density = config.air_density
    if config.source_motion == SourceMotion.AXIAL:
        if axial_element_scale is None:
            if grid is None or source_axis is None:
                raise ValueError("axial source motion requires a grid and source_axis")
            axial_element_scale = _build_axial_element_scale(
                grid, physical_tags, config.velocity_sources.keys(), source_axis
            )
    for tag, weight in config.velocity_sources.items():
        if tag in excluded_tags:
            continue
        mask = physical_tags == tag
        if not np.any(mask):
            continue
        v_n = weight
        if config.velocity_mode is VelocityMode.ACCELERATION:
            # Under this package's e^{-i omega t} convention, a*cos(omega t)
            # integrates to v = a/(-i omega), so q = i rho omega v = -rho a
            # (momentum: dp/dn = -rho a_n). Matches metal-bem; cross-validated
            # against ABEC3 absolute pressure (2026-07-09).
            v_n = weight / (-1j * omega) if omega > 0 else 0.0
        g_val = 1j * air_density * omega * v_n
        dofs = np.where(mask)[0]
        if axial_element_scale is not None:
            coeffs[dofs] = g_val * axial_element_scale[dofs]
        else:
            coeffs[dofs] = g_val
    return coeffs


def _assemble_and_solve_impedance(
    grid,
    p1_space,
    dp0_space,
    physical_tags: NDArray[np.int32],
    k: complex,
    omega: float,
    config: SolveConfig,
    op_kwargs_low: dict,
    source_axis: NDArray[np.float64] | None = None,
    axial_element_scale: NDArray[np.float64] | None = None,
):
    """Direct (non-iterative) solve for the BIE with Robin BCs.

    Substitutes ∂p/∂n = i·k·β·p on the impedance tags directly into the
    BIE, producing a single linear system

        [(K − ½I) − i·k · V · diag(β) · M_proj] p = V · g_drv

    where M_proj is the P1→DP0 vertex-averaging projection and g_drv is the
    Neumann data with driver/aperture sources only (zero on impedance tags).
    This avoids the Picard fixed point, which diverges near interior
    Dirichlet eigenvalues of the enclosed cavities (chamber/port modes).
    The −i·k·V·M_β term also acts as a complex shift that pushes those
    fictitious eigenvalues off the real axis, making the system more
    robust than the rigid solve.
    """
    import bempp_cl.api as bempp_api
    import scipy.linalg

    timings: dict[str, float] = {}

    if config.formulation is BIEFormulation.BURTON_MILLER:
        raise NotImplementedError(
            "Robin/impedance BC + Burton-Miller formulation is not "
            "supported. Use formulation=STANDARD or COMPLEX_K."
        )

    start = time.perf_counter()
    identity = bempp_api.operators.boundary.sparse.identity(
        p1_space, p1_space, p1_space,
    )
    dlp = bempp_api.operators.boundary.helmholtz.double_layer(
        p1_space, p1_space, p1_space, k, **op_kwargs_low,
    )
    slp = bempp_api.operators.boundary.helmholtz.single_layer(
        dp0_space, p1_space, p1_space, k, **op_kwargs_low,
    )
    timings["operator_construction_s"] = time.perf_counter() - start

    A_op = dlp - 0.5 * identity
    start = time.perf_counter()
    dlp.weak_form()
    timings["dlp_assembly_s"] = time.perf_counter() - start
    start = time.perf_counter()
    slp.weak_form()
    timings["slp_assembly_s"] = time.perf_counter() - start
    start = time.perf_counter()
    A_mat = bempp_api.as_matrix(A_op.weak_form())
    V_mat = bempp_api.as_matrix(slp.weak_form())
    timings["operator_materialization_s"] = time.perf_counter() - start

    start = time.perf_counter()
    dtype_solve = np.complex128
    A_mat = np.asarray(A_mat, dtype=dtype_solve)
    V_mat = np.asarray(V_mat, dtype=dtype_solve)

    # β as DP0 coefficients (zero on non-impedance tags)
    n_dp0 = dp0_space.global_dof_count
    beta_vec = np.zeros(n_dp0, dtype=dtype_solve)
    for tag, beta in config.impedance_sources.items():
        mask = physical_tags == tag
        beta_vec[np.where(mask)[0]] = complex(beta)

    # P1 → DP0 projection (sparse)
    M_proj = _build_p1_to_dp0_projection(p1_space, dp0_space)

    # Robin contribution: −i·k · V · diag(β) · M_proj  (n_p1 × n_p1 dense)
    robin = _build_robin_contribution(V_mat, beta_vec, M_proj, k)
    lhs_full = A_mat - robin

    # RHS = V · g_drv, with g_drv zero on impedance tags
    g_drv = _build_neumann_coefficients(
        dp0_space,
        physical_tags,
        omega,
        config,
        dtype_solve,
        excluded_tags=config.impedance_sources,
        grid=grid,
        source_axis=source_axis,
        axial_element_scale=axial_element_scale,
    )
    rhs_vec = V_mat @ g_drv
    timings["system_build_s"] = time.perf_counter() - start

    start = time.perf_counter()
    with limited_blas_threads(1):
        p_coeffs = scipy.linalg.solve(lhs_full, rhs_vec)
    timings["linear_solve_s"] = time.perf_counter() - start
    start = time.perf_counter()
    p_surface = bempp_api.GridFunction(p1_space, coefficients=p_coeffs)

    # Build a Neumann GridFunction for downstream far-field evaluation:
    # the TRUE Neumann data including the Robin contribution.
    # n_total = g_drv + i·k·β·p_dp0
    p_dp0 = M_proj @ p_coeffs                          # (n_dp0,)
    g_robin = (1j * k) * beta_vec * p_dp0              # (n_dp0,)
    # The Robin direct solve runs in complex128; only downcast the true
    # Neumann data when the configured precision asks for it.
    neumann_dtype = np.complex64 if config.precision == "single" else np.complex128
    neumann_total = (g_drv + g_robin).astype(neumann_dtype)
    neumann_fun = bempp_api.GridFunction(
        dp0_space, coefficients=neumann_total,
    )
    timings["solution_wrapping_s"] = time.perf_counter() - start

    return (
        p_surface,
        neumann_fun,
        None,
        True,
        timings,
    )  # iterations = None (direct solve)


def _assemble_and_solve(
    grid,
    p1_space,
    dp0_space,
    neumann_fun,
    k: complex,
    k_real: float,
    config: SolveConfig,
    n_triangles: int,
    op_kwargs_low: dict,
    op_kwargs_high: dict,
):
    """Assemble BIE operators and solve the linear system."""
    import bempp_cl.api as bempp_api

    timings: dict[str, float] = {}
    start = time.perf_counter()
    identity = bempp_api.operators.boundary.sparse.identity(
        p1_space, p1_space, p1_space,
    )
    dlp = bempp_api.operators.boundary.helmholtz.double_layer(
        p1_space, p1_space, p1_space, k, **op_kwargs_low,
    )
    slp = bempp_api.operators.boundary.helmholtz.single_layer(
        dp0_space, p1_space, p1_space, k, **op_kwargs_low,
    )
    timings["operator_construction_s"] = time.perf_counter() - start

    if config.formulation is BIEFormulation.BURTON_MILLER:
        hyp = bempp_api.operators.boundary.helmholtz.hypersingular(
            p1_space, p1_space, p1_space, k, **op_kwargs_high,
        )
        adlp = bempp_api.operators.boundary.helmholtz.adjoint_double_layer(
            dp0_space, p1_space, p1_space, k, **op_kwargs_high,
        )
        rhs_identity = bempp_api.operators.boundary.sparse.identity(
            dp0_space, p1_space, p1_space,
        )
        coupling = 1j / k
        lhs = 0.5 * identity - dlp - coupling * (-hyp)
        start = time.perf_counter()
        slp.weak_form()
        timings["slp_assembly_s"] = time.perf_counter() - start
        start = time.perf_counter()
        adlp.weak_form()
        timings["adlp_assembly_s"] = time.perf_counter() - start
        start = time.perf_counter()
        rhs = (-slp - coupling * (adlp + 0.5 * rhs_identity)) * neumann_fun
        timings["rhs_apply_s"] = time.perf_counter() - start
        start = time.perf_counter()
        dlp.weak_form()
        timings["dlp_assembly_s"] = time.perf_counter() - start
        start = time.perf_counter()
        hyp.weak_form()
        timings["hyp_assembly_s"] = time.perf_counter() - start
    else:
        # STANDARD or COMPLEX_K (complex k is already baked into the
        # wavenumber — the BIE structure is the same as STANDARD)
        lhs = dlp - 0.5 * identity
        start = time.perf_counter()
        slp.weak_form()
        timings["slp_assembly_s"] = time.perf_counter() - start
        start = time.perf_counter()
        rhs = slp * neumann_fun
        timings["rhs_apply_s"] = time.perf_counter() - start
        start = time.perf_counter()
        dlp.weak_form()
        timings["dlp_assembly_s"] = time.perf_counter() - start

    start = time.perf_counter()
    lhs.weak_form()
    timings["lhs_materialization_s"] = time.perf_counter() - start
    solver_choice = _choose_solver(config, n_triangles)
    iterations = None
    converged = True

    start = time.perf_counter()
    # LAPACK LU and the GMRES inner products both scale negatively with BLAS
    # threads at these matrix sizes; the limit is scoped so assembly and any
    # other array work in the process keep their threads. See _blas_threads.
    with limited_blas_threads(1):
        if solver_choice is LinearSolver.LU:
            p_surface = bempp_api.linalg.lu(lhs, rhs)
        else:
            result = bempp_api.linalg.gmres(
                lhs, rhs,
                tol=config.gmres_tol,
                restart=config.gmres_restart,
                maxiter=config.gmres_max_iter,
                use_strong_form=True,
                return_iteration_count=True,
            )
            p_surface, info, iterations = result
            converged = info == 0
            if info != 0:
                logger.warning(
                    "GMRES did not converge (info=%d) at k=%.4f", info, k_real,
                )
    timings["linear_solve_s"] = time.perf_counter() - start

    return p_surface, iterations, converged, timings


def _evaluate_far_field(
    p1_space,
    dp0_space,
    p_surface,
    neumann_fun,
    k_real: float,
    obs_points: NDArray[np.float64],
    op_kwargs: dict,
    restrict_neumann_space: bool = True,
) -> NDArray[np.complex128]:
    """Evaluate pressure at observation points via representation formula.

    p(x) = DLP_pot[p_surface](x) - SLP_pot[neumann_fun](x)
    """
    import bempp_cl.api as bempp_api

    # obs_points: (N, 3) → bempp wants (3, N)
    pts = np.ascontiguousarray(obs_points.T, dtype=np.float64)
    if restrict_neumann_space:
        dp0_space, neumann_fun = _restrict_neumann_to_nonzero_support(
            dp0_space.grid, neumann_fun,
        )

    dlp_pot = bempp_api.operators.potential.helmholtz.double_layer(
        p1_space, pts, k_real, **op_kwargs,
    )
    slp_pot = bempp_api.operators.potential.helmholtz.single_layer(
        dp0_space, pts, k_real, **op_kwargs,
    )

    pressure = (dlp_pot * p_surface - slp_pot * neumann_fun).flatten()
    return pressure


def _compute_impedance(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    source_tag: int = 2,
) -> complex:
    """Area-weighted average source pressure in pascals per unit drive."""
    return compute_surface_pressure_avg(
        grid,
        p_surface,
        physical_tags,
        p1_space,
        [source_tag],
    )[source_tag]


def compute_surface_pressure_avg(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    tags: list[int],
) -> dict[int, complex]:
    """Area-weighted average surface pressure on each tag.

    Returns a dict mapping tag -> complex average pressure.
    """
    areas = np.array(grid.volumes)
    coeffs = np.asarray(p_surface.coefficients)
    local2global = np.array(p1_space.local2global)
    local_multipliers = np.asarray(p1_space.local_multipliers)
    local_coeffs = coeffs[local2global] * local_multipliers

    result = {}
    for tag in tags:
        mask = physical_tags == tag
        elem_indices = np.where(mask)[0]
        if len(elem_indices) == 0:
            result[tag] = 0.0 + 0.0j
            continue

        elem_areas = areas[elem_indices]
        p_avg_per_elem = np.mean(local_coeffs[elem_indices], axis=1)

        total_area = np.sum(elem_areas)
        if total_area < 1e-30:
            result[tag] = 0.0 + 0.0j
        else:
            result[tag] = complex(np.sum(p_avg_per_elem * elem_areas) / total_area)

    return result


def solve_single_frequency(
    grid,
    physical_tags: NDArray[np.int32],
    frequency_hz: float,
    config: SolveConfig,
    p1_space=None,
    dp0_space=None,
    source_axis: NDArray[np.float64] | None = None,
    closed_mesh_validated: bool = False,
    axial_element_scale: NDArray[np.float64] | None = None,
    symmetry_context=None,
) -> FrequencyResult:
    """Solve the BEM problem at a single frequency.

    When ``config.impedance_sources`` is non-empty, dispatches to
    ``_assemble_and_solve_impedance``, which folds the Robin BC
    ∂p/∂n = i·k·β·p directly into the BIE for a single LU solve.
    """
    reject_unsupported_native_symmetry(config)
    if config.channels:
        # Channels are a frequency-dependent complex weight on each tag's
        # prescribed normal velocity, and nothing downstream of the boundary
        # condition needs to know they exist -- so resolve them here, once,
        # into the same velocity_sources mapping the rigid path already
        # consumes. The tag set is unchanged, which keeps the impedance tag,
        # the axial element scale, and the source-tag validation intact.
        from .channels import resolve_channel_drives

        config = replace(
            config,
            velocity_sources=resolve_channel_drives(
                config.channels, config.velocity_sources, frequency_hz,
            ),
            channels=[],
        )
    if uses_image_assembly(config) and config.impedance_sources:
        raise NotImplementedError(
            "native half/quarter symmetry with Robin impedance boundaries "
            "is not implemented yet"
        )
    if config.source_motion == SourceMotion.AXIAL and source_axis is None:
        if config.frame_override is not None:
            source_axis = np.asarray(config.frame_override.axis, dtype=np.float64)
        else:
            from .observation import infer_frame

            source_axis = infer_frame(
                grid,
                physical_tags,
                source_tag=min(config.velocity_sources.keys(), default=2),
                origin_at=config.observation.origin,
                # Without the plane the reduced-mesh PCA is quadrant-biased,
                # and an axis picking up an x/y component makes the per-face
                # ``n_hat . axis`` drive lose mirror invariance -- which the
                # even extension in expand_neumann_coefficients assumes.
                # _resolve_frame on the sweep path already passes this.
                symmetry_plane=config.native_symmetry_plane,
            ).axis
    t0 = time.perf_counter()
    phase_timings: dict[str, float] = {}

    start = time.perf_counter()
    if uses_image_assembly(config) and symmetry_context is None:
        from .symmetry import build_symmetry_context

        symmetry_context = build_symmetry_context(
            grid,
            physical_tags,
            config.native_symmetry_plane,
            ground_plane=config.ground_plane,
        )
    if config.require_closed_mesh and not closed_mesh_validated:
        validation_grid = (
            symmetry_context.expanded_grid
            if symmetry_context is not None
            else grid
        )
        _require_closed_surface(
            np.asarray(validation_grid.vertices, dtype=np.float64).T,
            np.asarray(validation_grid.elements, dtype=np.int32).T,
        )
    if p1_space is None or dp0_space is None:
        if symmetry_context is None:
            p1_space, dp0_space = _setup_function_spaces(grid)
        else:
            p1_space = symmetry_context.reduced_p1
            dp0_space = symmetry_context.reduced_dp0
    phase_timings["function_spaces_s"] = time.perf_counter() - start

    omega = 2.0 * np.pi * frequency_hz
    k_real = omega / SPEED_OF_SOUND

    # Wavenumber: real or complex-shifted
    if config.formulation is BIEFormulation.COMPLEX_K:
        k = k_real * (1 + 1j * config.complex_k_shift)
    else:
        k = k_real

    start = time.perf_counter()
    backend = resolve_assembly_backend(config).effective_backend
    if config.adaptive_quadrature:
        slp_dlp_order, hyp_adlp_order, quadrature_diag = (
            _select_regular_quadrature(config, k_real, _p90_element_length(grid))
        )
        if slp_dlp_order != config.slp_dlp_quadrature:
            logger.info(
                "%.1f Hz: k*h=%.2f in [%.2f, %.2f], regular quadrature "
                "order %d (base %d)",
                frequency_hz, quadrature_diag["kh"],
                quadrature_diag["kh_min"], quadrature_diag["kh_max"],
                slp_dlp_order, config.slp_dlp_quadrature,
            )
    else:
        slp_dlp_order = config.slp_dlp_quadrature
        hyp_adlp_order = config.hyp_adlp_quadrature
        quadrature_diag = {}
    op_kwargs_low = _operator_kwargs(
        backend, config.precision, config.opencl_device, slp_dlp_order,
        config.slp_dlp_singular_quadrature,
        vectorization_mode=getattr(config, "vectorization_mode", "auto"),
    )
    op_kwargs_high = _operator_kwargs(
        backend, config.precision, config.opencl_device, hyp_adlp_order,
        vectorization_mode=getattr(config, "vectorization_mode", "auto"),
    )
    phase_timings["backend_setup_s"] = time.perf_counter() - start

    n_tris = grid.number_of_elements

    # Impedance / Robin BC: filter to tags actually present in the mesh.
    active_impedance: dict[int, complex] = {}
    if config.impedance_sources:
        for tag, beta in config.impedance_sources.items():
            mask = physical_tags == tag
            if not np.any(mask):
                logger.warning(
                    "impedance_sources references tag %d but mesh has no "
                    "elements with that tag — skipping", tag,
                )
                continue
            active_impedance[tag] = beta

    field_pressure = None
    field_neumann = None
    if active_impedance:
        impedance_config = replace(config, impedance_sources=active_impedance)
        (
            p_surface,
            neumann_fun,
            iterations,
            converged,
            core_timings,
        ) = _assemble_and_solve_impedance(
            grid,
            p1_space,
            dp0_space,
            physical_tags,
            k,
            omega,
            impedance_config,
            op_kwargs_low,
            source_axis=source_axis,
            axial_element_scale=axial_element_scale,
        )
    else:
        start = time.perf_counter()
        neumann_fun = _build_neumann_data(
            dp0_space,
            physical_tags,
            omega,
            config,
            config.precision,
            grid=grid,
            source_axis=source_axis,
            axial_element_scale=axial_element_scale,
        )
        phase_timings["neumann_data_s"] = time.perf_counter() - start
        solve_dp0_space = dp0_space
        solve_neumann_fun = neumann_fun
        # Under symmetry the restriction happens on the expanded grid inside
        # assemble_and_solve_symmetry, which needs the unrestricted reduced
        # trace; restricting here would build a function space per frequency
        # and then discard it.
        if config.restrict_neumann_space and symmetry_context is None:
            start = time.perf_counter()
            solve_dp0_space, solve_neumann_fun = (
                _restrict_neumann_to_nonzero_support(grid, neumann_fun)
            )
            phase_timings["neumann_restriction_s"] = (
                time.perf_counter() - start
            )
        if symmetry_context is None:
            p_surface, iterations, converged, core_timings = _assemble_and_solve(
                grid, p1_space, solve_dp0_space, solve_neumann_fun,
                k, k_real, config, n_tris, op_kwargs_low, op_kwargs_high,
            )
        else:
            from .symmetry import assemble_and_solve_symmetry

            (
                p_surface,
                field_pressure,
                field_neumann,
                iterations,
                converged,
                core_timings,
            ) = assemble_and_solve_symmetry(
                symmetry_context,
                neumann_fun,
                k,
                config,
                op_kwargs_low,
            )
    phase_timings.update(core_timings)

    start = time.perf_counter()
    impedance = _compute_impedance(
        grid, p_surface, physical_tags, p1_space,
        source_tag=min(config.velocity_sources.keys(), default=2),
    )
    phase_timings["impedance_s"] = time.perf_counter() - start

    elapsed = time.perf_counter() - t0
    phase_timings["total_s"] = elapsed
    if active_impedance:
        logger.info(
            "%.1f Hz: solved in %.2fs (direct Robin BC)",
            frequency_hz, elapsed,
        )
    else:
        logger.info(
            "%.1f Hz: solved in %.2fs%s",
            frequency_hz, elapsed,
            f" ({iterations} iters)" if iterations is not None else " (LU)",
        )

    return FrequencyResult(
        frequency_hz=frequency_hz,
        pressure_on_surface=p_surface,
        neumann_data=neumann_fun,
        impedance=impedance,
        iterations=iterations,
        converged=converged,
        timing_s=elapsed,
        phase_timings=phase_timings,
        field_pressure_on_surface=field_pressure,
        field_neumann_data=field_neumann,
        regular_quadrature=quadrature_diag,
    )
