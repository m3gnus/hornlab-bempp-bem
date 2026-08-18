"""Post-solve exterior-field evaluation from retained surface traces."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .backends import resolve_fastest_backend
from .bie import _evaluate_far_field, _operator_kwargs, _setup_function_spaces
from .config import VECTORIZATION_MODES
from .mesh import LoadedMesh


def _trace_grid(mesh_or_grid):
    """Return a Bempp grid and its element tags, when available."""
    if isinstance(mesh_or_grid, LoadedMesh):
        return mesh_or_grid.grid, mesh_or_grid.physical_tags
    if not (
        hasattr(mesh_or_grid, "vertices")
        and hasattr(mesh_or_grid, "elements")
        and hasattr(mesh_or_grid, "number_of_elements")
    ):
        raise TypeError("mesh_or_grid must be a LoadedMesh or bempp Grid")
    return mesh_or_grid, None


def _trace_coefficients(
    values: NDArray[Any],
    expected_count: int,
    name: str,
) -> NDArray[np.complex128]:
    coefficients = np.asarray(values)
    if coefficients.ndim != 1 or coefficients.shape[0] != expected_count:
        raise ValueError(
            f"{name} must have shape ({expected_count},) in Bempp DOF order"
        )
    if not np.issubdtype(coefficients.dtype, np.number):
        raise TypeError(f"{name} must contain numeric coefficients")
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} must contain only finite values")
    return coefficients


def evaluate_exterior_from_traces(
    mesh_or_grid,
    frequency_hz: float,
    k_real: float,
    pressure_p1: NDArray[Any],
    neumann_dp0: NDArray[Any],
    points_xyz: NDArray[Any],
    *,
    symmetry_plane: str | None = None,
    assembly_backend: Literal["auto", "opencl", "numba"] = "auto",
    precision: Literal["single", "double"] = "double",
    opencl_device: Literal["cpu", "gpu"] = "cpu",
    vectorization_mode: str = "auto",
    restrict_neumann_space: bool = True,
) -> NDArray[np.complex128]:
    r"""Evaluate complex exterior pressure from one frequency's traces.

    Parameters
    ----------
    mesh_or_grid:
        The same :class:`~hornlab_bempp_bem.mesh.LoadedMesh` or Bempp ``Grid``
        used for the solve. Reconstructing a geometrically equivalent grid with
        different element or vertex ordering does not preserve the contract.
    frequency_hz, k_real:
        Frequency metadata and the real acoustic wavenumber in radians/metre.
        The solver also uses real ``k_real`` for exterior evaluation after a
        ``COMPLEX_K`` boundary solve.
    pressure_p1:
        Complex P1 pressure with shape ``(n_p1_dofs,)``.
    neumann_dp0:
        Complex *total* DP0 trace ``dp/dn`` with shape ``(n_dp0_dofs,)``.
        Robin faces must include ``q_driver + i*k*beta*p_dp0``.
    points_xyz:
        Exterior coordinates in metres with shape ``(N, 3)``.
    symmetry_plane:
        The solve's native symmetry plane, if any: ``"yz"``, ``"xz"``, or
        ``"yz+xz"``. Coefficients remain in the reduced mesh's DOF order and
        are expanded through the same image map used by the solve.
    assembly_backend, precision, opencl_device, vectorization_mode:
        Bempp potential-operator settings, in double precision. The default
        ``"auto"`` picks the fastest backend the machine can actually run,
        matching :class:`~hornlab_bempp_bem.config.SolveConfig`: OpenCL when a
        device initializes, Numba otherwise. Naming one outright takes it as
        given. Prefer ``"auto"`` -- the two agree to machine precision, while
        Numba costs about three times the one-off per-process kernel compile
        and twice the steady-state evaluation.
    restrict_neumann_space:
        Restrict the single-layer potential to nonzero DP0 support, matching
        the solve-time field evaluator's default.

    Returns
    -------
    numpy.ndarray
        Complex pressure with shape ``(N,)`` in the solver's
        :math:`e^{-i\omega t}` convention, identical to hornlab-metal-bem.

    Notes
    -----
    DOF ordering is part of the artifact contract. The arrays are coefficient
    vectors, not vertex/element samples: they must be passed back with the same
    grid connectivity and ordering used to create the original Bempp P1/DP0
    spaces. This function rebuilds those spaces and asserts their DOF counts;
    no same-length check can detect an already permuted coefficient vector.
    """
    if not (math.isfinite(float(frequency_hz)) and float(frequency_hz) > 0.0):
        raise ValueError("frequency_hz must be finite and positive")
    if not (math.isfinite(float(k_real)) and float(k_real) > 0.0):
        raise ValueError("k_real must be finite and positive")
    if symmetry_plane not in {None, "yz", "xz", "yz+xz"}:
        raise ValueError("symmetry_plane must be None, 'yz', 'xz', or 'yz+xz'")
    if assembly_backend not in {"auto", "opencl", "numba"}:
        raise ValueError("assembly_backend must be 'auto', 'opencl' or 'numba'")
    if precision not in {"single", "double"}:
        raise ValueError("precision must be 'single' or 'double'")
    if vectorization_mode not in VECTORIZATION_MODES:
        raise ValueError(
            "vectorization_mode must be one of: "
            + ", ".join(VECTORIZATION_MODES)
        )
    if not isinstance(restrict_neumann_space, bool):
        raise ValueError("restrict_neumann_space must be a boolean")

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points_xyz must be non-empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xyz must contain only finite values")

    import bempp_cl.api as bempp_api

    grid, physical_tags = _trace_grid(mesh_or_grid)
    if symmetry_plane is None:
        p1_space, dp0_space = _setup_function_spaces(grid)
        p_coefficients = _trace_coefficients(
            pressure_p1, p1_space.global_dof_count, "pressure_p1",
        )
        q_coefficients = _trace_coefficients(
            neumann_dp0, dp0_space.global_dof_count, "neumann_dp0",
        )
        pressure_fun = bempp_api.GridFunction(
            p1_space, coefficients=p_coefficients,
        )
        neumann_fun = bempp_api.GridFunction(
            dp0_space, coefficients=q_coefficients,
        )
    else:
        from .symmetry import build_symmetry_context

        if physical_tags is None:
            physical_tags = np.zeros(grid.number_of_elements, dtype=np.int32)
        context = build_symmetry_context(
            grid, np.asarray(physical_tags, dtype=np.int32), symmetry_plane,
        )
        p_coefficients = _trace_coefficients(
            pressure_p1,
            context.reduced_p1.global_dof_count,
            "pressure_p1",
        )
        q_coefficients = _trace_coefficients(
            neumann_dp0,
            context.reduced_dp0.global_dof_count,
            "neumann_dp0",
        )
        active_pressure = p_coefficients[context.active_reduced_dofs]
        pressure_fun = context.wrap_expanded_pressure(active_pressure)
        neumann_fun = bempp_api.GridFunction(
            context.full_dp0,
            coefficients=context.expand_neumann_coefficients(q_coefficients),
        )
        p1_space = context.full_p1
        dp0_space = context.full_dp0

    # Resolve after the geometry work above, so a machine with no OpenCL device
    # pays the probe only on a call that was going to assemble anyway.
    resolution = resolve_fastest_backend(
        assembly_backend, opencl_device=opencl_device,
    )
    op_kwargs = _operator_kwargs(
        resolution.effective_backend,
        precision,
        opencl_device,
        vectorization_mode=vectorization_mode,
    )
    pressure = _evaluate_far_field(
        p1_space,
        dp0_space,
        pressure_fun,
        neumann_fun,
        float(k_real),
        points,
        op_kwargs,
        restrict_neumann_space,
    )
    return np.asarray(pressure, dtype=np.complex128).reshape(points.shape[0])
