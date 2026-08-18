from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import SolveConfig


@dataclass
class MeshInfo:
    n_vertices: int
    n_triangles: int
    physical_groups: dict[int, str]
    bounding_box_m: tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass
class SolveResult:
    r"""Bempp BEM solve output.

    Array dimensions use ``F`` for frequency count, ``P`` for observation
    plane count, and ``N`` for points or angles per plane. Complex values use
    the solver's :math:`e^{-i\omega t}` phase convention, which is the same
    convention used by hornlab-metal-bem.

    ``surface_pressure_complex`` is the optional P1 pressure trace with shape
    ``(F, n_p1_dofs)``. ``surface_neumann_complex`` is the optional *total*
    DP0 outward-normal derivative ``dp/dn`` with shape ``(F, n_dp0_dofs)``;
    its Robin faces include ``q_driver + i*k*beta*p_dp0`` using the same
    P1-to-DP0 projection as the assembled system.
    """

    frequencies_hz: NDArray[np.float64]

    # (F, P, N_angles) — complex pressure at every observation point
    pressure_complex: NDArray[np.complex128]

    # (F, P, N_angles) — SPL in dB, normalised on-axis = 0 dB
    spl_db: NDArray[np.float64]

    # (F,) — raw area-weighted average pressure on the source tag.
    # This follows hornlab-metal-bem and is not normalized to rho*c.
    impedance: NDArray[np.complex128]

    observation_angles_deg: NDArray[np.float64]
    observation_points: NDArray[np.float64]
    observation_planes: list[str]

    config: SolveConfig
    mesh_info: MeshInfo
    timings: dict[str, float] = field(default_factory=dict)
    solver_log: list[dict] = field(default_factory=list)

    # Area-weighted average surface pressure per velocity-source tag.
    # tag -> (F,) complex array. Populated when velocity_sources has tags.
    surface_pressure_avg: dict[int, NDArray[np.complex128]] | None = None

    # Optional solved P1 pressure coefficients, shape (F, n_p1_dofs), in
    # Bempp space DOF order and the e^{-i omega t} phase convention.
    surface_pressure_complex: NDArray[np.complex128] | None = None

    # Optional total outward-normal derivative q = dp/dn on DP0, shape
    # (F, n_dp0_dofs), in Bempp space DOF order and the e^{-i omega t}
    # convention. Robin faces include q_driver + i*k*beta*p_dp0.
    surface_neumann_complex: NDArray[np.complex128] | None = None

    # Optional frame-relative spherical pressure field. Pressure is (F, M),
    # while theta/phi are flattened theta-major coordinate arrays of length M.
    sphere_pressure_complex: NDArray[np.complex128] | None = None
    sphere_theta_deg: NDArray[np.float64] | None = None
    sphere_phi_deg: NDArray[np.float64] | None = None

    @property
    def directivity_db(self) -> NDArray[np.float64]:
        """hornlab_metal_bem-compatible name for spl_db."""
        return self.spl_db
