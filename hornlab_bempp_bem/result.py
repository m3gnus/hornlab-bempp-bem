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


@dataclass
class ChannelBasisResult:
    r"""Per-channel pressure at unit drive, plus the synthesis that sums it.

    One :class:`SolveResult` per channel, each solved with only that channel's
    source tags driven and the channel's own DSP set aside. Because the
    boundary integral equation is linear in the prescribed Neumann data, any
    weighted sum of these rows is the solve that would have resulted from
    driving the channels together -- so a crossover can be re-tuned by
    re-summing rather than re-solving.

    ``pressure_complex`` is ``(C, F, P, N)``, channel-major, in the solver's
    :math:`e^{-i\omega t}` convention.
    """

    channel_names: tuple[str, ...]
    channels: tuple  # tuple[Channel, ...]; typed loosely to avoid a cycle
    frequencies_hz: NDArray[np.float64]

    # (C, F, P, N_angles) complex pressure per channel at unit channel drive.
    pressure_complex: NDArray[np.complex128]
    # (C, F) raw area-weighted source pressure per channel at unit drive.
    impedance: NDArray[np.complex128]

    observation_angles_deg: NDArray[np.float64]
    observation_points: NDArray[np.float64]
    observation_planes: list[str]

    config: SolveConfig
    mesh_info: MeshInfo
    results: tuple = ()  # tuple[SolveResult, ...], one per channel
    timings: dict[str, float] = field(default_factory=dict)

    # (C, F, M) per-channel spherical field, when observation.sphere_grid is set.
    sphere_pressure_complex: NDArray[np.complex128] | None = None
    sphere_theta_deg: NDArray[np.float64] | None = None
    sphere_phi_deg: NDArray[np.float64] | None = None

    # tag -> (C, F) area-weighted surface pressure on every driven tag, for
    # every channel. Cross terms are included: a channel excites all tags, not
    # only the ones it drives.
    surface_pressure_avg: dict[int, NDArray[np.complex128]] | None = None

    def synthesize(
        self,
        channels=None,
        *,
        flat_target: bool = False,
        flat_target_plane: int = 0,
        flat_target_angle_index: int | None = None,
    ) -> SolveResult:
        """Sum the basis under ``channels`` and return an ordinary result.

        ``channels`` defaults to the channels the basis was solved with; pass a
        re-tuned list, in the same order, to change the crossover without
        re-solving. ``flat_target`` divides each channel by the magnitude of
        its own pressure at the reference point first, so the synthesized
        response is the crossover's own shape rather than the crossover
        multiplied by each driver's native rolloff -- BEAT's
        ``flat_target_normalization``. The reference defaults to the on-axis
        angle of the first observation plane, which is where BEAT's default
        ``flat_target_reference_angle_deg=0`` lands.
        """
        from .channels import flat_target_corrections, synthesize_channel_pressure
        from .sweep import _normalized_spl_db

        channels = tuple(self.channels if channels is None else channels)
        if len(channels) != len(self.channels):
            raise ValueError(
                f"synthesize needs {len(self.channels)} channels in the basis "
                f"order {self.channel_names}, got {len(channels)}"
            )
        corrections = None
        if flat_target:
            if flat_target_angle_index is None:
                flat_target_angle_index = int(
                    np.argmin(np.abs(self.observation_angles_deg))
                )
            corrections = flat_target_corrections(
                self.pressure_complex[
                    :, :, flat_target_plane, flat_target_angle_index
                ],
            )

        pressure = synthesize_channel_pressure(
            self.pressure_complex, channels, self.frequencies_hz,
            corrections=corrections,
        )
        impedance = synthesize_channel_pressure(
            self.impedance, channels, self.frequencies_hz,
            corrections=corrections,
        )
        sphere = None
        if self.sphere_pressure_complex is not None:
            sphere = synthesize_channel_pressure(
                self.sphere_pressure_complex, channels, self.frequencies_hz,
                corrections=corrections,
            )
        surface = None
        if self.surface_pressure_avg is not None:
            surface = {
                tag: synthesize_channel_pressure(
                    values, channels, self.frequencies_hz,
                    corrections=corrections,
                )
                for tag, values in self.surface_pressure_avg.items()
            }
        on_axis = int(np.argmin(np.abs(self.observation_angles_deg)))
        return SolveResult(
            frequencies_hz=self.frequencies_hz,
            pressure_complex=pressure,
            spl_db=_normalized_spl_db(pressure, on_axis),
            impedance=impedance,
            observation_angles_deg=self.observation_angles_deg,
            observation_points=self.observation_points,
            observation_planes=list(self.observation_planes),
            config=self.config,
            mesh_info=self.mesh_info,
            timings=dict(self.timings),
            surface_pressure_avg=surface,
            sphere_pressure_complex=sphere,
            sphere_theta_deg=self.sphere_theta_deg,
            sphere_phi_deg=self.sphere_phi_deg,
        )
