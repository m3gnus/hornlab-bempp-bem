r"""Multi-channel drive with crossovers.

A *channel* is one amplifier output: a level, a polarity, a delay, an optional
highpass and an optional lowpass, driving one or more of the mesh's velocity
source tags. Its complex drive coefficient at a frequency multiplies the
prescribed normal velocity of every tag it owns, so a two-way horn can be
solved with its real crossover in place instead of one driver at a time.

Conventions
-----------
This package uses :math:`e^{-i\omega t}`, so a causal analog filter is
evaluated at :math:`s = -i\omega` and a pure delay of :math:`\tau` is
:math:`e^{+i\omega\tau}`. Both are the opposite sign from the
:math:`e^{+i\omega t}` form printed in most filter texts; evaluating at
:math:`s = +i\omega` here would conjugate every crossover's phase, which is
invisible in a magnitude plot and wrong the moment two channels sum. The
responses below are therefore the complex conjugates of the standard-audio
ones, which is exactly what ``numpy.conjugate(scipy.signal.freqs(...))``
produces, and matches BEAT's ``butterworth_response``.

Everything here is a pure function of frequency. None of it touches the BEM
solve: a channel's coefficient enters as a complex weight on the Neumann data,
and because the boundary integral equation is linear in that data, driving
every channel at once in a single solve is exactly equivalent to solving each
channel alone and summing. ``solve_channel_basis`` uses that equivalence in
the other direction, to make crossovers editable without re-solving.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CrossoverType = Literal["none", "lowpass", "highpass"]
CrossoverFilter = Literal["butterworth", "linkwitz_riley"]

CROSSOVER_TYPES = ("none", "lowpass", "highpass")
CROSSOVER_FILTERS = ("butterworth", "linkwitz_riley")
# Matches BEAT's validate_crossover_config: Butterworth 1/2/4/6, and
# Linkwitz-Riley 2/4/6 built as a squared Butterworth section of half the
# order. LR6 therefore uses a 3rd-order section even though a bare 3rd-order
# Butterworth is not itself selectable -- BEAT does the same.
BUTTERWORTH_ORDERS = (1, 2, 4, 6)
LINKWITZ_RILEY_ORDERS = (2, 4, 6)


def _butterworth_poles(order: int) -> NDArray[np.complex128]:
    r"""Normalised Butterworth poles :math:`e^{i(\pi/2 + (2k-1)\pi/2N)}`.

    All on the unit circle in the left half plane; for ``order == 1`` the
    single pole is exactly ``-1``.
    """
    k = np.arange(1, int(order) + 1, dtype=np.float64)
    return np.exp(1j * (np.pi / 2.0 + (2.0 * k - 1.0) * np.pi / (2.0 * order)))


def butterworth_response(
    crossover_type: str,
    order: int,
    cutoff_hz: float,
    frequency_hz: NDArray[np.float64] | float,
) -> NDArray[np.complex128] | complex:
    r"""Analog Butterworth response in this package's phase convention.

    Lowpass  :math:`\prod_k (-\omega_c p_k) / (s - \omega_c p_k)`, unity at DC.
    Highpass :math:`\prod_k s / (s - \omega_c / p_k)`, unity as
    :math:`\omega \to \infty`.

    with :math:`s = -i\omega` and :math:`\omega_c = 2\pi f_c`. The poles are
    scaled by :math:`\omega_c` rather than ``s`` being normalised by it, which
    is the same arrangement BEAT uses and keeps the two bit-comparable.
    """
    scalar = np.isscalar(frequency_hz)
    omega = 2.0 * np.pi * np.atleast_1d(
        np.asarray(frequency_hz, dtype=np.float64)
    )
    omega_c = 2.0 * np.pi * float(cutoff_hz)
    s = -1j * omega
    poles = _butterworth_poles(order)
    response = np.ones(omega.shape, dtype=np.complex128)
    for pole in poles:
        if crossover_type == "lowpass":
            scaled = omega_c * pole
            response = response * ((-scaled) / (s - scaled))
        else:
            scaled = omega_c / pole
            response = response * (s / (s - scaled))
    return complex(response[0]) if scalar else response


@dataclass(frozen=True)
class Crossover:
    """One highpass or lowpass section.

    ``type="none"`` is a flat unity response and ignores every other field, so
    an unused slot needs no special casing.
    """

    type: CrossoverType = "none"
    frequency_hz: float | None = None
    filter: CrossoverFilter = "butterworth"
    order: int = 1

    def __post_init__(self) -> None:
        if self.type not in CROSSOVER_TYPES:
            raise ValueError(
                "crossover type must be one of: " + ", ".join(CROSSOVER_TYPES)
            )
        if self.type == "none":
            return
        if self.filter not in CROSSOVER_FILTERS:
            raise ValueError(
                "crossover filter must be one of: "
                + ", ".join(CROSSOVER_FILTERS)
            )
        try:
            cutoff = float(self.frequency_hz)
        except (TypeError, ValueError, OverflowError):
            cutoff = float("nan")
        if not isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "crossover frequency_hz must be finite and greater than zero"
            )
        object.__setattr__(self, "frequency_hz", cutoff)
        allowed = (
            LINKWITZ_RILEY_ORDERS
            if self.filter == "linkwitz_riley"
            else BUTTERWORTH_ORDERS
        )
        if self.order not in allowed:
            raise ValueError(
                f"{self.filter} crossover order must be one of: "
                + ", ".join(str(value) for value in allowed)
            )

    def response(
        self, frequency_hz: NDArray[np.float64] | float,
    ) -> NDArray[np.complex128] | complex:
        """Complex response at ``frequency_hz``, unity for ``type='none'``."""
        if self.type == "none":
            if np.isscalar(frequency_hz):
                return 1.0 + 0.0j
            return np.ones(np.shape(frequency_hz), dtype=np.complex128)
        if self.filter == "linkwitz_riley":
            # LR(N) is a Butterworth section of order N/2, cascaded twice.
            section = butterworth_response(
                self.type, self.order // 2, self.frequency_hz, frequency_hz,
            )
            return section * section
        return butterworth_response(
            self.type, self.order, self.frequency_hz, frequency_hz,
        )


@dataclass(frozen=True)
class Channel:
    """One amplifier output driving a set of velocity source tags.

    ``sources`` names the mesh physical tags this channel drives. Give a
    sequence of tags for equal drive, or a mapping ``tag -> gain`` for a
    relative offset between radiators on the same channel. The gain may be
    complex, which expresses a fixed relative phase between two drivers on one
    channel -- something BEAT's radiator model cannot represent, since its
    per-radiator content is a real ``velocity_offset_db`` only.

    The effective drive for a tag is
    ``SolveConfig.velocity_sources[tag] * sources[tag] * channel.drive(f)``:
    ``velocity_sources`` keeps meaning the tag's own unit drive, and the
    channel supplies everything frequency dependent.
    """

    name: str
    sources: Mapping[int, complex] | Sequence[int]
    level_db: float = 0.0
    polarity: int = 1
    delay_ms: float = 0.0
    highpass: Crossover = field(default_factory=Crossover)
    lowpass: Crossover = field(default_factory=Crossover)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("channel name must be a non-empty string")
        if isinstance(self.sources, Mapping):
            gains = {int(tag): complex(gain) for tag, gain in self.sources.items()}
        else:
            try:
                tags = list(self.sources)
            except TypeError:
                raise ValueError(
                    f"channel {self.name!r} sources must be a sequence of tags "
                    "or a mapping of tag to gain"
                ) from None
            gains = {int(tag): 1.0 + 0.0j for tag in tags}
            if len(gains) != len(tags):
                raise ValueError(
                    f"channel {self.name!r} lists a source tag more than once"
                )
        if not gains:
            raise ValueError(f"channel {self.name!r} drives no source tags")
        for tag, gain in gains.items():
            if not (isfinite(gain.real) and isfinite(gain.imag)):
                raise ValueError(
                    f"channel {self.name!r} gain for tag {tag} must be finite"
                )
        object.__setattr__(self, "sources", gains)

        if self.polarity not in (1, -1):
            raise ValueError(
                f"channel {self.name!r} polarity must be 1 or -1"
            )
        for name in ("level_db", "delay_ms"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            if not isfinite(value):
                raise ValueError(f"channel {self.name!r} {name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("highpass", "lowpass"):
            section = getattr(self, name)
            if not isinstance(section, Crossover):
                raise ValueError(
                    f"channel {self.name!r} {name} must be a Crossover"
                )

    @property
    def source_tags(self) -> tuple[int, ...]:
        return tuple(sorted(self.sources))

    def drive(
        self, frequency_hz: NDArray[np.float64] | float,
    ) -> NDArray[np.complex128] | complex:
        r"""Complex drive coefficient: polarity, level, delay, crossovers.

        ``polarity * 10^(level_db/20) * e^{+i\omega\,\tau} * H_{hp} * H_{lp}``.
        """
        omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
        level = 10.0 ** (self.level_db / 20.0)
        delay = np.exp(1j * omega * (self.delay_ms / 1000.0))
        response = (
            float(self.polarity)
            * level
            * delay
            * self.highpass.response(frequency_hz)
            * self.lowpass.response(frequency_hz)
        )
        return complex(response) if np.isscalar(frequency_hz) else response


def validate_channels(
    channels: Sequence[Channel] | None,
    velocity_sources: Mapping[int, complex],
) -> None:
    """Reject channel sets that would silently drop or double-drive a tag.

    Every channel source must be a driven tag, and no tag may belong to two
    channels -- two channels on one tag would need two prescribed normal
    velocities on the same faces, which the boundary condition cannot express.
    A driven tag owned by no channel is also refused: it would keep its raw
    ``velocity_sources`` weight and radiate flat underneath the crossovers,
    which is never what a crossover configuration means.
    """
    if not channels:
        return
    names = [channel.name for channel in channels]
    if len(set(names)) != len(names):
        raise ValueError("channel names must be unique")
    owner: dict[int, str] = {}
    for channel in channels:
        for tag in channel.source_tags:
            if tag not in velocity_sources:
                raise ValueError(
                    f"channel {channel.name!r} drives tag {tag}, which is not "
                    f"in velocity_sources {sorted(velocity_sources)}"
                )
            if tag in owner:
                raise ValueError(
                    f"tag {tag} is driven by both channel {owner[tag]!r} and "
                    f"channel {channel.name!r}; a face can carry only one "
                    "prescribed normal velocity"
                )
            owner[tag] = channel.name
    unowned = sorted(set(velocity_sources) - set(owner))
    if unowned:
        raise ValueError(
            f"velocity_sources tag(s) {unowned} belong to no channel; every "
            "driven tag must be assigned once channels are in use"
        )


def resolve_channel_drives(
    channels: Sequence[Channel],
    velocity_sources: Mapping[int, complex],
    frequency_hz: float,
) -> dict[int, complex]:
    """Per-tag complex drive at one frequency.

    ``velocity_sources[tag] * channel_gain[tag] * channel.drive(f)``.
    """
    drives: dict[int, complex] = {}
    for channel in channels:
        coefficient = channel.drive(float(frequency_hz))
        for tag, gain in channel.sources.items():
            drives[tag] = complex(velocity_sources[tag]) * gain * coefficient
    return drives


def synthesize_channel_pressure(
    channel_pressure: NDArray[np.complex128],
    channels: Sequence[Channel],
    frequencies_hz: NDArray[np.float64],
    *,
    corrections: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Sum a per-channel pressure basis under the channels' own DSP.

    ``channel_pressure`` has shape ``(n_channels, n_frequencies, ...)`` and
    holds each channel's pressure at unit drive -- solved once, then reusable.
    Because the boundary integral equation is linear in the prescribed Neumann
    data, this weighted sum is the solve that would result from driving every
    channel simultaneously, to solver tolerance.

    ``corrections`` is an optional real per-channel, per-frequency gain of
    shape ``(n_channels, n_frequencies)``; see ``flat_target_corrections``.
    """
    pressure = np.asarray(channel_pressure, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if pressure.ndim < 2:
        raise ValueError(
            "channel_pressure must have shape (n_channels, n_frequencies, ...)"
        )
    if pressure.shape[0] != len(channels):
        raise ValueError(
            f"channel_pressure has {pressure.shape[0]} channel rows but "
            f"{len(channels)} channels were given"
        )
    if pressure.shape[1] != frequencies.shape[0]:
        raise ValueError(
            f"channel_pressure has {pressure.shape[1]} frequency columns but "
            f"{frequencies.shape[0]} frequencies were given"
        )
    weights = np.stack(
        [np.asarray(channel.drive(frequencies), dtype=np.complex128)
         for channel in channels],
        axis=0,
    )
    if corrections is not None:
        corrections = np.asarray(corrections, dtype=np.float64)
        if corrections.shape != weights.shape:
            raise ValueError(
                "corrections must have shape (n_channels, n_frequencies), got "
                f"{corrections.shape}"
            )
        weights = weights * corrections
    # Broadcast the (channel, frequency) weights over the trailing point axes.
    weights = weights.reshape(weights.shape + (1,) * (pressure.ndim - 2))
    return np.sum(pressure * weights, axis=0)


def flat_target_corrections(
    channel_pressure: NDArray[np.complex128],
    *,
    floor_pa: float = 1.0e-12,
) -> NDArray[np.float64]:
    r"""Per-channel gains that flatten each channel's own reference response.

    ``1 / |p_c(f)|`` at the reference observation point, so the synthesized
    response is the crossover's own shape rather than the crossover multiplied
    by each driver's native rolloff. ``channel_pressure`` is
    ``(n_channels, n_frequencies)`` complex pressure at that one point.
    Magnitudes at or below ``floor_pa`` get a correction of ``1`` rather than
    a division blow-up, matching BEAT's guard.

    BEAT interpolates its reference point from the horizontal polar arc at
    ``flat_target_reference_angle_deg``; here the caller picks the point, which
    is the same operation without the arc-interpolation step.
    """
    magnitude = np.abs(np.asarray(channel_pressure, dtype=np.complex128))
    corrections = np.ones(magnitude.shape, dtype=np.float64)
    usable = np.isfinite(magnitude) & (magnitude > float(floor_pa))
    corrections[usable] = 1.0 / magnitude[usable]
    return corrections


__all__ = [
    "BUTTERWORTH_ORDERS",
    "CROSSOVER_FILTERS",
    "CROSSOVER_TYPES",
    "LINKWITZ_RILEY_ORDERS",
    "Channel",
    "Crossover",
    "butterworth_response",
    "flat_target_corrections",
    "resolve_channel_drives",
    "synthesize_channel_pressure",
    "validate_channels",
]
