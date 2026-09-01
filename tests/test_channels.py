r"""Multi-channel drive with crossovers.

Two things need proving and they are proved differently. The filter algebra is
checked against ``scipy.signal.butter``, an independent implementation, and
against textbook invariants that pin the *phase* convention rather than only
the magnitude -- a crossover evaluated at :math:`s = +i\omega` instead of
:math:`s = -i\omega` has an identical magnitude plot and sums wrongly. The
solver integration is checked by superposition through the real assembler:
one solve with both channels driven must equal the sum of two solves driven
one at a time.
"""
from __future__ import annotations

import numpy as np
import pytest

from hornlab_bempp_bem.channels import (
    Channel,
    Crossover,
    butterworth_response,
    flat_target_corrections,
    resolve_channel_drives,
    synthesize_channel_pressure,
    validate_channels,
)
from hornlab_bempp_bem.config import LinearSolver, ObservationConfig, SolveConfig

_FREQUENCIES = np.geomspace(20.0, 20_000.0, 200)
_CUTOFF_HZ = 1200.0


# ---------------------------------------------------------------------------
# Filter algebra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["lowpass", "highpass"])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_butterworth_matches_scipy_in_this_packages_phase_convention(kind, order):
    r"""``conj(H(+i\omega))`` is ``H(-i\omega)`` for real coefficients.

    This is the check that would fail if the s-plane substitution were the
    textbook :math:`s = +i\omega`: the magnitudes would still agree and only
    the sign of every phase would flip.
    """
    scipy_signal = pytest.importorskip("scipy.signal")

    numerator, denominator = scipy_signal.butter(
        order, 2.0 * np.pi * _CUTOFF_HZ, btype=kind, analog=True,
    )
    _, reference = scipy_signal.freqs(
        numerator, denominator, worN=2.0 * np.pi * _FREQUENCIES,
    )
    ours = butterworth_response(kind, order, _CUTOFF_HZ, _FREQUENCIES)
    assert np.abs(ours - np.conjugate(reference)).max() < 1.0e-12
    # And explicitly not the unconjugated one, except where the response is
    # real anyway.
    assert np.abs(ours - reference).max() > 1.0e-3


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_butterworth_is_minus_three_db_at_cutoff_and_power_complementary(order):
    lowpass = butterworth_response("lowpass", order, _CUTOFF_HZ, _CUTOFF_HZ)
    assert abs(abs(lowpass) - 2.0 ** -0.5) < 1.0e-12

    low = butterworth_response("lowpass", order, _CUTOFF_HZ, _FREQUENCIES)
    high = butterworth_response("highpass", order, _CUTOFF_HZ, _FREQUENCIES)
    power = np.abs(low) ** 2 + np.abs(high) ** 2
    assert np.abs(power - 1.0).max() < 1.0e-12


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_butterworth_asymptotic_slope(order):
    """``20*log10(2)`` per octave per order -- 6.0206, not the shorthand 6."""
    low, high = _CUTOFF_HZ / 8192.0, _CUTOFF_HZ / 4096.0
    slope = 20.0 * np.log10(
        abs(butterworth_response("highpass", order, _CUTOFF_HZ, high))
        / abs(butterworth_response("highpass", order, _CUTOFF_HZ, low))
    )
    assert abs(slope - 20.0 * np.log10(2.0) * order) < 1.0e-6


@pytest.mark.parametrize("order", [2, 4, 6])
def test_linkwitz_riley_is_a_squared_butterworth_section(order):
    section = butterworth_response(
        "lowpass", order // 2, _CUTOFF_HZ, _FREQUENCIES,
    )
    crossover = Crossover("lowpass", _CUTOFF_HZ, "linkwitz_riley", order)
    assert np.abs(crossover.response(_FREQUENCIES) - section * section).max() < 1e-14
    # -6 dB at cutoff, the defining LR property.
    assert abs(abs(crossover.response(_CUTOFF_HZ)) - 0.5) < 1.0e-12


@pytest.mark.parametrize("order", [2, 4, 6])
def test_linkwitz_riley_branches_recombine_to_an_allpass(order):
    """LR4 sums flat; LR2 and LR6 need one branch inverted.

    Getting the phase convention wrong does not break this -- both branches
    conjugate together -- so it is a check on the LR construction rather than
    on the s-plane substitution.
    """
    low = Crossover("lowpass", _CUTOFF_HZ, "linkwitz_riley", order)
    high = Crossover("highpass", _CUTOFF_HZ, "linkwitz_riley", order)
    summed = low.response(_FREQUENCIES) + high.response(_FREQUENCIES)
    differenced = low.response(_FREQUENCIES) - high.response(_FREQUENCIES)
    flat = summed if order % 4 == 0 else differenced
    assert np.abs(np.abs(flat) - 1.0).max() < 1.0e-12


def test_delay_is_a_true_delay_under_the_negative_time_convention():
    r"""Under :math:`e^{-i\omega t}`, ``x(t - \tau)`` transforms to
    :math:`X(\omega)e^{+i\omega\tau}`. The opposite sign is a time *advance*,
    which is the mistake this pins."""
    channel = Channel("d", [2], delay_ms=1.0)
    omega = 2.0 * np.pi * _FREQUENCIES
    assert np.allclose(channel.drive(_FREQUENCIES), np.exp(1j * omega * 1.0e-3))

    # Read the delay back off the phase slope. Linear and closely spaced, so
    # the wrapped phase advances well under pi per step and unwrap can follow;
    # the log grid above steps past pi over most of its range.
    dense = np.linspace(20.0, 2000.0, 400)
    dense_omega = 2.0 * np.pi * dense
    phase = np.unwrap(np.angle(channel.drive(dense)))
    # d(phase)/d(omega) = +tau here; the group delay is its negative.
    assert abs(np.mean(np.gradient(phase, dense_omega)) - 1.0e-3) < 1.0e-9


def test_level_and_polarity():
    assert abs(
        abs(Channel("g", [2], level_db=-6.0).drive(1000.0)) - 10.0 ** (-6.0 / 20.0)
    ) < 1.0e-12
    plain = Channel("p", [2]).drive(1000.0)
    inverted = Channel("p", [2], polarity=-1).drive(1000.0)
    assert inverted == -plain


def test_crossover_and_channel_validation():
    assert Crossover().response(1000.0) == 1.0 + 0.0j
    with pytest.raises(ValueError, match="crossover type"):
        Crossover("bandpass", 100.0)
    with pytest.raises(ValueError, match="frequency_hz"):
        Crossover("lowpass", None)
    with pytest.raises(ValueError, match="order must be one of"):
        Crossover("lowpass", 100.0, "butterworth", 3)
    with pytest.raises(ValueError, match="order must be one of"):
        Crossover("lowpass", 100.0, "linkwitz_riley", 1)
    with pytest.raises(ValueError, match="polarity"):
        Channel("c", [2], polarity=0)
    with pytest.raises(ValueError, match="drives no source tags"):
        Channel("c", [])


def test_channels_must_partition_the_driven_tags():
    sources = {2: 1.0, 3: 0.5}
    with pytest.raises(ValueError, match="not in velocity_sources"):
        validate_channels([Channel("a", [2, 9])], sources)
    with pytest.raises(ValueError, match="only one prescribed normal velocity"):
        validate_channels([Channel("a", [2]), Channel("b", [2, 3])], sources)
    with pytest.raises(ValueError, match="belong to no channel"):
        validate_channels([Channel("a", [2])], sources)
    with pytest.raises(ValueError, match="names must be unique"):
        validate_channels(
            [Channel("a", [2]), Channel("a", [3])], sources,
        )
    validate_channels([Channel("a", [2]), Channel("b", [3])], sources)


def test_config_rejects_a_channel_set_that_does_not_partition():
    with pytest.raises(ValueError, match="belong to no channel"):
        SolveConfig(
            velocity_sources={2: 1.0, 3: 1.0},
            channels=[Channel("a", [2])],
        )
    assert SolveConfig().channels == []


def test_resolved_drive_multiplies_the_tags_own_weight():
    channels = [Channel("lf", {2: 1.0, 3: 0.25}, level_db=-6.0)]
    drives = resolve_channel_drives(channels, {2: 2.0, 3: 4.0}, 1000.0)
    gain = 10.0 ** (-6.0 / 20.0)
    assert abs(drives[2] - 2.0 * 1.0 * gain) < 1.0e-15
    assert abs(drives[3] - 4.0 * 0.25 * gain) < 1.0e-15


def test_flat_target_corrections_normalise_each_channel():
    pressure = np.array([[2.0 + 0.0j, 0.0 + 4.0j], [1e-30 + 0j, 1.0 + 1.0j]])
    corrections = flat_target_corrections(pressure)
    normalised = np.abs(pressure * corrections)
    assert np.allclose(normalised[0], 1.0)
    assert np.isclose(normalised[1, 1], 1.0)
    # Below the floor the correction is 1, not a division blow-up.
    assert corrections[1, 0] == 1.0


def test_synthesis_rejects_mismatched_shapes():
    basis = np.zeros((2, 3, 4), dtype=np.complex128)
    channels = [Channel("a", [2]), Channel("b", [3])]
    with pytest.raises(ValueError, match="channel rows"):
        synthesize_channel_pressure(basis, channels[:1], np.zeros(3))
    with pytest.raises(ValueError, match="frequency columns"):
        synthesize_channel_pressure(basis, channels, np.zeros(5))


# ---------------------------------------------------------------------------
# Through the solver
# ---------------------------------------------------------------------------

def _require_bempp_cpu():
    try:
        import bempp_cl.api  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"bempp-cl unavailable: {exc}")
    try:
        from hornlab_bempp_bem import configure_opencl

        configure_opencl("cpu")
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"OpenCL CPU runtime unavailable: {exc}")


_SOLVE_FREQUENCIES = np.array([400.0, 1200.0, 3500.0])
_SOURCES = {2: 1.0, 3: 0.6}


def _two_way_sphere():
    """One sphere, two disjoint driven patches: a low and a high channel."""
    import bempp_cl.api as bempp_api

    from hornlab_bempp_bem.mesh import LoadedMesh
    from hornlab_bempp_bem.result import MeshInfo

    grid = bempp_api.shapes.regular_sphere(3)
    verts = np.asarray(grid.vertices, dtype=np.float64).T * 0.2
    tris = np.asarray(grid.elements, dtype=np.int64).T
    tags = np.ones(tris.shape[0], dtype=np.int32)
    tags[:48] = 2
    tags[48:96] = 3
    reduced = bempp_api.Grid(
        np.ascontiguousarray(verts.T),
        np.ascontiguousarray(tris.T.astype(np.uint32)),
    )
    return LoadedMesh(
        grid=reduced,
        physical_tags=tags,
        info=MeshInfo(
            verts.shape[0], tris.shape[0], {2: "lf", 3: "hf"},
            (verts.min(axis=0), verts.max(axis=0)),
        ),
    )


def _frame():
    from hornlab_bempp_bem.observation import ObservationFrame

    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.zeros(3),
        source_center=np.zeros(3),
    )


def _observation():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(16, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(1.5, 3.0, (16, 1))
    return ObservationConfig(
        planes=["horizontal"],
        custom_points={"horizontal": points},
        angle_count=16,
    )


def _config(**overrides):
    base = dict(
        velocity_sources=dict(_SOURCES),
        solver=LinearSolver.LU,
        precision="double",
        assembly_backend="opencl",
        observation=_observation(),
    )
    base.update(overrides)
    return SolveConfig(**base)


def _crossed_channels():
    return [
        Channel(
            "lf", [2],
            lowpass=Crossover("lowpass", 1500.0, "linkwitz_riley", 4),
        ),
        Channel(
            "hf", [3], level_db=-2.5, polarity=-1, delay_ms=0.15,
            highpass=Crossover("highpass", 1500.0, "linkwitz_riley", 4),
        ),
    ]


def _run(mesh, config, frequencies=_SOLVE_FREQUENCIES):
    from hornlab_bempp_bem.sweep import run_sweep_serial

    return run_sweep_serial(mesh, frequencies, _frame(), config)


@pytest.mark.slow
def test_no_channels_is_bit_identical_to_the_plain_drive():
    _require_bempp_cpu()

    mesh = _two_way_sphere()
    without = _run(mesh, _config())
    empty = _run(mesh, _config(channels=[]))
    assert np.array_equal(without.pressure_complex, empty.pressure_complex)
    assert np.array_equal(without.impedance, empty.impedance)


@pytest.mark.slow
def test_one_solve_with_channels_equals_the_sum_of_per_channel_solves():
    """The linearity the whole feature rests on, through the real assembler.

    Each reference solve prescribes the exact complex weight the channel
    resolves to and uses no channel machinery at all, so this catches a wrong
    weight, a dropped tag, or a real-valued truncation of the coefficient.
    """
    _require_bempp_cpu()

    mesh = _two_way_sphere()
    channels = _crossed_channels()
    combined = _run(mesh, _config(channels=channels))

    total = np.zeros_like(combined.pressure_complex)
    total_impedance = np.zeros_like(combined.impedance)
    for channel in channels:
        pressure_rows, impedance_rows = [], []
        for frequency in _SOLVE_FREQUENCIES:
            weight = channel.drive(float(frequency))
            driven = {
                tag: (
                    _SOURCES[tag] * channel.sources[tag] * weight
                    if tag in channel.sources
                    else 0.0
                )
                for tag in _SOURCES
            }
            one = _run(
                mesh, _config(velocity_sources=driven),
                np.array([frequency]),
            )
            pressure_rows.append(one.pressure_complex[0])
            impedance_rows.append(one.impedance[0])
        total += np.stack(pressure_rows, axis=0)
        total_impedance += np.asarray(impedance_rows)

    relative = np.abs(combined.pressure_complex - total) / np.abs(total)
    assert relative.max() < 1.0e-10, relative.max()
    assert np.abs(combined.impedance - total_impedance).max() < 1.0e-14


@pytest.mark.slow
def test_channel_basis_resynthesizes_the_combined_solve_and_retunes_for_free():
    _require_bempp_cpu()

    import hornlab_bempp_bem as package

    mesh = _two_way_sphere()
    channels = _crossed_channels()
    combined = _run(mesh, _config(channels=channels))

    basis = package.solve_channel_basis(
        mesh, _config(channels=channels), _SOLVE_FREQUENCIES,
    )
    assert basis.channel_names == ("lf", "hf")
    assert basis.pressure_complex.shape == (2,) + combined.pressure_complex.shape

    synthesized = basis.synthesize()
    relative = (
        np.abs(synthesized.pressure_complex - combined.pressure_complex)
        / np.abs(combined.pressure_complex)
    )
    assert relative.max() < 1.0e-10, relative.max()
    assert np.abs(synthesized.spl_db - combined.spl_db).max() < 1.0e-9
    assert np.abs(synthesized.impedance - combined.impedance).max() < 1.0e-14

    # A different crossover, re-summed from the SAME basis, must match a fresh
    # solve at that setting. This is the whole point of the basis.
    retuned = [
        Channel(
            "lf", [2], level_db=-1.0,
            lowpass=Crossover("lowpass", 900.0, "butterworth", 2),
        ),
        Channel(
            "hf", [3], level_db=1.5, delay_ms=0.4,
            highpass=Crossover("highpass", 900.0, "butterworth", 2),
        ),
    ]
    fresh = _run(mesh, _config(channels=retuned))
    resynthesized = basis.synthesize(retuned)
    relative = (
        np.abs(resynthesized.pressure_complex - fresh.pressure_complex)
        / np.abs(fresh.pressure_complex)
    )
    assert relative.max() < 1.0e-10, relative.max()

    with pytest.raises(ValueError, match="channels in the basis order"):
        basis.synthesize(retuned[:1])


@pytest.mark.slow
def test_flat_target_normalises_each_channel_at_the_reference_point():
    """Each channel's own contribution at the reference point becomes unity.

    It normalises magnitude only -- the channel keeps its acoustic phase, which
    is why the synthesized sum is not simply the sum of the channel drives.
    """
    _require_bempp_cpu()

    import hornlab_bempp_bem as package

    basis = package.solve_channel_basis(
        _two_way_sphere(),
        _config(channels=_crossed_channels()),
        _SOLVE_FREQUENCIES,
    )
    on_axis = int(np.argmin(np.abs(basis.observation_angles_deg)))
    reference = basis.pressure_complex[:, :, 0, on_axis]
    corrections = flat_target_corrections(reference)
    assert np.allclose(np.abs(reference * corrections), 1.0)

    # Without normalisation the raw magnitudes are nowhere near unity, so the
    # check above is not vacuous.
    assert np.abs(reference).max() < 0.1

    flattened = basis.synthesize(flat_target=True)
    expected = np.sum(
        np.stack(
            [channel.drive(basis.frequencies_hz) for channel in basis.channels],
            axis=0,
        )
        * corrections
        * reference,
        axis=0,
    )
    assert np.abs(flattened.pressure_complex[:, 0, on_axis] - expected).max() < 1e-12
