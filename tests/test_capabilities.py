"""Capability declarations exercised against the real config and solver guards."""

from dataclasses import fields
from importlib import import_module
import json
import subprocess
import sys

import pytest

import hornlab_bempp_bem as package
from hornlab_bempp_bem.config import SolveConfig
from hornlab_bempp_bem.config import reject_unsupported_native_symmetry as image_guard

capabilities = package.capabilities


def test_schema_exports_and_request_field_coverage():
    report = capabilities()
    assert report["schema"] == "hornlab-bem-capabilities"
    assert report["schema_version"] == package.CAPABILITY_SCHEMA_VERSION == 1
    assert report["request_schema_version"] == package.REQUEST_SCHEMA_VERSION == 1
    assert report["package"] == "hornlab-bempp-bem"
    assert json.loads(json.dumps(report)) == report
    actual = {item.name for item in fields(SolveConfig) if item.init}
    assert set(report["request_fields"]) == actual
    assert len(report["request_fields"]) == len(actual)
    assert {
        "capabilities",
        "CAPABILITY_SCHEMA_VERSION",
        "REQUEST_SCHEMA_VERSION",
    } <= set(package.__all__)
    assert report["conventions"]["time_convention"] == "exp(-i*omega*t)"


def test_distribution_metadata_and_missing_metadata(monkeypatch):
    module = import_module("hornlab_bempp_bem.capabilities")
    calls = []

    def version(name):
        calls.append(name)
        return "1.2.3"

    monkeypatch.setattr(module, "version", version)
    assert capabilities()["package_version"] == "1.2.3"
    assert calls == ["hornlab-bempp-bem"]

    def missing(name):
        raise module.PackageNotFoundError(name)

    monkeypatch.setattr(module, "version", missing)
    assert capabilities()["package_version"] is None


def test_report_is_a_fresh_snapshot():
    original = capabilities()
    changed = capabilities()
    changed["request_fields"].clear()
    changed["features"]["source_motion"]["values"].clear()
    changed["features"]["ground_plane"]["requires"]["formulation"].clear()
    changed["features"]["infinite_baffle"]["refuses"][0].clear()
    assert capabilities() == original


def test_handshake_does_not_load_numerical_backends():
    script = """
import sys
class ForbidBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bempp_cl', 'pyopencl'}:
            raise AssertionError('handshake loaded numerical backend')
sys.meta_path.insert(0, ForbidBackends())
from hornlab_bempp_bem import capabilities
assert capabilities()['features']['ground_plane']['supported']
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


@pytest.mark.parametrize(
    "feature,field",
    [
        ("source_motion", "source_motion"),
        ("formulation", "formulation"),
        ("native_symmetry", "native_symmetry_plane"),
        ("assembly_backend", "assembly_backend"),
    ],
)
def test_enumerated_choices_pass_config_and_guard(feature, field):
    for value in capabilities()["features"][feature]["values"]:
        config = SolveConfig(**{field: value})
        image_guard(config)
    with pytest.raises((ValueError, NotImplementedError)):
        config = SolveConfig(**{field: "invalid"})
        image_guard(config)


def test_ground_compositions_and_formulations_match_actual_guards():
    detail = capabilities()["features"]["ground_plane"]
    compositions = (
        detail["composes_with"][0]["values"] if detail["composes_with"] else []
    )
    for ground in detail["values"]:
        for symmetry in [
            None,
            *capabilities()["features"]["native_symmetry"]["values"],
        ]:
            for formulation in capabilities()["features"]["formulation"]["values"]:
                try:
                    image_guard(
                        SolveConfig(
                            ground_plane=ground,
                            native_symmetry_plane=symmetry,
                            formulation=formulation,
                        )
                    )
                except (ValueError, NotImplementedError):
                    allowed = False
                else:
                    allowed = True
                expected = formulation in detail["requires"]["formulation"] and (
                    symmetry is None
                    or {"native_symmetry_plane": symmetry, "ground_plane": ground}
                    in compositions
                )
                assert allowed == expected


def ib_guard(config):
    from hornlab_bempp_bem.infinite_baffle import _validate_coupled_infinite_baffle

    _validate_coupled_infinite_baffle(None, config, None)


def robin_guard(config):
    from hornlab_bempp_bem.bie import _assemble_and_solve_impedance

    _assemble_and_solve_impedance(None, None, None, None, 1.0, 1.0, config, {})


def parallel_guard(config):
    from hornlab_bempp_bem.sweep import run_sweep_parallel

    run_sweep_parallel(None, [100.0], None, config, 2)


def case(
    feature,
    identifier,
    related,
    kwargs,
    guard=image_guard,
    error=NotImplementedError,
    match="",
):
    return (feature, identifier, related, kwargs, guard, error, match)


CASES = [
    case(
        "infinite_baffle",
        "ib_burton_miller",
        ["formulation"],
        dict(aperture_tag=3, formulation="burton_miller"),
        ib_guard,
        match="Burton-Miller",
    ),
    case(
        "infinite_baffle",
        "ib_surface_traces",
        ["return_surface_traces"],
        dict(aperture_tag=3, return_surface_traces=True),
        ib_guard,
        match="return_surface_traces",
    ),
    case(
        "return_surface_traces",
        "traces_infinite_baffle",
        ["aperture_tag"],
        dict(aperture_tag=3, return_surface_traces=True),
        ib_guard,
        match="return_surface_traces",
    ),
    *[
        case(
            "infinite_baffle",
            "ib_symmetry",
            ["native_symmetry_plane"],
            dict(aperture_tag=3, native_symmetry_plane=plane),
            ib_guard,
            match="full-domain",
        )
        for plane in ("xy", "yz", "xz", "yz+xz")
    ],
    case(
        "infinite_baffle",
        "ib_robin",
        ["impedance_sources"],
        dict(aperture_tag=3, impedance_sources={1: 0.1}),
        ib_guard,
        match="Robin",
    ),
    case(
        "formulation",
        "bm_robin",
        ["impedance_sources"],
        dict(formulation="burton_miller", impedance_sources={1: 0.1}),
        robin_guard,
        match="Robin/impedance",
    ),
    case(
        "formulation",
        "bm_infinite_baffle",
        ["aperture_tag"],
        dict(aperture_tag=3, formulation="burton_miller"),
        ib_guard,
        match="Burton-Miller",
    ),
    *[
        case(
            "formulation",
            "bm_ground",
            ["ground_plane"],
            dict(formulation="burton_miller", ground_plane=plane),
            match="Burton-Miller",
        )
        for plane in ("xy", "yz", "xz")
    ],
    *[
        case(
            "formulation",
            "bm_symmetry",
            ["native_symmetry_plane"],
            dict(formulation="burton_miller", native_symmetry_plane=plane),
            match="Burton-Miller",
        )
        for plane in ("yz", "xz", "yz+xz")
    ],
    *[
        case(
            "native_symmetry",
            "symmetry_xy",
            ["native_symmetry_plane"],
            dict(native_symmetry_plane="xy"),
            match="legacy 'xy'",
        )
    ],
    *[
        case(
            "native_symmetry",
            "symmetry_burton_miller",
            ["formulation"],
            dict(formulation="burton_miller", native_symmetry_plane=plane),
            match="Burton-Miller",
        )
        for plane in ("yz", "xz", "yz+xz")
    ],
    *[
        case(
            "native_symmetry",
            "symmetry_robin",
            ["impedance_sources"],
            dict(impedance_sources={1: 0.1}, native_symmetry_plane=plane),
            match="Robin",
        )
        for plane in ("yz", "xz", "yz+xz")
    ],
    *[
        case(
            "ground_plane",
            "ground_burton_miller",
            ["formulation"],
            dict(formulation="burton_miller", ground_plane=plane),
            match="Burton-Miller",
        )
        for plane in ("xy", "yz", "xz")
    ],
    *[
        case(
            "ground_plane",
            "ground_robin",
            ["impedance_sources"],
            dict(impedance_sources={1: 0.1}, ground_plane=plane),
            match="Robin",
        )
        for plane in ("xy", "yz", "xz")
    ],
    *[
        case(
            "ground_plane",
            "ground_same_plane",
            ["native_symmetry_plane"],
            dict(ground_plane=plane, native_symmetry_plane=symmetry),
            error=ValueError,
            match="also declared",
        )
        for symmetry in ("xy", "yz", "xz", "yz+xz")
        for plane in symmetry.split("+")
    ],
    case(
        "on_frequency_result",
        "callback_parallel",
        ["workers"],
        dict(workers=2, on_frequency_result=lambda *args: None),
        parallel_guard,
        ValueError,
        "on_frequency_result",
    ),
    case(
        "workers",
        "workers_callback",
        ["on_frequency_result"],
        dict(workers=2, on_frequency_result=lambda *args: None),
        parallel_guard,
        ValueError,
        "on_frequency_result",
    ),
]


def test_bempp_full_domain_and_serial_prerequisites():
    report = capabilities()["features"]
    assert report["infinite_baffle"]["requires"] == {
        "formulation": ["standard", "complex_k"],
        "native_symmetry_plane": [None],
    }
    assert report["infinite_baffle"]["composes_with"] == []
    assert report["on_frequency_result"]["requires"] == {"workers": [1]}


# Each declaration must have a test case and vice versa; removing a refusal or
# adding an untested declaration fails coverage. Multiple cases exercise every
# enum arm where a refusal applies to several modes.
def test_every_declared_refusal_has_guard_cases():
    declared = {
        (name, refusal["id"]): refusal["request_fields"]
        for name, feature in capabilities()["features"].items()
        for refusal in feature["refuses"]
    }
    expected = {(name, identifier): related for name, identifier, related, *_ in CASES}
    assert declared == expected


@pytest.mark.parametrize("feature,identifier,related,kwargs,guard,error,match", CASES)
def test_declared_refusal_reaches_actual_guard(
    feature, identifier, related, kwargs, guard, error, match
):
    declaration = next(
        item
        for item in capabilities()["features"][feature]["refuses"]
        if item["id"] == identifier
    )
    assert declaration["request_fields"] == related
    with pytest.raises(error, match=match):
        config = SolveConfig(**kwargs)
        guard(config)
