"""Capability contract checks; no numerical solve or runtime probe required."""
from __future__ import annotations

from dataclasses import fields
import importlib
import json
import subprocess
import sys
from typing import get_args

import pytest

import hornlab_bempp_bem as package
from hornlab_bempp_bem import (
    CAPABILITY_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    BIEFormulation,
    SolveConfig,
    SourceMotion,
    capabilities,
)
from hornlab_bempp_bem.config import (
    GROUND_PLANES,
    AssemblyBackend,
    NativeSymmetryPlane,
    reject_unsupported_native_symmetry,
)


def test_capabilities_schema_and_json_contract():
    report = capabilities()
    assert report["schema_version"] == CAPABILITY_SCHEMA_VERSION == 1
    assert report["request_schema_version"] == REQUEST_SCHEMA_VERSION == 1
    assert report["package"] == "hornlab-bempp-bem"
    assert json.loads(json.dumps(report)) == report
    assert report["conventions"]["time_convention"] == "exp(-i*omega*t)"
    assert {"capabilities", "CAPABILITY_SCHEMA_VERSION", "REQUEST_SCHEMA_VERSION"} <= set(package.__all__)


def test_capabilities_declare_exactly_the_solve_config_fields():
    declared = capabilities()["request_fields"]
    actual = {item.name for item in fields(SolveConfig) if item.init}
    assert len(declared) == len(set(declared))
    assert set(declared) == actual


def test_capabilities_package_version_uses_distribution_metadata(monkeypatch):
    module = importlib.import_module("hornlab_bempp_bem.capabilities")
    calls = []

    def version(name):
        calls.append(name)
        return "1.2.3"

    monkeypatch.setattr(module, "version", version)
    assert capabilities()["package_version"] == "1.2.3"
    assert calls == ["hornlab-bempp-bem"]


def test_capabilities_without_distribution_metadata(monkeypatch):
    module = importlib.import_module("hornlab_bempp_bem.capabilities")

    def version(name):
        raise module.PackageNotFoundError(name)

    monkeypatch.setattr(module, "version", version)
    assert capabilities()["package_version"] is None


def test_capabilities_is_a_fresh_snapshot():
    expected = capabilities()
    modified = capabilities()
    modified["request_fields"].clear()
    modified["features"]["ground_plane"]["planes"].clear()
    modified["features"]["ground_plane"]["symmetry_compositions"][0].clear()
    modified["features"]["native_symmetry"]["formulations"].clear()
    assert capabilities() == expected
    assert modified["features"]["ground_plane"]["formulations"] == expected["features"]["ground_plane"]["formulations"]


def test_import_and_handshake_do_not_load_numerical_backends():
    script = '''
import sys
class ForbidBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"bempp_cl", "pyopencl"}:
            raise AssertionError("capabilities loaded a numerical backend")
sys.meta_path.insert(0, ForbidBackends())
from hornlab_bempp_bem import capabilities
assert capabilities()["features"]["ground_plane"]["supported"]
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_capabilities_cover_wg_keyword_and_attribute_probes():
    report = capabilities()
    probed = {
        "source_motion", "aperture_tag", "formulation", "complex_k_shift",
        "frame_override", "ground_plane", "native_symmetry_plane",
        "source_axes", "on_frequency_result", "return_surface_traces",
        "require_closed_mesh", "workers",
    }
    assert probed <= set(report["request_fields"])
    feature_fields = {"infinite_baffle": "aperture_tag", "native_symmetry": "native_symmetry_plane"}
    for feature, detail in report["features"].items():
        if feature == "explicit_frequencies":
            assert callable(getattr(package, detail["entry_point"]))
        else:
            assert detail["supported"] == (feature_fields.get(feature, feature) in report["request_fields"])
    assert report["features"]["source_motion"]["values"] == [SourceMotion.NORMAL, SourceMotion.AXIAL]
    assert report["features"]["source_axes"]["requires_source_motion"] == SourceMotion.AXIAL
    assert report["features"]["formulation"]["values"] == [item.value for item in BIEFormulation]
    assert report["features"]["assembly_backend"]["values"] == list(get_args(AssemblyBackend))


@pytest.mark.parametrize("plane", get_args(NativeSymmetryPlane))
def test_declared_symmetry_modes_match_solver_guard(plane):
    try:
        reject_unsupported_native_symmetry(SolveConfig(native_symmetry_plane=plane))
    except NotImplementedError:
        supported = False
    else:
        supported = True
    assert (plane in capabilities()["features"]["native_symmetry"]["planes"]) == supported


@pytest.mark.parametrize("symmetry", [None, *get_args(NativeSymmetryPlane)])
@pytest.mark.parametrize("ground", GROUND_PLANES)
def test_declared_ground_compositions_match_solver_guards(symmetry, ground):
    try:
        reject_unsupported_native_symmetry(SolveConfig(native_symmetry_plane=symmetry, ground_plane=ground))
    except (ValueError, NotImplementedError):
        supported = False
    else:
        supported = True
    ground_feature = capabilities()["features"]["ground_plane"]
    if symmetry is None:
        assert (ground in ground_feature["planes"]) == supported
    else:
        assert ({"native_symmetry_plane": symmetry, "ground_plane": ground} in ground_feature["symmetry_compositions"]) == supported
    assert ground_feature["composes_with_symmetry"] == bool(ground_feature["symmetry_compositions"])


@pytest.mark.parametrize("formulation", list(BIEFormulation))
def test_image_formulations_and_robin_restrictions_match_guards(formulation):
    config = SolveConfig(ground_plane="xy", formulation=formulation)
    try:
        reject_unsupported_native_symmetry(config)
    except NotImplementedError:
        supported = False
    else:
        supported = True
    features = capabilities()["features"]
    for name in ("ground_plane", "native_symmetry"):
        assert (formulation.value in features[name]["formulations"]) == supported
        assert features[name]["supports_impedance_sources"] is False
    config.impedance_sources = {1: 0.1}
    with pytest.raises(NotImplementedError):
        reject_unsupported_native_symmetry(config)


def test_infinite_baffle_restrictions_match_its_validation():
    from hornlab_bempp_bem.infinite_baffle import _validate_coupled_infinite_baffle

    detail = capabilities()["features"]["infinite_baffle"]
    assert detail["request_field"] == "aperture_tag"
    assert detail["formulations"] == [item.value for item in BIEFormulation if item is not BIEFormulation.BURTON_MILLER]
    for field, value, declared in (
        ("native_symmetry_plane", "yz", "composes_with_symmetry"),
        ("impedance_sources", {1: 0.1}, "supports_impedance_sources"),
        ("return_surface_traces", True, "supports_surface_traces"),
        ("formulation", BIEFormulation.BURTON_MILLER, None),
    ):
        if declared is not None:
            assert detail[declared] is False
        # These guards precede geometry access, so no mesh/assembly is needed.
        with pytest.raises(NotImplementedError):
            _validate_coupled_infinite_baffle(None, SolveConfig(aperture_tag=3, **{field: value}), None)
    assert capabilities()["features"]["return_surface_traces"]["supports_infinite_baffle"] is False
    assert capabilities()["features"]["on_frequency_result"]["serial_only"] is True
    assert capabilities()["features"]["workers"]["parallel_supports_on_frequency_result"] is False
