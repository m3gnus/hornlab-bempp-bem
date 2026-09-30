"""Shared BEM capability contract; package support is separate from readiness."""

from __future__ import annotations

from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from typing import get_args
from .config import (
    AssemblyBackend,
    BIEFormulation,
    GROUND_PLANES,
    NativeSymmetryPlane,
    SolveConfig,
    SourceMotion,
    reject_unsupported_native_symmetry,
)

# This named schema is independent of BEAT's provider schema.
CAPABILITY_SCHEMA_VERSION = 1
# Version of the Python SolveConfig contract, not a native/wire protocol.
REQUEST_SCHEMA_VERSION = 1


def _feature(
    request_fields,
    *names,
    values=None,
    requires=None,
    refuses=(),
    composes_with=(),
    supported=None,
):
    return {
        "supported": all(name in request_fields for name in names)
        if supported is None
        else supported,
        "request_fields": list(names),
        "values": values,
        "requires": {} if requires is None else requires,
        "refuses": list(refuses),
        "composes_with": list(composes_with),
    }


def _refusal(identifier, reason, *names):
    return {"id": identifier, "request_fields": list(names), "reason": reason}


def _supports(**kwargs):
    """Read support from real config/image guards without a mesh or runtime."""
    try:
        config = SolveConfig(**kwargs)
        reject_unsupported_native_symmetry(config)
    except (ValueError, NotImplementedError):
        return False
    return True


def capabilities() -> dict[str, Any]:
    """Return fresh package support metadata, without probing runtime readiness."""
    request_fields = [item.name for item in fields(SolveConfig) if item.init]
    try:
        package_version = version("hornlab-bempp-bem")
    except PackageNotFoundError:
        package_version = None
    symmetry = [
        plane
        for plane in get_args(NativeSymmetryPlane)
        if _supports(native_symmetry_plane=plane)
    ]
    ground = list(GROUND_PLANES)
    image_formulations = [
        value
        for value in [item.value for item in BIEFormulation]
        if _supports(ground_plane=ground[0], formulation=value)
    ]
    compositions = [
        {"native_symmetry_plane": plane, "ground_plane": gp}
        for plane in symmetry
        for gp in ground
        if _supports(native_symmetry_plane=plane, ground_plane=gp)
    ]

    def f(*names, **kwargs):
        return _feature(request_fields, *names, **kwargs)

    r = _refusal
    ib_refusals = [
        r(
            "ib_burton_miller",
            "Coupled infinite baffle refuses Burton-Miller.",
            "formulation",
        ),
        r(
            "ib_surface_traces",
            "Coupled infinite baffle cannot retain generic surface traces.",
            "return_surface_traces",
        ),
        r(
            "ib_symmetry",
            "Coupled infinite baffle requires the full domain; all native symmetry is refused.",
            "native_symmetry_plane",
        ),
        r(
            "ib_robin",
            "Coupled infinite baffle refuses Robin impedance data.",
            "impedance_sources",
        ),
    ]
    formulation_refusals = [
        r(
            "bm_robin",
            "Burton-Miller refuses Robin impedance data.",
            "impedance_sources",
        ),
        r(
            "bm_infinite_baffle",
            "Burton-Miller refuses coupled infinite baffle.",
            "aperture_tag",
        ),
        r(
            "bm_ground",
            "Burton-Miller refuses ground-plane image assembly.",
            "ground_plane",
        ),
        r(
            "bm_symmetry",
            "Burton-Miller refuses native image symmetry.",
            "native_symmetry_plane",
        ),
    ]
    ground_refusals = [
        r("ground_burton_miller", "Ground images refuse Burton-Miller.", "formulation"),
        r(
            "ground_robin",
            "Ground images refuse Robin impedance data.",
            "impedance_sources",
        ),
        r(
            "ground_same_plane",
            "A ground plane cannot also be a reduced mesh cut plane.",
            "native_symmetry_plane",
        ),
    ]
    return {
        "schema": "hornlab-bem-capabilities",
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "package": "hornlab-bempp-bem",
        "package_version": package_version,
        "request_fields": request_fields,
        "conventions": {
            "time_convention": "exp(-i*omega*t)",
            "outgoing_wave": "exp(+i*k*r)",
        },
        "features": {
            "source_motion": f(
                "source_motion", values=[SourceMotion.NORMAL, SourceMotion.AXIAL]
            ),
            "source_axes": f(
                "source_axes",
                requires={
                    "motion": "source_motion='axial'.",
                    "coverage": "Every axial velocity-source tag must have an axis.",
                    "geometry": "Axes must be finite, nonzero, and lie in the symmetry subspace.",
                },
            ),
            "formulation": f(
                "formulation",
                values=[item.value for item in BIEFormulation],
                refuses=formulation_refusals,
            ),
            "complex_k_shift": f("complex_k_shift"),
            "frame_override": f("frame_override"),
            "infinite_baffle": f(
                "aperture_tag",
                requires={
                    "formulation": list(image_formulations),
                    "native_symmetry_plane": [None],
                },
                refuses=ib_refusals,
                composes_with=[],
            ),
            "native_symmetry": f(
                "native_symmetry_plane",
                values=symmetry,
                requires={"formulation": list(image_formulations)},
                refuses=[
                    r(
                        "symmetry_xy",
                        "Legacy xy native symmetry is not implemented.",
                        "native_symmetry_plane",
                    ),
                    r(
                        "symmetry_burton_miller",
                        "Native image symmetry refuses Burton-Miller.",
                        "formulation",
                    ),
                    r(
                        "symmetry_robin",
                        "Native image symmetry refuses Robin impedance data.",
                        "impedance_sources",
                    ),
                ],
                composes_with=[{"feature": "ground_plane", "values": compositions}]
                if compositions
                else [],
            ),
            "ground_plane": f(
                "ground_plane",
                values=ground,
                requires={"formulation": image_formulations},
                refuses=ground_refusals,
                composes_with=[{"feature": "native_symmetry", "values": compositions}]
                if compositions
                else [],
            ),
            "explicit_frequencies": f(values=["solve_frequencies"], supported=True),
            "on_frequency_result": f(
                "on_frequency_result",
                requires={"workers": [1]},
                refuses=[
                    r(
                        "callback_parallel",
                        "Frequency-result callbacks refuse workers > 1.",
                        "workers",
                    )
                ],
            ),
            "return_surface_traces": f(
                "return_surface_traces",
                refuses=[
                    r(
                        "traces_infinite_baffle",
                        "Surface traces refuse coupled infinite baffle.",
                        "aperture_tag",
                    )
                ],
            ),
            "require_closed_mesh": f("require_closed_mesh"),
            "workers": f(
                "workers",
                refuses=[
                    r(
                        "workers_callback",
                        "Parallel workers refuse a frequency-result callback.",
                        "on_frequency_result",
                    )
                ],
            ),
            "assembly_backend": f(
                "assembly_backend", values=list(get_args(AssemblyBackend))
            ),
        },
    }
