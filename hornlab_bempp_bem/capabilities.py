"""Versioned package capabilities, independent of this host's runtime readiness."""
from __future__ import annotations

from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from typing import Any, get_args

from .config import (
    GROUND_PLANES,
    AssemblyBackend,
    BIEFormulation,
    NativeSymmetryPlane,
    SolveConfig,
    SourceMotion,
    reject_unsupported_native_symmetry,
)

# Change this when the report's shape changes, not when existing support changes.
CAPABILITY_SCHEMA_VERSION = 1
# Version of the Python SolveConfig request contract; there is no wire protocol.
REQUEST_SCHEMA_VERSION = 1


def _supports_image_config(**kwargs: Any) -> bool:
    """Use the solver's config guards without loading a mesh or a backend."""
    try:
        reject_unsupported_native_symmetry(SolveConfig(**kwargs))
    except (ValueError, NotImplementedError):
        return False
    return True


def capabilities() -> dict[str, Any]:
    """Return a fresh, JSON-serializable description of the supported API.

    This reports package support, not hardware availability. It does not import
    bempp-cl, enumerate OpenCL devices, assemble operators, or run a solve.
    ``package_version`` is ``None`` in a source checkout without distribution
    metadata. Request fields are the constructor fields of ``SolveConfig``;
    feature entries additionally describe value and composition restrictions.
    """
    try:
        package_version = version("hornlab-bempp-bem")
    except PackageNotFoundError:
        package_version = None

    request_fields = [item.name for item in fields(SolveConfig) if item.init]
    symmetry_planes = [
        plane for plane in get_args(NativeSymmetryPlane)
        if _supports_image_config(native_symmetry_plane=plane)
    ]
    image_formulations = [
        formulation.value for formulation in BIEFormulation
        if _supports_image_config(
            ground_plane=GROUND_PLANES[0], formulation=formulation,
        )
    ]
    compositions = [
        {"native_symmetry_plane": symmetry, "ground_plane": ground}
        for symmetry in symmetry_planes
        for ground in GROUND_PLANES
        if _supports_image_config(native_symmetry_plane=symmetry, ground_plane=ground)
    ]
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "package": "hornlab-bempp-bem",
        "package_version": package_version,
        "request_fields": request_fields,
        "conventions": {"time_convention": "exp(-i*omega*t)", "outgoing_wave": "exp(+i*k*r)"},
        "features": {
            "source_motion": {
                "supported": "source_motion" in request_fields,
                "values": [SourceMotion.NORMAL, SourceMotion.AXIAL],
            },
            "source_axes": {
                "supported": "source_axes" in request_fields,
                "requires_source_motion": SourceMotion.AXIAL,
                "requires_all_source_tags": True,
                "must_lie_in_symmetry_subspace": True,
            },
            "formulation": {
                "supported": "formulation" in request_fields,
                "values": [item.value for item in BIEFormulation],
            },
            "complex_k_shift": {"supported": "complex_k_shift" in request_fields},
            "frame_override": {"supported": "frame_override" in request_fields},
            "infinite_baffle": {
                "supported": "aperture_tag" in request_fields,
                "request_field": "aperture_tag",
                "formulations": [
                    item.value for item in BIEFormulation
                    if item is not BIEFormulation.BURTON_MILLER
                ],
                "composes_with_symmetry": False,
                "supports_impedance_sources": False,
                "supports_surface_traces": False,
            },
            "native_symmetry": {
                "supported": "native_symmetry_plane" in request_fields,
                "planes": symmetry_planes,
                "formulations": image_formulations,
                "supports_impedance_sources": False,
            },
            "ground_plane": {
                "supported": "ground_plane" in request_fields,
                "planes": list(GROUND_PLANES),
                "formulations": list(image_formulations),
                "supports_impedance_sources": False,
                "composes_with_symmetry": bool(compositions),
                "symmetry_compositions": compositions,
            },
            "explicit_frequencies": {"supported": True, "entry_point": "solve_frequencies"},
            "on_frequency_result": {
                "supported": "on_frequency_result" in request_fields,
                "serial_only": True,
            },
            "return_surface_traces": {
                "supported": "return_surface_traces" in request_fields,
                "supports_infinite_baffle": False,
            },
            "require_closed_mesh": {"supported": "require_closed_mesh" in request_fields},
            "workers": {
                "supported": "workers" in request_fields,
                "parallel_supports_on_frequency_result": False,
            },
            "assembly_backend": {
                "supported": "assembly_backend" in request_fields,
                "values": list(get_args(AssemblyBackend)),
            },
        },
    }
