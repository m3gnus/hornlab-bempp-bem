"""Assembly backend discovery and production-safe resolution."""
from __future__ import annotations

from dataclasses import dataclass

from .config import SolveConfig

BEMPP_BACKENDS = frozenset({"opencl", "numba"})


class AssemblyBackendUnavailable(RuntimeError):
    """Raised when a requested experimental backend cannot be used."""


@dataclass(frozen=True)
class AssemblyBackendResolution:
    """Effective backend used by the current production solver path."""

    requested_backend: str
    effective_backend: str
    fallback_used: bool
    reason: str | None = None


def resolve_assembly_backend(
    config: SolveConfig,
    *,
    required_precision: str | None = None,
) -> AssemblyBackendResolution:
    """Resolve a solve backend, probing before selecting OpenCL for ``auto``."""

    resolution = resolve_fastest_backend(
        config.assembly_backend,
        opencl_device=config.opencl_device,
        required_precision=(
            config.precision
            if required_precision is None
            else required_precision
        ),
    )
    if (
        resolution.effective_backend == "opencl"
        and config.slp_dlp_singular_quadrature in {3, 5}
    ):
        raise ValueError(
            "slp_dlp_singular_quadrature orders 3 and 5 are unsupported "
            "on the OpenCL backend: bempp-cl's singular assembler silently "
            "discards quadrature points at those orders; use an even order "
            "such as 4 or select the Numba backend"
        )
    return resolution


def resolve_fastest_backend(
    requested: str = "auto",
    *,
    opencl_device: str = "cpu",
    required_precision: str | None = None,
) -> AssemblyBackendResolution:
    """Resolve ``requested`` to the fastest backend this machine can run.

    Solve-time and post-solve field evaluation share this policy. On one CPU
    OpenCL device, Numba cost roughly three times the one-off per-process kernel
    compile and twice the steady-state evaluation, for results that agreed to
    machine precision; the ratios are machine-specific but the ordering has not
    been observed to invert. So ``auto`` probes, prefers OpenCL, and falls back
    when there is no usable device or the device lacks a required capability.
    """

    if requested == "auto":
        from .device import OpenCLError, configure_opencl

        try:
            configure_opencl(opencl_device)
        except OpenCLError as exc:
            return AssemblyBackendResolution(
                requested_backend=requested,
                effective_backend="numba",
                fallback_used=True,
                reason=str(exc),
            )
        resolution = AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend="opencl",
            fallback_used=False,
            reason="auto selected an available OpenCL Bempp backend",
        )
    else:
        resolution = _resolve_explicit(requested)

    return _enforce_required_precision(
        resolution,
        required_precision=required_precision,
        opencl_device=opencl_device,
    )


def _enforce_required_precision(
    resolution: AssemblyBackendResolution,
    *,
    required_precision: str | None,
    opencl_device: str,
) -> AssemblyBackendResolution:
    """Keep double-precision work away from OpenCL devices without fp64."""
    if (
        required_precision != "double"
        or resolution.effective_backend != "opencl"
    ):
        return resolution

    from .device import OpenCLError, opencl_devices_without_fp64

    try:
        unsupported_devices = opencl_devices_without_fp64(opencl_device)
    except OpenCLError as exc:
        reason = str(exc)
        if resolution.requested_backend == "auto":
            return AssemblyBackendResolution(
                requested_backend="auto",
                effective_backend="numba",
                fallback_used=True,
                reason=reason,
            )
        raise AssemblyBackendUnavailable(reason) from exc

    if not unsupported_devices:
        return resolution

    device_names = ", ".join(unsupported_devices)
    reason = (
        f"OpenCL device(s) {device_names} lack fp64 support required for "
        "double-precision assembly"
    )
    if resolution.requested_backend == "auto":
        return AssemblyBackendResolution(
            requested_backend="auto",
            effective_backend="numba",
            fallback_used=True,
            reason=reason,
        )
    raise AssemblyBackendUnavailable(
        f"{reason}; select the Numba backend or an fp64-capable OpenCL device"
    )


def _resolve_explicit(requested: str) -> AssemblyBackendResolution:
    """Resolve a backend named outright, which both callers take as given."""

    if requested in BEMPP_BACKENDS:
        return AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend=requested,
            fallback_used=False,
        )

    raise ValueError(
        "assembly_backend must be one of: auto, opencl, numba"
    )
