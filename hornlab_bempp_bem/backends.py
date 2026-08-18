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


def resolve_assembly_backend(config: SolveConfig) -> AssemblyBackendResolution:
    """Resolve ``SolveConfig.assembly_backend`` to a current Bempp backend."""

    requested = config.assembly_backend
    if requested == "auto":
        return AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend="opencl",
            fallback_used=False,
            reason="auto selects the production OpenCL Bempp backend",
        )

    return _resolve_explicit(requested)


def resolve_fastest_backend(
    requested: str = "auto",
    *,
    opencl_device: str = "cpu",
) -> AssemblyBackendResolution:
    """Resolve ``requested`` to the fastest backend this machine can run.

    ``resolve_assembly_backend`` maps ``auto`` to OpenCL without probing, which
    is what a solve wants: a machine that cannot assemble on OpenCL should say
    so rather than quietly take several times as long. Post-solve field
    evaluation is the other case. It re-evaluates interactively from traces
    already on disk, so it has to keep working wherever the solve ran, and
    Numba is the slower option on both axes that matter. On one CPU OpenCL
    device it cost roughly three times the one-off per-process kernel compile
    and twice the steady-state evaluation, for results that agreed to machine
    precision; the ratios are machine-specific but the ordering has not been
    observed to invert. So ``auto`` probes, prefers OpenCL, and falls back
    only when there is no device to prefer.
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
        return AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend="opencl",
            fallback_used=False,
            reason="auto selects the production OpenCL Bempp backend",
        )

    return _resolve_explicit(requested)


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
