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

    if requested in BEMPP_BACKENDS:
        return AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend=requested,
            fallback_used=False,
        )

    raise ValueError(
        "assembly_backend must be one of: auto, opencl, numba"
    )
