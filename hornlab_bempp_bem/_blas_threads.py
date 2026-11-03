"""Scoped BLAS thread control for the dense linear solve.

LAPACK's LU scales *negatively* at the matrix sizes a waveguide solve produces.
Measured on the 12-vCPU reference host, 2026-07-30, complex64:

    N=260   1.73 ms on one thread   28.86 ms on twelve
    N=479   7.02 ms on one thread   49.07 ms on twelve
    N=898  34.10 ms on one thread   80.44 ms on twelve

The panel factorization synchronises far more than the work justifies at this
size. GEMM does *not* behave that way -- an 898x898 product goes the other way,
7.7 ms on twelve threads against 42 ms on one -- so pinning the whole process
would pay for the solve by taxing every other array operation in it, including
the symmetry reduction's own dense product and anything else sharing the
interpreter.

NumPy and SciPy each bundle their own OpenBLAS, and both export runtime
thread setters and getters, so the limit can be scoped to the factorization
instead of being fixed for the process at import time.

The limiter is best-effort by design: an unrecognised BLAS build, a missing
symbol, or a load failure leaves the thread count alone rather than failing a
solve over a performance hint.

The OpenBLAS thread count is process-global, so two threads entering the block
concurrently would fight over it and the second to leave would win. That is
acceptable here: bempp-cl's assemblers are serial within a process and the
parallel sweep uses processes, not threads. Nothing is corrupted either way --
the worst case is a solve running with the wrong number of threads.
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# (module name, library subdirectory, symbol suffix). NumPy's bundled build
# uses the ILP64 symbol, SciPy's the LP64 one.
_CANDIDATES = (
    ("numpy", "numpy.libs", "scipy_openblas_set_num_threads64_"),
    ("scipy", "scipy.libs", "scipy_openblas_set_num_threads"),
    ("numpy", "numpy.libs", "openblas_set_num_threads64_"),
    ("scipy", "scipy.libs", "openblas_set_num_threads"),
    ("numpy", ".libs", "openblas_set_num_threads"),
)

_lock = threading.Lock()
# Each entry pairs one library's setter with its matching runtime getter.
_setters: list[tuple] | None = None


def _discover_setters() -> list:
    """Find paired OpenBLAS thread setters and getters. Never raises."""
    setters = []
    seen: set[str] = set()
    for module_name, libdir, symbol in _CANDIDATES:
        try:
            module = __import__(module_name)
            root = os.path.join(os.path.dirname(module.__file__), "..", libdir)
            pattern = os.path.join(root, "*openblas*")
            for path in glob.glob(pattern + ".dll") + glob.glob(pattern + ".so*"):
                if path in seen:
                    continue
                try:
                    library = ctypes.CDLL(path)
                    setter = getattr(library, symbol)
                    getter = getattr(
                        library, symbol.replace("_set_num_threads", "_get_num_threads"),
                    )
                except (OSError, AttributeError):
                    continue
                setter.argtypes = [ctypes.c_int]
                setter.restype = None
                getter.argtypes = []
                getter.restype = ctypes.c_int
                setters.append((setter, getter))
                seen.add(path)
        except Exception:  # pragma: no cover - defensive
            continue
    if not setters:
        logger.debug("no paired OpenBLAS thread setter/getter found; not limiting")
    return setters


def _get_setters() -> list:
    global _setters
    with _lock:
        if _setters is None:
            _setters = _discover_setters()
        return _setters


def blas_thread_control_available() -> bool:
    """Whether the scoped limiter can actually do anything on this install."""
    return bool(_get_setters())


@contextmanager
def limited_blas_threads(threads: int = 1):
    """Restrict BLAS to ``threads`` for the duration of the block.

    Restores the previous count on exit, including on exception. A no-op when
    no paired setter/getter is available or ``threads`` is not positive.
    Each library's current runtime count is saved separately, so nested scopes
    and budgets set by an embedding application are preserved.
    """
    setters = _get_setters()
    if not setters or threads <= 0:
        yield
        return

    previous = []
    for setter, getter in setters:
        try:
            count = int(getter())
            if count <= 0:
                continue
            # Even a setter that raises after changing the count needs a
            # restoration attempt. A failed getter leaves the library alone.
            previous.append((setter, count))
            setter(int(threads))
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        yield
    finally:
        for setter, count in reversed(previous):
            try:
                setter(count)
            except Exception:  # pragma: no cover - defensive
                pass
