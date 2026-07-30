"""Scoped BLAS thread limiting must be safe, reversible, and never fatal."""
from __future__ import annotations

import os

import pytest

from hornlab_bempp_bem import _blas_threads
from hornlab_bempp_bem._blas_threads import (
    _ambient_thread_count,
    blas_thread_control_available,
    limited_blas_threads,
)


class _RecordingSetter:
    def __init__(self):
        self.calls = []
        self.argtypes = None

    def __call__(self, value):
        self.calls.append(int(value))


@pytest.fixture
def recording(monkeypatch):
    setter = _RecordingSetter()
    monkeypatch.setattr(_blas_threads, "_setters", [setter])
    return setter


def test_limit_is_applied_and_restored(recording, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")

    with limited_blas_threads(1):
        assert recording.calls == [1]

    assert recording.calls == [1, 8]


def test_limit_is_restored_after_an_exception(recording, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")

    with pytest.raises(RuntimeError):
        with limited_blas_threads(1):
            raise RuntimeError("solve blew up")

    assert recording.calls == [1, 6], "the limit must not leak out of the block"


def test_non_positive_thread_counts_are_a_no_op(recording):
    with limited_blas_threads(0):
        pass
    with limited_blas_threads(-1):
        pass

    assert recording.calls == []


def test_a_failing_setter_does_not_break_the_solve(monkeypatch):
    def explode(_value):
        raise OSError("vendor library went away")

    monkeypatch.setattr(_blas_threads, "_setters", [explode])

    with limited_blas_threads(1):
        pass  # must simply not raise


def test_missing_setters_are_a_no_op(monkeypatch):
    monkeypatch.setattr(_blas_threads, "_setters", [])

    with limited_blas_threads(1):
        pass


def test_ambient_count_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    assert _ambient_thread_count() == 3

    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    assert _ambient_thread_count() == 5


def test_ambient_count_falls_back_to_the_cpu_count(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    assert _ambient_thread_count() == (os.cpu_count() or 1)


@pytest.mark.parametrize("value", ["", "0", "-2", "not-a-number"])
def test_unusable_environment_values_fall_back(monkeypatch, value):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", value)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    assert _ambient_thread_count() == (os.cpu_count() or 1)


def test_discovery_reports_what_this_install_can_do():
    """Not an assertion about the host; it must simply answer without raising."""
    assert isinstance(blas_thread_control_available(), bool)
