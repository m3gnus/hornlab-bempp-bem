"""Scoped BLAS thread limiting must restore each library's runtime budget."""

from __future__ import annotations

import ctypes
import threading
from types import SimpleNamespace

import pytest

from hornlab_bempp_bem import _blas_threads as blas


class _Function:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.function(*args)


class _Library:
    def __init__(self, threads, symbol):
        self.threads = threads
        self.calls = []
        self.setter = _Function(self.set_threads)
        self.getter = _Function(lambda: self.threads)
        self.symbol = symbol
        self.get_symbol = symbol.replace("_set_num_threads", "_get_num_threads")
        setattr(self, symbol, self.setter)
        setattr(self, self.get_symbol, self.getter)

    def set_threads(self, value):
        self.calls.append(value)
        self.threads = value


@pytest.fixture
def libraries(monkeypatch):
    # Exercise the real discovery path, with separate NumPy and SciPy BLASes.
    candidates = blas._CANDIDATES[:2]
    libs = [_Library(n, row[2]) for n, row in zip((3, 6), candidates)]
    paths = {f"{row[1]}/openblas.so": lib for row, lib in zip(candidates, libs)}
    monkeypatch.setattr(blas, "_CANDIDATES", candidates)
    monkeypatch.setattr(blas, "_setters", None)
    monkeypatch.setattr(blas.ctypes, "CDLL", paths.__getitem__)

    def discover(pattern):
        if not pattern.endswith(".so*"):
            return []
        return [path for path in paths if path.split("/")[0] in pattern]

    monkeypatch.setattr(blas.glob, "glob", discover)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    return libs


def test_each_library_restores_its_runtime_count(libraries):
    with blas.limited_blas_threads(1):
        assert [lib.threads for lib in libraries] == [1, 1]
    assert [lib.threads for lib in libraries] == [3, 6]


def test_nested_limits_restore_outer_then_original_counts(libraries):
    with blas.limited_blas_threads(2):
        with blas.limited_blas_threads(1):
            assert [lib.threads for lib in libraries] == [1, 1]
        assert [lib.threads for lib in libraries] == [2, 2]
    assert [lib.threads for lib in libraries] == [3, 6]


def test_overlapping_threads_restore_original_counts(libraries):
    started = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    errors = []
    observed = []

    def second_scope():
        try:
            started.set()
            with blas.limited_blas_threads(2):
                observed.append([lib.threads for lib in libraries])
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("second BLAS scope was not released")
        except Exception as error:  # noqa: BLE001 - report failures in the test thread
            errors.append(error)

    worker = threading.Thread(target=second_scope, daemon=True)
    try:
        with blas.limited_blas_threads(1):
            worker.start()
            assert started.wait(5)
            # Keep the second scope open until the first has restored its
            # counts. It must wait outside the scope while the first owns it.
            overlapped = entered.wait(0.2)
        assert entered.wait(5)
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(5)

    assert not worker.is_alive()
    assert errors == []
    assert observed == [[2, 2]]
    assert [lib.threads for lib in libraries] == [3, 6]
    assert not overlapped


def test_exception_restores_each_runtime_count(libraries):
    with (
        pytest.raises(RuntimeError, match="solve blew up"),
        blas.limited_blas_threads(1),
    ):
        raise RuntimeError("solve blew up")
    assert [lib.threads for lib in libraries] == [3, 6]


def test_non_positive_thread_counts_are_a_no_op(libraries):
    for count in (0, -1):
        with blas.limited_blas_threads(count):
            pass
    assert all(lib.calls == [] for lib in libraries)


def test_missing_getter_leaves_library_untouched(libraries):
    lib = libraries[0]
    delattr(lib, lib.get_symbol)
    with blas.limited_blas_threads(1):
        assert lib.threads == 3
        assert libraries[1].threads == 1
    assert lib.calls == []


@pytest.mark.parametrize("count", [0, -1, None])
def test_unusable_runtime_count_leaves_library_untouched(libraries, count):
    libraries[0].getter.function = lambda: count
    with blas.limited_blas_threads(1):
        assert libraries[0].threads == 3
    assert libraries[0].calls == []


def test_failing_getter_leaves_library_untouched(libraries):
    def fail():
        raise OSError("getter failed")

    libraries[0].getter.function = fail
    with blas.limited_blas_threads(1):
        assert libraries[0].threads == 3
    assert libraries[0].calls == []


def test_failing_setter_still_attempts_restoration(libraries):
    lib = libraries[0]

    def fail_after_setting(value):
        lib.set_threads(value)
        raise OSError("setter failed")

    lib.setter.function = fail_after_setting
    with blas.limited_blas_threads(1):
        assert libraries[1].threads == 1
    assert [lib.threads for lib in libraries] == [3, 6]


@pytest.mark.parametrize("candidate", blas._CANDIDATES)
def test_discovery_pairs_matching_getter_with_setter(monkeypatch, candidate):
    lib = _Library(3, candidate[2])
    monkeypatch.setattr(blas, "_CANDIDATES", [candidate])
    monkeypatch.setattr(blas, "_setters", None)
    monkeypatch.setattr(blas.ctypes, "CDLL", lambda path: lib)
    monkeypatch.setattr(blas.glob, "glob", lambda pattern: ["openblas.so"])
    with blas.limited_blas_threads(1):
        assert lib.threads == 1
    assert lib.threads == 3
    assert lib.setter.argtypes == [ctypes.c_int]
    assert lib.setter.restype is None
    assert lib.getter.argtypes == []
    assert lib.getter.restype is ctypes.c_int


def test_missing_symbols_are_a_no_op(monkeypatch):
    monkeypatch.setattr(blas, "_setters", None)
    monkeypatch.setattr(blas.ctypes, "CDLL", lambda path: SimpleNamespace())
    monkeypatch.setattr(blas.glob, "glob", lambda pattern: ["openblas.so"])
    assert not blas.blas_thread_control_available()
    with blas.limited_blas_threads(1):
        pass


def test_discovery_reports_what_this_install_can_do():
    assert isinstance(blas.blas_thread_control_available(), bool)
