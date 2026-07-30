"""Auto worker selection must never make a sweep slower.

A spawned worker re-imports bempp-cl and re-JITs its numba kernels before it
can solve anything -- about 5 s against roughly 0.13 s per warm frequency -- so
splitting a short sweep across processes is far slower than running it in one
warm process. Measured 2026-07-30: 16 frequencies took 1.7 s serial and 16.8 s
across two workers.
"""
from __future__ import annotations

import logging

import pytest

from hornlab_bempp_bem import (
    _MIN_FREQUENCIES_PER_WORKER,
    _detect_worker_count,
    _resolve_worker_count,
)


@pytest.mark.parametrize("n_frequencies", [1, 5, 16, 40, 79])
def test_auto_stays_serial_until_a_worker_can_pay_for_itself(n_frequencies):
    assert _resolve_worker_count(0, n_frequencies) == 1


def test_auto_splits_once_the_sweep_is_long_enough():
    cores = _detect_worker_count()
    n_frequencies = 2 * _MIN_FREQUENCIES_PER_WORKER

    resolved = _resolve_worker_count(0, n_frequencies)

    assert resolved == min(2, cores)


def test_auto_never_exceeds_the_core_count():
    cores = _detect_worker_count()

    resolved = _resolve_worker_count(0, 10_000 * _MIN_FREQUENCIES_PER_WORKER)

    assert resolved == cores


def test_auto_always_gives_each_worker_enough_frequencies():
    for n_frequencies in range(1, 500, 7):
        resolved = _resolve_worker_count(0, n_frequencies)
        assert resolved >= 1
        if resolved > 1:
            assert n_frequencies // resolved >= _MIN_FREQUENCIES_PER_WORKER


def test_explicit_serial_is_respected():
    assert _resolve_worker_count(1, 10_000) == 1


def test_explicit_worker_count_is_honoured_even_when_it_will_lose():
    """The caller may know something the arithmetic does not."""
    assert _resolve_worker_count(4, 8) == 4


def test_explicit_worker_count_warns_when_the_sweep_is_too_short(caplog):
    with caplog.at_level(logging.WARNING, logger="hornlab_bempp_bem"):
        _resolve_worker_count(4, 8)

    assert "likely slower than workers=1" in caplog.text


def test_no_warning_when_an_explicit_count_is_justified(caplog):
    with caplog.at_level(logging.WARNING, logger="hornlab_bempp_bem"):
        _resolve_worker_count(2, 4 * _MIN_FREQUENCIES_PER_WORKER)

    assert "likely slower" not in caplog.text
