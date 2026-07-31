"""``gmres_max_iter`` must bound the iterations it names.

scipy counts GMRES ``maxiter`` in outer restart cycles. Passing this package's
inner-iteration budget straight through let a non-converging frequency run
``restart * gmres_max_iter`` iterations -- 500,000 at the defaults.
"""
from __future__ import annotations

import pytest

from hornlab_bempp_bem.symmetry import _gmres_outer_cycles


@pytest.mark.parametrize(
    ("max_iter", "restart", "expected_cycles"),
    [
        (5000, 100, 50),      # package defaults, exact
        (5000, 50, 100),
        (200, 100, 2),
        (100, 100, 1),
        (150, 100, 2),        # round up: 100 would cut off a solve needing 148
        (99, 100, 1),
        (1, 100, 1),
        (0, 100, 1),          # never ask scipy for zero cycles
        (20, 10, 2),
    ],
)
def test_inner_budget_becomes_outer_cycles(max_iter, restart, expected_cycles):
    assert _gmres_outer_cycles(max_iter, restart) == expected_cycles


@pytest.mark.parametrize(
    ("max_iter", "restart"),
    [(5000, 100), (5000, 50), (200, 100), (150, 100), (20, 10), (7919, 97)],
)
def test_the_budget_is_never_truncated(max_iter, restart):
    """A configured budget must always be reachable."""
    cap = _gmres_outer_cycles(max_iter, restart) * restart

    assert cap >= max_iter


@pytest.mark.parametrize(
    ("max_iter", "restart"),
    [(5000, 100), (5000, 50), (200, 100), (150, 100), (20, 10), (7919, 97)],
)
def test_the_overshoot_is_bounded_by_one_cycle(max_iter, restart):
    """Rounding up costs at most restart-1 extra iterations, never a multiple."""
    cap = _gmres_outer_cycles(max_iter, restart) * restart

    assert cap - max_iter < restart


def test_the_pre_fix_blowup_is_gone():
    """maxiter=gmres_max_iter allowed restart * max_iter = 500,000."""
    cap = _gmres_outer_cycles(5000, 100) * 100

    assert cap == 5000
    assert cap < 5000 * 100
