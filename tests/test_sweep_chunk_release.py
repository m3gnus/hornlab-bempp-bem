"""A merged chunk must not stay resident once it is in the full-sweep arrays.

``run_sweep_parallel`` preallocates full-sweep arrays and copies each worker
chunk into them. It also kept every ``Future`` in a ``futures`` dict for the
whole merge, and a completed ``Future`` keeps its payload alive in ``_result``.
With ``return_surface_traces`` that payload is the chunk's complex128 pressure
and Neumann traces, so every chunk stayed resident *next to* the arrays it had
already been copied into.

Peak was therefore about twice ``estimate_field_trace_retention_bytes``, which
counts one copy -- so the caller's retention cap could admit a solve that then
needed double the budget it was checked against. That only became reachable when
WG enabled splitting.

Two things this test has to get right, both learned by getting them wrong:

* **Measure during the merge, not after.** Once ``run_sweep_parallel`` returns,
  its locals are gone and every chunk is collectable whether or not the loop
  released them as it went. A version that checked afterwards passed against the
  unfixed code and proved nothing.
* **Do not assume an order.** The loop drains a ``set``, so chunks are consumed
  in arbitrary order and a probe pinned to one particular future fires at an
  unpredictable point -- passing or failing by luck. The probe therefore runs on
  every chunk and only asserts on the last one *consumed*.
"""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np

from hornlab_bempp_bem.config import ObservationConfig, SolveConfig
from hornlab_bempp_bem.observation import ObservationFrame
from hornlab_bempp_bem.result import MeshInfo
from hornlab_bempp_bem import sweep as sweep_module


N_FREQ = 8
N_CHUNKS = 4
N_DOF = 64


def _frame() -> ObservationFrame:
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.array([0.0, 0.0, 1.0]),
        source_center=np.zeros(3),
    )


def _mesh():
    mesh = MagicMock()
    mesh.info = MeshInfo(
        n_vertices=100,
        n_triangles=200,
        physical_groups={1: "body", 2: "source"},
        bounding_box_m=(np.zeros(3), np.ones(3)),
    )
    mesh.grid = MagicMock()
    mesh.physical_tags = np.array([1] * 180 + [2] * 20, dtype=np.int32)
    return mesh


class _ProbingFuture(Future):
    """A Future that reports the moment its chunk is consumed."""

    probe = None

    def result(self, timeout=None):
        if self.probe is not None:
            self.probe()
        return super().result(timeout)


class _FakeExecutor:
    """Hands back completed Futures carrying real numpy payloads."""

    def __init__(self, *args, **kwargs) -> None:
        self.tracked: list[weakref.ref] = []
        self.consumed = 0
        self.live_at_final_merge: int | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _probe(self) -> None:
        self.consumed += 1
        if self.consumed < N_CHUNKS:
            return
        # The final chunk. Everything else has been merged into the full-sweep
        # arrays already; only this one legitimately remains, because its result
        # is being returned right now.
        gc.collect()
        self.live_at_final_merge = sum(
            1 for ref in self.tracked if ref() is not None
        )

    def submit(self, _fn, **kwargs):
        indices = np.asarray(kwargs["global_indices"])
        n = len(indices)
        n_planes, n_angles = 1, 5
        traces = np.ones((n, N_DOF), dtype=np.complex128)
        neumann = np.ones((n, N_DOF), dtype=np.complex128)
        # Tracked weakly, so the tracking itself never holds a chunk alive.
        self.tracked.append(weakref.ref(traces))
        future = _ProbingFuture()
        future.set_result(
            (
                np.ones((n, n_planes, n_angles), dtype=np.complex128),
                np.zeros((n, n_planes, n_angles), dtype=np.float64),
                np.ones(n, dtype=np.complex128),
                [{"fallback_used": False} for _ in range(n)],
                {2: np.ones(n, dtype=np.complex128)},
                None,
                traces,
                neumann,
            )
        )
        future.probe = self._probe
        return future


def test_merged_chunks_are_released_as_the_merge_proceeds():
    config = SolveConfig(
        freq_min_hz=100.0,
        freq_max_hz=800.0,
        freq_count=N_FREQ,
        observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        return_surface_traces=True,
    )
    frequencies = np.linspace(100.0, 800.0, N_FREQ)
    executors: list[_FakeExecutor] = []

    def _make_executor(*args, **kwargs):
        executor = _FakeExecutor()
        executors.append(executor)
        return executor

    with patch.object(sweep_module, "ProcessPoolExecutor", _make_executor):
        result = sweep_module.run_sweep_parallel(
            _mesh(),
            frequencies,
            _frame(),
            config,
            worker_count=N_CHUNKS,
            mesh_contracts_validated=True,
        )

    assert result.surface_pressure_complex is not None
    assert result.surface_pressure_complex.shape == (N_FREQ, N_DOF)

    executor = executors[0]
    assert executor.consumed == N_CHUNKS, "not every chunk was consumed"
    assert executor.live_at_final_merge is not None, "the probe never ran"

    # Exactly one chunk may be alive at that instant: the one being returned.
    # Every other survivor is a second copy of data the retention cap was only
    # charged for once -- the doubling this fix removes. Unfixed, all four are
    # resident.
    assert executor.live_at_final_merge == 1, (
        f"{executor.live_at_final_merge} chunks were resident when the final "
        f"chunk was merged; only the final one should be (of {N_CHUNKS})"
    )
