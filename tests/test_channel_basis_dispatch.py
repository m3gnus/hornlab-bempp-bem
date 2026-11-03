"""Channel bases must never substitute ordinary physics for coupled IB."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import hornlab_bempp_bem as package
from hornlab_bempp_bem import sweep


@pytest.mark.parametrize("mesh_tag,config_tag", [(12, None), (None, 12), (12, 12)])
def test_channel_basis_refuses_coupled_ib_before_standard_dispatch(
    monkeypatch,
    mesh_tag,
    config_tag,
):
    mesh = SimpleNamespace(
        physical_tags=np.array([2, 12]),
        coupled_ib_aperture_tag=mesh_tag,
    )
    resolve_mesh = Mock(return_value=mesh)
    monkeypatch.setattr(package, "_resolve_mesh", resolve_mesh)
    monkeypatch.setattr(package, "_resolve_frame", Mock(return_value=None))
    standard = Mock(side_effect=AssertionError("ordinary sweep was reached"))
    monkeypatch.setattr(sweep, "run_sweep_serial", standard)
    config = package.SolveConfig(
        channels=[package.Channel("driver", [2])],
        aperture_tag=config_tag,
    )

    with pytest.raises(
        NotImplementedError, match="coupled infinite-baffle.*solve_frequencies"
    ):
        package.solve_channel_basis(mesh, config, [1000.0])

    standard.assert_not_called()
    assert resolve_mesh.call_args.kwargs["aperture_tag"] == config_tag


def test_channel_basis_uses_common_aperture_conflict_guard(monkeypatch):
    mesh = SimpleNamespace(coupled_ib_aperture_tag=12)
    monkeypatch.setattr(package, "_resolve_mesh", lambda *a, **kw: mesh)
    config = package.SolveConfig(
        channels=[package.Channel("driver", [2])],
        aperture_tag=13,
    )
    with pytest.raises(ValueError, match="conflicts with loaded mesh"):
        package.solve_channel_basis(mesh, config, [1000.0])


def test_channel_basis_without_aperture_metadata_keeps_standard_dispatch(monkeypatch):
    mesh = SimpleNamespace(
        physical_tags=np.array([2, 12]),
        coupled_ib_aperture_tag=None,
    )
    frame = object()
    monkeypatch.setattr(package, "_resolve_mesh", lambda *a, **kw: mesh)
    monkeypatch.setattr(package, "_resolve_frame", lambda *a: frame)

    class ReachedStandard(Exception):
        pass

    standard = Mock(side_effect=ReachedStandard)
    monkeypatch.setattr(sweep, "run_sweep_serial", standard)
    config = package.SolveConfig(channels=[package.Channel("driver", [2])])
    with pytest.raises(ReachedStandard):
        package.solve_channel_basis(mesh, config, [1000.0])
    dispatched = standard.call_args.args[3]
    assert dispatched.aperture_tag is None
    assert dispatched.velocity_sources == config.velocity_sources
    assert dispatched.frame_override is frame
