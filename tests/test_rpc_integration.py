"""Tests for optional db0-rpc integration helpers."""

from statek import rpc_integration


def test_no_rpc_adapter_init_is_noop():
    rpc_integration._NO_RPC_ADAPTER.init(dangerously_skip_auth=True)  # pylint: disable=protected-access


def test_has_rpc_checks_loaded_rpc_identity(monkeypatch):
    no_rpc_adapter = rpc_integration._NO_RPC_ADAPTER  # pylint: disable=protected-access
    monkeypatch.setattr(rpc_integration, "rpc", no_rpc_adapter)
    assert not rpc_integration.has_rpc()

    monkeypatch.setattr(rpc_integration, "rpc", object())
    assert rpc_integration.has_rpc()
