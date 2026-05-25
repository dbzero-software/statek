"""Tests for optional db0-rpc integration helpers."""

from statek import rpc_integration


def test_rpc_mock_init_is_noop():
    rpc_integration._RPC_MOCK.init(dangerously_skip_auth=True)  # pylint: disable=protected-access


def test_has_rpc_checks_loaded_rpc_identity(monkeypatch):
    rpc_mock = rpc_integration._RPC_MOCK  # pylint: disable=protected-access
    monkeypatch.setattr(rpc_integration, "rpc", rpc_mock)
    assert not rpc_integration.has_rpc()

    monkeypatch.setattr(rpc_integration, "rpc", object())
    assert rpc_integration.has_rpc()
