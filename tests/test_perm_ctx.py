"""Tests for perm_ctx_set and perm_ctx_get."""

import pytest

from statek.utils import perm_ctx_set, perm_ctx_get


def test_set_and_get():
    _PERM_CTX = {}  # noqa: F841
    perm_ctx_set(foo="bar")
    assert perm_ctx_get("foo") == "bar"


def test_get_missing_key_raises():
    _PERM_CTX = {}  # noqa: F841
    with pytest.raises(KeyError):
        perm_ctx_get("missing")


def test_get_missing_key_with_default():
    _PERM_CTX = {}  # noqa: F841
    assert perm_ctx_get("missing", None) is None
    assert perm_ctx_get("missing", 42) == 42


def test_set_overwrites_existing():
    _PERM_CTX = {"x": 1}  # noqa: F841
    perm_ctx_set(x=2)
    assert perm_ctx_get("x") == 2


def test_set_multiple_keys():
    _PERM_CTX = {}  # noqa: F841
    perm_ctx_set(a=1, b=2)
    assert perm_ctx_get("a") == 1
    assert perm_ctx_get("b") == 2


def test_get_no_context_raises():
    with pytest.raises(RuntimeError):
        perm_ctx_get("key")


def test_get_no_context_with_default():
    assert perm_ctx_get("key", "fallback") == "fallback"


def test_set_no_context_creates_on_demand():
    _STATEK_CTX = {"job": _FakeJob()}  # noqa: F841
    perm_ctx_set(x=10)
    assert perm_ctx_get("x") == 10


def test_set_creates_perm_ctx_on_job():
    job = _FakeJob()
    _STATEK_CTX = {"job": job}  # noqa: F841
    assert job.perm_ctx is None
    perm_ctx_set(key="value")
    assert job.perm_ctx == {"key": "value"}


def test_get_wrong_arg_count():
    _PERM_CTX = {}  # noqa: F841
    with pytest.raises(TypeError):
        perm_ctx_get()
    with pytest.raises(TypeError):
        perm_ctx_get("a", "b", "c")


class _FakeJob:
    """Minimal stub with _perm_ctx attribute for on-demand creation tests."""
    perm_ctx = None
