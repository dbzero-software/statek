"""Tests for statek_ctx_set and statek_ctx_get."""

import pytest

from statek.utils import statek_ctx_set, statek_ctx_get


def test_set_and_get():
    _STATEK_CTX = {}  # noqa: F841
    statek_ctx_set(foo="bar")
    assert statek_ctx_get("foo") == "bar"


def test_get_missing_key_raises():
    _STATEK_CTX = {}  # noqa: F841
    with pytest.raises(KeyError):
        statek_ctx_get("missing")


def test_get_missing_key_with_default():
    _STATEK_CTX = {}  # noqa: F841
    assert statek_ctx_get("missing", None) is None
    assert statek_ctx_get("missing", 42) == 42


def test_set_overwrites_existing():
    _STATEK_CTX = {"x": 1}  # noqa: F841
    statek_ctx_set(x=2)
    assert statek_ctx_get("x") == 2


def test_set_multiple_keys():
    _STATEK_CTX = {}  # noqa: F841
    statek_ctx_set(a=1, b=2)
    assert statek_ctx_get("a") == 1
    assert statek_ctx_get("b") == 2
