"""Tests for perm_ctx_set and perm_ctx_get."""

import pytest

from statek.utils import perm_ctx_set, perm_ctx_get
from statek.utils import register_local_context, unregister_local_context


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


def test_set_creates_perm_ctx_in_pyenv_local_state():
    job = _FakeJob()
    _STATEK_CTX = {"job": job}  # noqa: F841
    assert job.py_env.local_state is None
    perm_ctx_set(key="value")
    assert job.py_env.local_state["_PERM_CTX"] == {"key": "value"}


def test_set_creates_perm_ctx_in_active_local_context():
    job = _FakeJob()
    local_context = {}
    context_id = register_local_context(local_context)
    try:
        _STATEK_CTX = {"job": job, "_local_context_id": context_id}  # noqa: F841
        perm_ctx_set(key="value")
        assert local_context["_PERM_CTX"] == {"key": "value"}
        assert job.py_env.local_state is None
    finally:
        unregister_local_context(context_id)


def test_set_sync_mirrors_active_local_context_to_pyenv_local_state():
    job = _FakeJob()
    local_context = {}
    context_id = register_local_context(local_context)
    try:
        _STATEK_CTX = {"job": job, "_local_context_id": context_id}  # noqa: F841
        perm_ctx_set(sync=True, key="value")
        assert local_context["_PERM_CTX"] == {"key": "value"}
        assert job.py_env.local_state["_PERM_CTX"] == {"key": "value"}
    finally:
        unregister_local_context(context_id)


def test_set_sync_updates_local_context_without_current_job():
    _PERM_CTX = {}  # noqa: F841
    perm_ctx_set(sync=True, key="value")
    assert perm_ctx_get("key") == "value"


def test_get_wrong_arg_count():
    _PERM_CTX = {}  # noqa: F841
    with pytest.raises(TypeError):
        perm_ctx_get()
    with pytest.raises(TypeError):
        perm_ctx_get("a", "b", "c")


class _FakeJob:
    """Minimal stub with PyEnv local state for on-demand creation tests."""

    class _FakePyEnv:
        local_state = None

        @property
        def perm_ctx(self):
            return None if self.local_state is None else self.local_state.get("_PERM_CTX")

        def update_locals(self, **kwargs):
            if self.local_state is None:
                self.local_state = {}
            self.local_state.update(kwargs)

    def __init__(self):
        self.py_env = self._FakePyEnv()
