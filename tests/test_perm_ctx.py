"""Tests for perm_ctx_set and perm_ctx_get."""

import pytest

from statek.utils import perm_ctx_set, perm_ctx_get


def _run_with_registered_job(job, func):
    """Run func while a fake current job is visible through _STATEK_CTX."""
    _STATEK_CTX = {"job": job}  # noqa: F841
    return func()


def test_set_and_get():
    job = _FakeJob()

    def exercise():
        perm_ctx_set(foo="bar")
        return perm_ctx_get("foo")

    assert _run_with_registered_job(job, exercise) == "bar"
    assert job.py_env.local_state["_PERM_CTX"] == {"foo": "bar"}


def test_get_missing_key_raises():
    job = _FakeJob()
    job.py_env.local_state = {"_PERM_CTX": {}}

    def exercise():
        perm_ctx_get("missing")

    with pytest.raises(KeyError):
        _run_with_registered_job(job, exercise)


def test_get_missing_key_with_default():
    job = _FakeJob()
    job.py_env.local_state = {"_PERM_CTX": {}}

    def exercise():
        return perm_ctx_get("missing", None), perm_ctx_get("missing", 42)

    assert _run_with_registered_job(job, exercise) == (None, 42)


def test_set_overwrites_existing():
    job = _FakeJob()
    job.py_env.local_state = {"_PERM_CTX": {"x": 1}}

    def exercise():
        perm_ctx_set(x=2)
        return perm_ctx_get("x")

    assert _run_with_registered_job(job, exercise) == 2
    assert job.py_env.local_state["_PERM_CTX"] == {"x": 2}


def test_set_multiple_keys():
    job = _FakeJob()

    def exercise():
        perm_ctx_set(a=1, b=2)
        return perm_ctx_get("a"), perm_ctx_get("b")

    assert _run_with_registered_job(job, exercise) == (1, 2)
    assert job.py_env.local_state["_PERM_CTX"] == {"a": 1, "b": 2}


def test_get_no_perm_ctx_raises():
    job = _FakeJob()

    def exercise():
        perm_ctx_get("key")

    with pytest.raises(RuntimeError):
        _run_with_registered_job(job, exercise)


def test_get_no_perm_ctx_with_default():
    job = _FakeJob()

    def exercise():
        return perm_ctx_get("key", "fallback")

    assert _run_with_registered_job(job, exercise) == "fallback"


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


def test_get_wrong_arg_count():
    job = _FakeJob()

    def get_without_args():
        perm_ctx_get()

    def get_with_too_many_args():
        perm_ctx_get("a", "b", "c")

    with pytest.raises(TypeError):
        _run_with_registered_job(job, get_without_args)
    with pytest.raises(TypeError):
        _run_with_registered_job(job, get_with_too_many_args)


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
