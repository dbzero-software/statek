"""Tests for perm_ctx_set and perm_ctx_get."""

import pytest

from statek.utils import perm_ctx_set, perm_ctx_set_unique, perm_ctx_get, _statek_ctx_scope


def _run_with_registered_job(job, func):
    """Run func while a fake current job is visible through Statek context."""
    with _statek_ctx_scope({"job": job}):
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


def test_set_creates_perm_ctx_on_demand():
    job = _FakeJob()
    with _statek_ctx_scope({"job": job}):
        perm_ctx_set(x=10)
        assert perm_ctx_get("x") == 10


def test_set_creates_perm_ctx_in_pyenv_local_state():
    job = _FakeJob()
    assert job.py_env.local_state is None
    with _statek_ctx_scope({"job": job}):
        perm_ctx_set(key="value")
    assert job.py_env.local_state["_PERM_CTX"] == {"key": "value"}


def test_set_without_job_context_raises_runtime_error():
    with pytest.raises(RuntimeError):
        perm_ctx_set(key="value")


def test_get_without_job_context_raises_runtime_error():
    with pytest.raises(RuntimeError):
        perm_ctx_get("key")


def test_get_without_job_context_returns_default():
    assert perm_ctx_get("key", "fallback") == "fallback"


def test_set_unique_without_job_context_raises_runtime_error():
    with pytest.raises(RuntimeError):
        perm_ctx_set_unique("result", "value")


def test_set_unique_uses_key_when_available():
    job = _FakeJob()

    def exercise():
        assigned = perm_ctx_set_unique("result", 123)
        return assigned, perm_ctx_get("result")

    assert _run_with_registered_job(job, exercise) == ("result", 123)
    assert job.py_env.local_state["_PERM_CTX"] == {"result": 123}


def test_set_unique_suffixes_when_perm_ctx_key_exists():
    job = _FakeJob()
    job.py_env.local_state = {"_PERM_CTX": {"result": "old", "result_1": "older"}}

    def exercise():
        assigned = perm_ctx_set_unique("result", "new")
        return assigned, perm_ctx_get(assigned)

    assert _run_with_registered_job(job, exercise) == ("result_2", "new")
    assert job.py_env.local_state["_PERM_CTX"] == {
        "result": "old",
        "result_1": "older",
        "result_2": "new",
    }


def test_set_unique_suffixes_when_local_name_exists():
    job = _FakeJob()

    def exercise():
        result = "local value"
        assigned = perm_ctx_set_unique("result", "persistent value")
        assert result == "local value"
        return assigned, perm_ctx_get(assigned)

    assert _run_with_registered_job(job, exercise) == ("result_1", "persistent value")


def test_set_unique_checks_local_context_names():
    job = _FakeJob()

    def exercise():
        return _call_with_local_context({"result": "context value"})

    def _call_with_local_context(_local_context):
        assigned = perm_ctx_set_unique("result", "persistent value")
        return assigned, perm_ctx_get(assigned)

    assert _run_with_registered_job(job, exercise) == ("result_1", "persistent value")


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
