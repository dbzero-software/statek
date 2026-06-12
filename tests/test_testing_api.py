"""Tests for the public Statek testing context API."""

import pytest

from statek.testing import statek_context
from statek.utils import get_current_agent, get_current_job, perm_ctx_get, perm_ctx_set


def test_statek_context_sets_current_job():
    """statek_context makes the supplied job visible as the current job."""
    job = _FakeJob()

    with statek_context(job=job):
        assert get_current_job() is job


def test_statek_context_infers_agent_from_job():
    """statek_context infers the current agent from job.job_def.agent."""
    agent = object()
    job = _FakeJob(agent=agent)

    with statek_context(job=job):
        assert get_current_agent() is agent


def test_statek_context_uses_explicit_agent_override():
    """An explicit agent overrides the agent inferred from the job."""
    inferred_agent = object()
    explicit_agent = object()
    job = _FakeJob(agent=inferred_agent)

    with statek_context(job=job, agent=explicit_agent):
        assert get_current_agent() is explicit_agent


def test_statek_context_resets_after_exit():
    """statek_context clears the current job when its scope exits."""
    job = _FakeJob()

    with statek_context(job=job):
        assert get_current_job() is job

    assert get_current_job() is None


def test_statek_context_restores_outer_context():
    """Nested statek_context scopes restore the outer current job."""
    outer_job = _FakeJob()
    inner_job = _FakeJob()

    with statek_context(job=outer_job):
        with statek_context(job=inner_job):
            assert get_current_job() is inner_job
        assert get_current_job() is outer_job


def test_statek_context_resets_after_exception():
    """statek_context resets even when test code raises inside the block."""
    job = _FakeJob()

    with pytest.raises(ValueError, match="boom"):
        with statek_context(job=job):
            assert get_current_job() is job
            raise ValueError("boom")

    assert get_current_job() is None


def test_perm_ctx_helpers_work_under_statek_context():
    """Persistent context helpers use the job established by statek_context."""
    job = _FakeJob()

    with statek_context(job=job):
        perm_ctx_set(answer=42)
        assert perm_ctx_get("answer") == 42

    assert job.py_env.local_state == {"_PERM_CTX": {"answer": 42}}


class _FakeJob:
    """Minimal job stub with job_def and py_env attributes."""

    class _FakeJobDef:
        """Minimal job definition stub."""

        def __init__(self, agent=None):
            self.agent = agent

    class _FakePyEnv:
        """Minimal PyEnv stub with persistent context support."""

        local_state = None

        @property
        def perm_ctx(self):
            return None if self.local_state is None else self.local_state.get("_PERM_CTX")

    def __init__(self, agent=None):
        self.job_def = self._FakeJobDef(agent=agent)
        self.py_env = self._FakePyEnv()
