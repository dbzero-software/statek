"""Tests for statek_print integration in exec_step print() context."""

from dataclasses import dataclass
import pytest
import dbzero as db0

from statek.executors.utils import exec_step


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


@db0.memo
@dataclass
class _LLMReprObject:  # pylint: disable=too-few-public-methods
    name: str

    def __llm_repr__(self):
        return f"custom:{self.name}"


class TestExecStepPrint:
    """Verify that print() inside exec_step uses statek_print formatting."""

    def _make_job(self, job_factory):
        return job_factory(job_params=DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_print_string_outputs_raw(self, job_factory):
        """print() outputs strings as-is, without wrapping quotes."""
        job = self._make_job(job_factory)
        await exec_step('print("hello")', job)
        assert job.py_env.console[0] == 'hello'

    @pytest.mark.asyncio
    async def test_print_object_uses_llm_repr(self, job_factory):
        """print() calls __llm_repr__ when available, as statek_print does."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'obj': _LLMReprObject(name="test")}
        await exec_step('print(obj)', job)
        assert job.py_env.console[0] == "custom:test"
