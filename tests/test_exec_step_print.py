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


class TestExecStepFStringFormatting:
    """Verify that f-string expressions in exec_step use format_default_llm_repr."""

    def _make_job(self, job_factory):
        return job_factory(job_params=DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_fstring_object_uses_llm_repr(self, job_factory):
        """F-string expressions use __llm_repr__ for objects."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'obj': _LLMReprObject(name="test")}
        await exec_step('result = f"Object: {obj}"', job)
        assert job.py_env.local_state['result'] == 'Object: custom:test'

    @pytest.mark.asyncio
    async def test_fstring_string_passes_through(self, job_factory):
        """F-string string variables pass through without extra quoting."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'greeting': 'hello'}
        await exec_step('result = f"Say: {greeting}"', job)
        assert job.py_env.local_state['result'] == 'Say: hello'

    @pytest.mark.asyncio
    async def test_fstring_int_formatted_as_str(self, job_factory):
        """F-string integer variables are formatted as plain numbers."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'count': 42}
        await exec_step('result = f"Count: {count}"', job)
        assert job.py_env.local_state['result'] == 'Count: 42'

    @pytest.mark.asyncio
    async def test_fstring_explicit_r_conversion_not_transformed(self, job_factory):
        """F-string with !r conversion uses repr(), not format_default_llm_repr."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'obj': _LLMReprObject(name="test")}
        await exec_step('result = f"{obj!r}"', job)
        assert 'custom:test' not in job.py_env.local_state['result']

    @pytest.mark.asyncio
    async def test_fstring_format_spec_not_transformed(self, job_factory):
        """F-string with explicit format spec is not transformed."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'value': 3.14159}
        await exec_step('result = f"{value:.2f}"', job)
        assert job.py_env.local_state['result'] == '3.14'

    @pytest.mark.asyncio
    async def test_print_fstring_uses_llm_repr(self, job_factory):
        """print() with f-string uses format_default_llm_repr for embedded objects."""
        job = self._make_job(job_factory)
        job.py_env.local_state = {'obj': _LLMReprObject(name="slot")}
        await exec_step('print(f"Object is: {obj}")', job)
        assert job.py_env.console[0] == 'Object is: custom:slot'
