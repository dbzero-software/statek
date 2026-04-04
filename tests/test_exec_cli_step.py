"""Tests for exec_cli_step function."""

import pytest

from statek.executors.utils import exec_cli_step


def _render_calendar_media(d):
    return f'/tmp/calendar_{d}.png'


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


class TestExecCliStep:
    """Test cases for exec_cli_step function."""

    def create_job(self, job_factory, job_params=None):
        """Helper to create job with default params."""
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_print_goes_to_callback(self, job_factory):
        """Print output is sent to the console_append callback, not the job console."""
        job = self.create_job(job_factory)
        outputs = []
        await exec_cli_step('print("hello")', job, outputs.append)
        assert outputs == ["hello"]
        assert not job.py_env.console

    @pytest.mark.asyncio
    async def test_returns_true_on_exit(self, job_factory):
        """Returns True when exit() is called."""
        job = self.create_job(job_factory)
        result = await exec_cli_step('exit("done")', job, lambda _: None)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_normally(self, job_factory):
        """Returns False when code finishes without exit()."""
        job = self.create_job(job_factory)
        result = await exec_cli_step('x = 1', job, lambda _: None)
        assert result is False

    @pytest.mark.asyncio
    async def test_expression_output_goes_to_callback(self, job_factory):
        """Standalone expressions send their result to the callback (REPL-style)."""
        job = self.create_job(job_factory)
        outputs = []
        await exec_cli_step('var = 123\nvar', job, outputs.append)
        assert outputs == ["123"]
        assert not job.py_env.console

    @pytest.mark.asyncio
    async def test_expression_string_output(self, job_factory):
        """String expressions are output unquoted at top level."""
        job = self.create_job(job_factory)
        outputs = []
        await exec_cli_step('text = "hello"\ntext', job, outputs.append)
        assert outputs == ['hello']
        assert not job.py_env.console

    @pytest.mark.asyncio
    async def test_expression_none_output(self, job_factory):
        """Expressions evaluating to None output 'None'."""
        job = self.create_job(job_factory)
        outputs = []
        await exec_cli_step('result = None\nresult', job, outputs.append)
        assert outputs == ["None"]

    @pytest.mark.asyncio
    async def test_expression_after_function_call(self, job_factory):
        """Expression output works when preceded by a function call from local_state."""
        job = self.create_job(job_factory)

        job.py_env.local_state = {'render_calendar_media': _render_calendar_media}
        outputs = []
        code = ('from datetime import date\n'
                'media_path = render_calendar_media(date(2026, 4, 1))\n'
                'media_path')
        await exec_cli_step(code, job, outputs.append)
        assert len(outputs) == 1
        assert '/tmp/calendar_' in outputs[0]

    @pytest.mark.asyncio
    async def test_shares_job_state(self, job_factory):
        """Execution modifies job local_state like exec_step does."""
        job = self.create_job(job_factory)
        await exec_cli_step('x = 42', job, lambda _: None)
        assert job.py_env.local_state.get('x') == 42
