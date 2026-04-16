"""Tests for exec_cli_step function."""

import pytest

from statek.executors.utils import exec_cli_step
from statek.exceptions import FutureError
from tests.test_exec_step import (
    create_future_not_ready, create_future_ready,
    get_value, get_value_not_ready,
)


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

    @pytest.mark.asyncio
    async def test_expression_string_output(self, job_factory):
        """String expressions are output unquoted at top level."""
        job = self.create_job(job_factory)
        outputs = []
        await exec_cli_step('text = "hello"\ntext', job, outputs.append)
        assert outputs == ['hello']

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

    # --- Temporal function tests ---

    @pytest.mark.asyncio
    async def test_future_error_raised_on_not_ready_access(self, job_factory):
        """Accessing a not-ready FutureResult in print raises FutureError."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'future_val': create_future_not_ready()}

        with pytest.raises(FutureError) as exc_info:
            await exec_cli_step('print(future_val)', job, lambda _: None)

        assert exc_info.value.instr_num == 0

    @pytest.mark.asyncio
    async def test_future_error_has_correct_instr_num(self, job_factory):
        """FutureError carries the instruction index where access failed."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'future_val': create_future_not_ready()}
        code = 'x = 1\nprint(future_val)'

        with pytest.raises(FutureError) as exc_info:
            await exec_cli_step(code, job, lambda _: None)

        # x = 1 is instruction 0, print is instruction 1
        assert exc_info.value.instr_num == 1

    @pytest.mark.asyncio
    async def test_ready_future_unwrapped_in_print(self, job_factory):
        """A ready FutureResult is unwrapped to its value inside print."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'future_val': create_future_ready(42)}
        outputs = []
        await exec_cli_step('print(future_val)', job, outputs.append)
        assert outputs == ["42"]

    @pytest.mark.asyncio
    async def test_temporal_function_returns_future_result(self, job_factory):
        """Calling a temporal function stores a FutureResult in local state."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value': get_value}
        await exec_cli_step('result = get_value()', job, lambda _: None)
        assert job.py_env.local_state.get('result') is not None

    @pytest.mark.asyncio
    async def test_temporal_not_ready_blocks_on_access(self, job_factory):
        """Temporal function returning not-ready future raises FutureError on use."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}
        code = 'result = get_value_not_ready()\nprint(result)'

        with pytest.raises(FutureError) as exc_info:
            await exec_cli_step(code, job, lambda _: None)

        # Assignment is instruction 0, print is instruction 1
        assert exc_info.value.instr_num == 1

    @pytest.mark.asyncio
    async def test_continuation_after_future_error(self, job_factory):
        """Resuming from instr_num skips already-executed instructions."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'future_val': create_future_not_ready()}
        code = 'x = 1\nprint(future_val)'

        # First run — blocks at instruction 1
        with pytest.raises(FutureError):
            await exec_cli_step(code, job, lambda _: None)
        assert job.py_env.local_state.get('x') == 1

        # Replace with ready future and resume from instruction 1
        job.py_env.local_state['future_val'] = create_future_ready(99)
        outputs = []
        await exec_cli_step(code, job, outputs.append, instr_num=1)
        assert outputs == ["99"]

    @pytest.mark.asyncio
    async def test_ready_future_expression_output(self, job_factory):
        """A ready FutureResult as a standalone expression outputs its value."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value': get_value}
        outputs = []
        code = 'result = get_value()\nresult'
        await exec_cli_step(code, job, outputs.append)
        assert outputs == ["42"]

    @pytest.mark.asyncio
    async def test_error_output_goes_to_callback(self, job_factory):
        """Errors from FutureError-unrelated exceptions go to callback, not job console."""
        job = self.create_job(job_factory)
        outputs = []
        with pytest.raises(NameError):
            await exec_cli_step('undefined_var', job, outputs.append)
        assert any("NameError" in o for o in outputs)
