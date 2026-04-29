"""Tests for python_cli output routing.

exec_all_steps captures python_cli output through its callback. The caller
decides where to persist that output; run_job_step stores it in tool_log
rather than duplicating it into job.py_env.console.
"""

import pytest

from statek.executors.utils import exec_all_steps
from statek.utils import CodeBlock, CallSpec


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


def _make_cli_code_block(code_str):
    """Build a CodeBlock with a single python_cli tool call."""
    call = CallSpec(id="T-001", func_name="python_cli", args=[], kwargs={"code": code_str})
    return CodeBlock(code=None, tool_calls=[call])


class TestCliConsoleAppendRouting:
    """python_cli output is routed to the supplied callback."""

    def create_job(self, job_factory, job_params=None):
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_cli_output_written_to_callback_only(self, job_factory):
        """python_cli print output reaches the callback, not job.py_env.console."""
        job = self.create_job(job_factory)
        cli_outputs = {}

        def _cli_console_append(cli_idx, text):
            cli_outputs.setdefault(cli_idx, []).append(text)

        code = _make_cli_code_block('print("hello")')
        await exec_all_steps(code, job, _cli_console_append)

        assert cli_outputs[0] == ["hello"]
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_cli_execution_does_not_advance_console_position(self, job_factory):
        """Capturing python_cli output does not mutate the job console."""
        job = self.create_job(job_factory)
        pos_before = len(job.py_env.console) if job.py_env.console else 0

        code = _make_cli_code_block('x = 1\nprint("done")')
        await exec_all_steps(code, job, lambda _cli_idx, _text: None)

        pos_after = len(job.py_env.console) if job.py_env.console else 0
        assert pos_after == pos_before

    @pytest.mark.asyncio
    async def test_two_cli_executions_leave_console_position_unchanged(self, job_factory):
        """Sequential python_cli executions still do not touch the job console."""
        job = self.create_job(job_factory)

        code1 = _make_cli_code_block('print("first")')
        await exec_all_steps(code1, job, lambda _cli_idx, _text: None)
        pos_after_first = len(job.py_env.console) if job.py_env.console else 0

        code2 = _make_cli_code_block('print("second")')
        await exec_all_steps(code2, job, lambda _cli_idx, _text: None)
        pos_after_second = len(job.py_env.console) if job.py_env.console else 0

        assert pos_after_second == pos_after_first == 0
