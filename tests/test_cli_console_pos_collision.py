"""Tests for console_pos collision bug in DIRECT chat style.

In DIRECT style, python_cli output must reach job.py_env.console so the
console position advances between LLM turns.  Without this, multiple
LLM_LogItems share the same console_pos and a single exception stored at
that shared position is attributed to every item, inflating
max_consecutive_exceptions.

In run_job_step the CLI output is captured during execution and pushed
to job.py_env.console at the end of the step (not during).
"""

import pytest

from statek.executors.utils import exec_all_steps
from statek.utils import CodeBlock, CallSpec


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


def _make_cli_code_block(code_str):
    """Build a CodeBlock with a single python_cli tool call."""
    call = CallSpec(id="T-001", func_name="python_cli", args=[], kwargs={"code": code_str})
    return CodeBlock(code=None, tool_calls=[call])


class TestCliConsoleAppendAdvancesPosition:
    """_cli_console_append must mirror output to job.py_env.console."""

    def create_job(self, job_factory, job_params=None):
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_cli_output_written_to_job_console(self, job_factory):
        """python_cli print output must appear in job.py_env.console."""
        job = self.create_job(job_factory)
        cli_outputs = {}

        def _cli_console_append(cli_idx, text):
            cli_outputs.setdefault(cli_idx, []).append(text)
            job.py_env.console_append(text)

        code = _make_cli_code_block('print("hello")')
        await exec_all_steps(code, job, _cli_console_append)

        assert cli_outputs[0] == ["hello"]
        assert job.py_env.console == ["hello"]

    @pytest.mark.asyncio
    async def test_console_pos_advances_after_cli_execution(self, job_factory):
        """After CLI execution the console length must have grown, so the next
        LLM_LogItem gets a different console_pos."""
        job = self.create_job(job_factory)
        pos_before = len(job.py_env.console) if job.py_env.console else 0

        def _cli_console_append(_cli_idx, text):
            job.py_env.console_append(text)

        code = _make_cli_code_block('x = 1\nprint("done")')
        await exec_all_steps(code, job, _cli_console_append)

        pos_after = len(job.py_env.console) if job.py_env.console else 0
        assert pos_after > pos_before

    @pytest.mark.asyncio
    async def test_two_cli_executions_produce_different_positions(self, job_factory):
        """Two sequential CLI executions must yield different console positions."""
        job = self.create_job(job_factory)

        def _cli_console_append(_cli_idx, text):
            job.py_env.console_append(text)

        code1 = _make_cli_code_block('print("first")')
        await exec_all_steps(code1, job, _cli_console_append)
        pos_after_first = len(job.py_env.console) if job.py_env.console else 0

        code2 = _make_cli_code_block('print("second")')
        await exec_all_steps(code2, job, _cli_console_append)
        pos_after_second = len(job.py_env.console) if job.py_env.console else 0

        assert pos_after_second > pos_after_first
