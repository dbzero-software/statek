"""Tests for exec_all_steps function."""

import pytest

from statek.executors.utils import exec_all_steps
from statek.exceptions import FutureError
from statek.utils import CodeBlock, CallSpec
from tests.test_exec_step import (
    create_future_not_ready, create_future_ready,
    get_value, get_value_not_ready,
)


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


class TestExecAllSteps:
    """Test cases for exec_all_steps function."""

    def create_job(self, job_factory, job_params=None):
        """Helper to create job with default params."""
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_plain_string_executes_via_exec_step(self, job_factory):
        """A plain string code block runs through exec_step on the job console."""
        job = self.create_job(job_factory)
        result = await exec_all_steps('x = 42', job, lambda idx, s: None)
        assert result is False
        assert job.py_env.local_state.get('x') == 42

    @pytest.mark.asyncio
    async def test_code_block_code_runs_via_exec_step(self, job_factory):
        """CodeBlock.code is executed through exec_step (job console)."""
        job = self.create_job(job_factory)
        block = CodeBlock(code='print("hello")', tool_calls=[])
        result = await exec_all_steps(block, job, lambda idx, s: None)
        assert result is False
        assert job.py_env.console == ["hello"]

    @pytest.mark.asyncio
    async def test_cli_tool_calls_routed_to_callback(self, job_factory):
        """python_cli tool calls route output to the console_append callback."""
        job = self.create_job(job_factory)
        cs = CallSpec(id="1", func_name="python_cli", kwargs={"code": 'print("cli-out")'})
        block = CodeBlock(code=None, tool_calls=[cs])
        outputs = []
        await exec_all_steps(block, job, lambda idx, s: outputs.append((idx, s)))
        assert outputs == [(0, "cli-out")]

    @pytest.mark.asyncio
    async def test_exit_in_regular_code_stops_execution(self, job_factory):
        """exit() in regular code prevents subsequent CLI steps from running."""
        job = self.create_job(job_factory)
        cs = CallSpec(id="1", func_name="python_cli", kwargs={"code": 'y = 99'})
        block = CodeBlock(code='exit("done")', tool_calls=[cs])
        result = await exec_all_steps(block, job, lambda idx, s: None)
        assert result is True
        assert job.py_env.local_state.get('y') is None

    @pytest.mark.asyncio
    async def test_exit_in_cli_step_stops_remaining(self, job_factory):
        """exit() in a CLI step prevents subsequent CLI steps from running."""
        job = self.create_job(job_factory)
        cs1 = CallSpec(id="1", func_name="python_cli", kwargs={"code": 'exit("done")'})
        cs2 = CallSpec(id="2", func_name="python_cli", kwargs={"code": 'y = 99'})
        block = CodeBlock(code=None, tool_calls=[cs1, cs2])
        result = await exec_all_steps(block, job, lambda idx, s: None)
        assert result is True
        assert job.py_env.local_state.get('y') is None

    @pytest.mark.asyncio
    async def test_multiple_cli_steps_with_correct_indices(self, job_factory):
        """Each CLI step's output carries the correct 0-based tool index."""
        job = self.create_job(job_factory)
        cs0 = CallSpec(id="a", func_name="python_cli", kwargs={"code": 'print("zero")'})
        cs1 = CallSpec(id="b", func_name="python_cli", kwargs={"code": 'print("one")'})
        block = CodeBlock(code=None, tool_calls=[cs0, cs1])
        outputs = []
        await exec_all_steps(block, job, lambda idx, s: outputs.append((idx, s)))
        assert outputs == [(0, "zero"), (1, "one")]

    @pytest.mark.asyncio
    async def test_future_error_in_cli_step_has_tuple_instr_num(self, job_factory):
        """FutureError from a CLI step carries (cli_step_index, instr_num) tuple."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}
        cs = CallSpec(id="1", func_name="python_cli",
                      kwargs={"code": "result = get_value_not_ready()\nprint(result)"})
        block = CodeBlock(code=None, tool_calls=[cs])
        with pytest.raises(FutureError) as exc_info:
            await exec_all_steps(block, job, lambda idx, s: None)
        assert isinstance(exc_info.value.instr_num, tuple)
        assert exc_info.value.instr_num[0] == 0  # first cli step

    @pytest.mark.asyncio
    async def test_regular_tool_calls_are_skipped(self, job_factory):
        """Non-python_cli tool calls are not executed by exec_all_steps."""
        job = self.create_job(job_factory)
        cs_regular = CallSpec(id="1", func_name="my_tool", kwargs={"x": 1})
        cs_cli = CallSpec(id="2", func_name="python_cli", kwargs={"code": 'x = 1'})
        block = CodeBlock(code=None, tool_calls=[cs_regular, cs_cli])
        await exec_all_steps(block, job, lambda idx, s: None)
        assert job.py_env.local_state.get('x') == 1

    @pytest.mark.asyncio
    async def test_continuation_with_int_instr_num(self, job_factory):
        """An int instr_num is forwarded to exec_step for regular code continuation."""
        job = self.create_job(job_factory)
        block = CodeBlock(code='x = 1\nx = 2', tool_calls=[])
        await exec_all_steps(block, job, lambda idx, s: None, instr_num=1)
        # Skipped x=1, only ran x=2
        assert job.py_env.local_state.get('x') == 2

    @pytest.mark.asyncio
    async def test_continuation_with_tuple_instr_num(self, job_factory):
        """A tuple instr_num skips regular code and resumes at the right CLI step."""
        job = self.create_job(job_factory)
        cs0 = CallSpec(id="a", func_name="python_cli", kwargs={"code": 'a = 1'})
        cs1 = CallSpec(id="b", func_name="python_cli", kwargs={"code": 'b = 2\nb = 3'})
        block = CodeBlock(code='x = 99', tool_calls=[cs0, cs1])
        # Continue from cli step 1, instruction 1 (skip b=2, run b=3)
        await exec_all_steps(block, job, lambda idx, s: None, instr_num=(1, 1))
        assert job.py_env.local_state.get('x') is None  # regular code skipped
        assert job.py_env.local_state.get('a') is None   # cli step 0 skipped
        assert job.py_env.local_state.get('b') == 3      # resumed at instr 1

    @pytest.mark.asyncio
    async def test_cli_expression_output_routed_to_callback(self, job_factory):
        """Standalone expressions in python_cli produce REPL-style output via callback."""
        job = self.create_job(job_factory)
        cs = CallSpec(id="1", func_name="python_cli",
                      kwargs={"code": "var = 123;var"})
        block = CodeBlock(code=None, tool_calls=[cs])
        outputs = []
        await exec_all_steps(block, job, lambda idx, s: outputs.append((idx, s)))
        assert outputs == [(0, "123")]

    # --- Temporal function tests: multiple CLI steps ---

    @pytest.mark.asyncio
    async def test_future_error_in_first_cli_step_blocks_remaining(self, job_factory):
        """FutureError in the first CLI step prevents subsequent CLI steps from running."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}
        cs0 = CallSpec(id="a", func_name="python_cli",
                       kwargs={"code": "result = get_value_not_ready()\nprint(result)"})
        cs1 = CallSpec(id="b", func_name="python_cli",
                       kwargs={"code": "y = 99"})
        block = CodeBlock(code=None, tool_calls=[cs0, cs1])

        with pytest.raises(FutureError) as exc_info:
            await exec_all_steps(block, job, lambda idx, s: None)

        # First CLI step (index 0), second instruction (print)
        assert exc_info.value.instr_num == (0, 1)
        # Second CLI step never ran
        assert job.py_env.local_state.get('y') is None

    @pytest.mark.asyncio
    async def test_future_error_in_second_cli_step(self, job_factory):
        """First CLI step succeeds; FutureError in second carries correct tuple."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}
        cs0 = CallSpec(id="a", func_name="python_cli",
                       kwargs={"code": "x = 1"})
        cs1 = CallSpec(id="b", func_name="python_cli",
                       kwargs={"code": "result = get_value_not_ready()\nprint(result)"})
        block = CodeBlock(code=None, tool_calls=[cs0, cs1])

        with pytest.raises(FutureError) as exc_info:
            await exec_all_steps(block, job, lambda idx, s: None)

        # Second CLI step (index 1), second instruction (print)
        assert exc_info.value.instr_num == (1, 1)
        # First CLI step completed
        assert job.py_env.local_state.get('x') == 1

    @pytest.mark.asyncio
    async def test_continuation_resumes_blocked_cli_step(self, job_factory):
        """Tuple instr_num resumes at the correct CLI step and instruction."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'future_val': create_future_not_ready()}
        cs0 = CallSpec(id="a", func_name="python_cli",
                       kwargs={"code": "a = 1"})
        cs1 = CallSpec(id="b", func_name="python_cli",
                       kwargs={"code": "b = 2\nprint(future_val)"})
        block = CodeBlock(code=None, tool_calls=[cs0, cs1])

        # First run — blocks at cli step 1, instruction 1
        with pytest.raises(FutureError) as exc_info:
            await exec_all_steps(block, job, lambda idx, s: None)
        assert exc_info.value.instr_num == (1, 1)
        assert job.py_env.local_state.get('b') == 2

        # Replace with ready future and resume
        job.py_env.local_state['future_val'] = create_future_ready(77)
        outputs = []
        await exec_all_steps(block, job,
                             lambda idx, s: outputs.append((idx, s)),
                             instr_num=(1, 1))
        assert (1, "77") in outputs

    @pytest.mark.asyncio
    async def test_multiple_cli_steps_each_with_temporal_sequential_blocking(self, job_factory):
        """Two CLI steps each call a temporal function; first blocks the entire step."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {
            'get_value_not_ready': get_value_not_ready,
            'get_value': get_value,
        }
        cs0 = CallSpec(id="a", func_name="python_cli",
                       kwargs={"code": "r0 = get_value_not_ready()\nprint(r0)"})
        cs1 = CallSpec(id="b", func_name="python_cli",
                       kwargs={"code": "r1 = get_value()\nprint(r1)"})
        block = CodeBlock(code=None, tool_calls=[cs0, cs1])

        # The not-ready future in cli step 0 blocks before cli step 1 executes
        with pytest.raises(FutureError) as exc_info:
            await exec_all_steps(block, job, lambda idx, s: None)

        assert exc_info.value.instr_num == (0, 1)
        # cli step 1 never ran — r1 not set
        assert job.py_env.local_state.get('r1') is None

    @pytest.mark.asyncio
    async def test_ready_temporal_in_cli_step(self, job_factory):
        """A ready temporal function in a CLI step produces output normally."""
        job = self.create_job(job_factory)
        job.py_env.local_state = {'get_value': get_value}
        cs = CallSpec(id="1", func_name="python_cli",
                      kwargs={"code": "result = get_value()\nresult"})
        block = CodeBlock(code=None, tool_calls=[cs])
        outputs = []
        await exec_all_steps(block, job, lambda idx, s: outputs.append((idx, s)))
        assert (0, "42") in outputs
