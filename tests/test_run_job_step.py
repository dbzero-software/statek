"""Tests for run_job_step resumption after FutureError."""
# pylint: disable=no-member,R0903

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import dbzero as db0

from statek.agents.agent import Agent
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.utils import run_job_step
from statek.future import FutureResult
from statek.exceptions import FutureError
from statek.llm_api import LLM_Response, LLM_Stats, CallParams
from statek.utils import CodeBlock, CallSpec


@db0.memo
@dataclass
class MemoObject:
    value: int = 0


def _check_condition_false(_):
    return False


def _fetch_result_not_ready(future_result):
    raise FutureError(future_result=future_result)


def _check_condition_true(_):
    return True


def _fetch_result_from_deps(self):
    return self.deps.value


def create_future_not_ready():
    """Create a FutureResult that raises FutureError when accessed."""
    future = FutureResult(deps=MemoObject(value=0), state_num=0)
    future.set_complement_functions(
        complement=_fetch_result_not_ready,
        condition=_check_condition_false
    )
    return future


def create_future_ready(value=42):
    """Create a ready FutureResult with a specific value."""
    future = FutureResult(deps=MemoObject(value=value), state_num=0)
    future.set_complement_functions(
        complement=_fetch_result_from_deps,
        condition=_check_condition_true
    )
    return future


class TestRunJobStepFutureErrorResumption:
    """Test that run_job_step resumes execution from the exact line that threw FutureError."""

    @pytest.mark.asyncio
    async def test_resumes_from_future_error_line(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Test that execution resumes from the exact line that threw FutureError,
        skipping already-executed instructions."""
        # Code block with multiple statements: the FutureError occurs on the
        # third instruction (idx=2). On resume, instructions 0 and 1 must be
        # skipped so side-effects (like the counter increment) don't repeat.
        code = (
            'counter = counter + 1\n'      # idx 0 - should run once
            'before_flag = True\n'          # idx 1 - should run once
            'result = future_val\n'         # idx 2 - raises FutureError first time
            'after_flag = True'             # idx 3 - runs only after resume
        )
        job_def = job_def_factory(warmup_code=[code, 'exit("ok")'])
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY
        )

        # Seed the environment with initial values
        future_not_ready = create_future_not_ready()
        job.py_env.local_state['counter'] = 0
        job.py_env.local_state['before_flag'] = False
        job.py_env.local_state['after_flag'] = False
        job.py_env.local_state['future_val'] = future_not_ready

        # First run: executes idx 0 (counter=1), idx 1 (before_flag=True),
        # then idx 2 raises FutureError
        result1 = await run_job_step(job)
        assert result1 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state['counter'] == 1
        assert job.py_env.local_state['before_flag'] is True
        assert job.py_env.local_state['after_flag'] is False
        assert job.awaited_result is future_not_ready
        assert job.next_instr_num == 2  # suspended at idx 2
        assert job.warmup_block_num is None  # block did not advance (None means block 0)

        # Make the future ready and replace it in env
        future_ready = create_future_ready(99)
        job.py_env.local_state['future_val'] = future_ready

        # Second run: resumes from idx 2, skipping idx 0 and 1.
        # counter must stay 1 (not incremented again).
        result2 = await run_job_step(job)
        assert result2 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state['counter'] == 1  # NOT 2 — proves skip
        assert job.py_env.local_state['result'] == 99
        assert job.py_env.local_state['after_flag'] is True
        assert job.awaited_result is None
        assert job.next_instr_num is None
        assert job.warmup_block_num == 1  # block advanced

        # Final block: exit
        result3 = await run_job_step(job)
        assert result3 is True
        assert job.status == JobStatus.DONE


class TestRunJobStepToolCallResponse:
    """Test that run_job_step stores a CodeBlock when the LLM responds with tool calls."""

    @pytest.mark.asyncio
    async def test_stores_code_block_when_tool_calls_present(
        self, job_def_factory, db0_fixture  # pylint: disable=unused-argument
    ):
        """append_chat_log receives a CodeBlock when call_requests is non-empty."""
        job_def = job_def_factory()
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.STARTED,
        )

        call_params = CallParams(call_id="T-001", name="my_tool", args=[], kwargs={"x": 1})
        mock_response = LLM_Response(
            text="",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=[call_params],
        )

        mock_api = MagicMock()
        mock_api.process_request = AsyncMock(return_value=mock_response)
        mock_api.build_messages.return_value = []

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            mock_llm_api_cls.get.return_value = mock_api
            await run_job_step(job, provider="OPENROUTER")

        assert len(job.chat_log) == 1
        llm_resp = job.chat_log[0].llm_resp
        assert isinstance(llm_resp, CodeBlock)
        assert len(llm_resp.tool_calls) == 1
        assert llm_resp.tool_calls[0].func_name == "my_tool"

    @pytest.mark.asyncio
    async def test_stores_str_when_no_tool_calls(
        self, job_def_factory, db0_fixture  # pylint: disable=unused-argument
    ):
        """append_chat_log receives a plain str when call_requests is None."""
        job_def = job_def_factory()
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.STARTED,
        )

        mock_response = LLM_Response(
            text="x = 42",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )

        mock_api = MagicMock()
        mock_api.process_request = AsyncMock(return_value=mock_response)
        mock_api.build_messages.return_value = []

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            mock_llm_api_cls.get.return_value = mock_api
            await run_job_step(job, provider="OPENROUTER")

        assert len(job.chat_log) == 1
        llm_resp = job.chat_log[0].llm_resp
        assert isinstance(llm_resp, str)
        assert llm_resp == "x = 42"


def _make_job_with_tool(role, tool_name, tool_fn, warmup_code):
    """Create a Job with a named tool in agent context and given warmup code."""
    agent = Agent(role=role, _system_prompt="Test", _tools=[])
    agent.context[tool_name] = tool_fn
    job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
    return Job(
        job_def=job_def,
        model_family="test",
        model="test-model",
        job_status=JobStatus.READY,
    )


class TestRunJobStepToolExecution:
    """Tests for step #5: tool execution before code and tool_log population."""

    @pytest.mark.asyncio
    async def test_tool_result_stored_in_tool_log(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool results from a warmup CodeBlock are stored in tool_log at key 0."""
        call_spec = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[call_spec])
        job = _make_job_with_tool("role_tl", "my_tool", lambda: "result_value", warmup_code)

        result = await run_job_step(job)

        assert result is True
        assert job.py_env.tool_log is not None
        assert 0 in job.py_env.tool_log
        assert "'result_value'" in job.py_env.tool_log[0]

    @pytest.mark.asyncio
    async def test_multiple_tool_results_stored_as_list(self, db0_fixture):  # pylint: disable=unused-argument
        """Multiple tool calls produce a List[str] in tool_log."""
        cs1 = CallSpec(id="STATEK-001", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="tool_b", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs1, cs2])
        agent = Agent(role="role_multi", _system_prompt="Test", _tools=[])
        agent.context["tool_a"] = lambda: "alpha"
        agent.context["tool_b"] = lambda: "beta"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.READY,
        )

        await run_job_step(job)

        stored = job.py_env.tool_log[0]
        assert not isinstance(stored, str)
        assert len(stored) == 2
        assert "'alpha'" in stored[0]
        assert "'beta'" in stored[1]

    @pytest.mark.asyncio
    async def test_tool_not_executed_on_continuation(self, db0_fixture):  # pylint: disable=unused-argument
        """Tools are not re-executed when next_instr_num is set (FutureError continuation)."""
        call_count = 0

        def counting_tool():
            nonlocal call_count
            call_count += 1
            return "counted"

        call_spec = CallSpec(id="STATEK-001", func_name="counting_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[call_spec])
        job = _make_job_with_tool("role_cont", "counting_tool", counting_tool, warmup_code)
        # Simulate a FutureError continuation: status stays WARMING_UP, next_instr_num is set
        job.set_status(JobStatus.WARMING_UP)
        job.next_instr_num = 0

        await run_job_step(job)

        assert call_count == 0

    @pytest.mark.asyncio
    async def test_no_tool_calls_leaves_tool_log_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Warmup code without tool calls does not create tool_log."""
        agent = Agent(role="role_no_tc", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code='exit("ok")')
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.READY,
        )

        await run_job_step(job)

        assert job.py_env.tool_log is None
