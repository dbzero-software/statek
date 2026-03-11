"""Tests for run_job_step resumption after FutureError."""
# pylint: disable=no-member,R0903,C0415

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


class TestRunJobStepToolCallLogging:
    """Tests that tool calls and their results are logged in run_job_step."""

    @staticmethod
    def _make_log_settings(tmp_path):
        """Create mock settings that enable file logging to tmp_path."""
        mock_settings = MagicMock()
        mock_settings.logs_path = str(tmp_path)
        mock_settings.chat_style = None  # defaults to CONSOLE style
        mock_settings.get_xml_box_tags.return_value = {}
        return mock_settings

    @staticmethod
    def _read_logs(tmp_path):
        """Return concatenated content of all .log files under tmp_path."""
        return "".join(f.read_text() for f in tmp_path.glob("*.log"))

    @pytest.mark.asyncio
    async def test_warmup_tool_call_logged_with_statek_marker(self, db0_fixture, tmp_path):  # pylint: disable=unused-argument
        """Warmup tool call is logged with #STATEK: as tool marker."""
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_tc1", "my_tool", lambda: "result_value", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert '#STATEK: as tool' in content
        assert 'my_tool()' in content

    @pytest.mark.asyncio
    async def test_warmup_tool_call_result_in_log(self, db0_fixture, tmp_path):  # pylint: disable=unused-argument
        """Warmup tool call result appears in the log after the call line."""
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_tc2", "my_tool", lambda: "tool_result", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert "'tool_result'" in content

    @pytest.mark.asyncio
    async def test_warmup_tool_result_prefixed_with_gt_in_console_style(  # pylint: disable=unused-argument
        self, db0_fixture, tmp_path
    ):
        """Tool call result lines are prefixed with '> ' in CONSOLE (default) style."""
        cs = CallSpec(id="STATEK-001", func_name="fetch", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_tc3", "fetch", lambda: "fetched", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert '> ' in content

    @pytest.mark.asyncio
    async def test_tool_call_line_wrapped_in_python_fence_in_markdown_style(
        self, db0_fixture, tmp_path  # pylint: disable=unused-argument
    ):
        """Tool call line is wrapped in ```python fences in MARKDOWN style."""
        from statek.settings import ChatStyle  # pylint: disable=import-outside-toplevel
        cs = CallSpec(id="STATEK-001", func_name="md_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_md_fence", "md_tool", lambda: "res", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        mock_settings.chat_style = ChatStyle.MARKDOWN  # pylint: disable=no-member
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert '```python\nmd_tool()  #STATEK: as tool\n```' in content

    @pytest.mark.asyncio
    async def test_tool_call_line_not_fenced_in_console_style(
        self, db0_fixture, tmp_path  # pylint: disable=unused-argument
    ):
        """Tool call line is NOT wrapped in python fences in CONSOLE style."""
        cs = CallSpec(id="STATEK-001", func_name="plain_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_console_fence", "plain_tool", lambda: "res", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert 'plain_tool()  #STATEK: as tool' in content
        assert '```python\nplain_tool()' not in content

    @pytest.mark.asyncio
    async def test_multiple_warmup_tool_calls_all_logged(self, db0_fixture, tmp_path):  # pylint: disable=unused-argument
        """Multiple warmup tool calls each appear in the log."""
        cs1 = CallSpec(id="STATEK-001", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="tool_b", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs1, cs2])
        agent = Agent(role="log_multi", _system_prompt="Test", _tools=[])
        agent.context["tool_a"] = lambda: "alpha"
        agent.context["tool_b"] = lambda: "beta"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert 'tool_a()' in content
        assert 'tool_b()' in content
        assert content.count('#STATEK: as tool') == 2

    @pytest.mark.asyncio
    async def test_tool_call_with_kwargs_formatted_in_log(self, db0_fixture, tmp_path):  # pylint: disable=unused-argument
        """Tool call with kwargs is formatted as func(key='val') in the log."""
        cs = CallSpec(id="STATEK-001", func_name="search", args=[], kwargs={"query": "test"})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_kwargs", "search",
                                   lambda query=None: "found", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert "search(query='test')" in content

    @pytest.mark.asyncio
    async def test_no_tool_calls_no_statek_log_entry(self, db0_fixture, tmp_path):  # pylint: disable=unused-argument
        """No #STATEK: as tool entries in the log when warmup has no tool calls."""
        agent = Agent(role="log_no_tc", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code='exit("ok")')
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert '#STATEK: as tool' not in content

    @pytest.mark.asyncio
    async def test_tool_call_log_appears_before_console_output(self, db0_fixture, tmp_path):  # pylint: disable=unused-argument
        """Tool call log entry appears before the console output in the log file."""
        cs = CallSpec(id="STATEK-001", func_name="setup", args=[], kwargs={})
        warmup_code = CodeBlock(code='print("printed_output")\nexit("ok")', tool_calls=[cs])
        agent = Agent(role="log_order", _system_prompt="Test", _tools=[])
        agent.context["setup"] = lambda: "ready"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        mock_settings = self._make_log_settings(tmp_path)
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        statek_pos = content.find('#STATEK: as tool')
        console_pos = content.find('> printed_output')
        assert statek_pos != -1
        assert console_pos != -1
        assert statek_pos < console_pos


    @pytest.mark.asyncio
    async def test_tool_result_wrapped_in_xml_tags_console_style(
        self, db0_fixture, tmp_path  # pylint: disable=unused-argument
    ):
        """Tool call result is wrapped in XML console tags when xml_box_console is set."""
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_xml_console", "my_tool", lambda: "xml_result", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        mock_settings.get_xml_box_tags.return_value = {"console": "output"}
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert '<output>' in content
        assert '</output>' in content
        assert 'xml_result' in content

    @pytest.mark.asyncio
    async def test_tool_result_wrapped_in_xml_tags_markdown_style(
        self, db0_fixture, tmp_path  # pylint: disable=unused-argument
    ):
        """Tool call result is wrapped in XML console tags in MARKDOWN style too."""
        from statek.settings import ChatStyle  # pylint: disable=import-outside-toplevel
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_xml_md", "my_tool", lambda: "md_result", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        mock_settings.chat_style = ChatStyle.MARKDOWN  # pylint: disable=no-member
        mock_settings.get_xml_box_tags.return_value = {"console": "console_out"}
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        assert '<console_out>' in content
        assert '</console_out>' in content
        assert 'md_result' in content

    @pytest.mark.asyncio
    async def test_tool_result_xml_console_tag_wraps_result_not_call_line(
        self, db0_fixture, tmp_path  # pylint: disable=unused-argument
    ):
        """XML console tags wrap only the result, not the call line itself."""
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_code = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job_with_tool("log_xml_pos", "my_tool", lambda: "the_result", warmup_code)

        mock_settings = self._make_log_settings(tmp_path)
        mock_settings.get_xml_box_tags.return_value = {"console": "out"}
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings), \
             patch('statek.executors.utils.get_statek_settings', return_value=mock_settings):
            await run_job_step(job)

        content = self._read_logs(tmp_path)
        tool_marker_pos = content.find('#STATEK: as tool')
        open_tag_pos = content.find('<out>')
        assert tool_marker_pos != -1
        assert open_tag_pos != -1
        assert tool_marker_pos < open_tag_pos


class TestRunJobStepWarmupException:
    """Tests for critical failure handling when warmup code raises a non-FutureError exception."""

    @pytest.mark.asyncio
    async def test_warmup_exception_sets_job_done(self, db0_fixture):  # pylint: disable=unused-argument
        """A non-FutureError exception in warmup sets job status to DONE."""
        agent = Agent(role="warmup_exc_done", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code="raise ValueError('boom')")
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        await run_job_step(job)

        assert job.status == JobStatus.DONE

    @pytest.mark.asyncio
    async def test_warmup_exception_calls_set_error_on_job_def(self, db0_fixture):  # pylint: disable=unused-argument
        """A non-FutureError exception in warmup calls job_def.set_error with the exception."""
        agent = Agent(role="warmup_exc_set_error", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code="raise RuntimeError('critical')")
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        await run_job_step(job)

        assert job_def.has_errors() is True
        errors = list(job_def.get_errors())
        assert len(errors) == 1
        assert "critical" in errors[0].error_message

    @pytest.mark.asyncio
    async def test_warmup_exception_returns_true(self, db0_fixture):  # pylint: disable=unused-argument
        """run_job_step returns True when warmup raises a non-FutureError exception."""
        agent = Agent(role="warmup_exc_ret", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code="raise TypeError('bad type')")
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        result = await run_job_step(job)

        assert result is True

    @pytest.mark.asyncio
    async def test_warmup_exception_does_not_call_llm(self, db0_fixture):  # pylint: disable=unused-argument
        """No LLM API call is made when warmup raises a non-FutureError exception."""
        agent = Agent(role="warmup_exc_no_llm", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code="raise ValueError('no llm')")
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            await run_job_step(job)
            mock_llm_api_cls.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_warmup_future_error_still_suspends(self, db0_fixture):  # pylint: disable=unused-argument
        """FutureError from warmup still suspends the job (not treated as critical failure)."""
        agent = Agent(role="warmup_future_suspend", _system_prompt="Test", _tools=[])
        future_not_ready = create_future_not_ready()
        warmup_code = "result = future_val"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)
        job.py_env.local_state['future_val'] = future_not_ready

        result = await run_job_step(job)

        assert result is False
        assert job.status == JobStatus.WARMING_UP
        assert job_def.has_errors() is False

    @pytest.mark.asyncio
    async def test_non_warmup_exception_does_not_set_error(self, db0_fixture):  # pylint: disable=unused-argument
        """A non-FutureError exception in STARTED (non-warmup) code does NOT call set_error."""
        agent = Agent(role="started_exc_no_error", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.STARTED)
        # Simulate the job having a last response that raises an exception
        from tests.conftest import create_chat_log_item  # pylint: disable=import-outside-toplevel
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="raise KeyError('oops')"))

        mock_response = MagicMock()
        mock_response.text = "exit('done')"
        mock_response.session_id = None
        mock_response.stats = MagicMock(total_bytes_sent=0, total_bytes_received=0, cost=None)
        mock_response.call_requests = None

        mock_api = MagicMock()
        mock_api.process_request = AsyncMock(return_value=mock_response)

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            mock_llm_api_cls.get.return_value = mock_api
            await run_job_step(job)

        assert job_def.has_errors() is False

    @pytest.mark.asyncio
    async def test_warmup_second_block_exception_sets_job_done(self, db0_fixture):  # pylint: disable=unused-argument
        """Exception in the second warmup block is also treated as critical failure."""
        agent = Agent(role="warmup_blk2_exc", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None,
                         warmup_code=["x = 1", "raise ValueError('second block fails')"])
        job = Job(job_def=job_def, model_family="test", model="test-model",
                  job_status=JobStatus.READY)

        # First block executes successfully, advances to second block
        result1 = await run_job_step(job)
        assert result1 is False
        assert job.status == JobStatus.WARMING_UP

        # Second block raises exception
        result2 = await run_job_step(job)
        assert result2 is True
        assert job.status == JobStatus.DONE
        assert job_def.has_errors() is True


class TestRunJobStepEmptyLLMSubmission:
    """Tests that run_job_step prints error when LLM submits empty code."""

    @pytest.mark.asyncio
    async def test_empty_string_response_prints_error(self, db0_fixture):  # pylint: disable=unused-argument
        """LLM responding with empty string prints error to console."""
        agent = Agent(role="empty_resp", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.STARTED,
        )
        # Simulate LLM having returned empty code
        from tests.conftest import create_chat_log_item
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=""))

        mock_response = LLM_Response(
            text="exit('done')", session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )
        mock_api = MagicMock()
        mock_api.process_request = AsyncMock(return_value=mock_response)

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            mock_llm_api_cls.get.return_value = mock_api
            await run_job_step(job)

        console_text = "\n".join(job.py_env.console) if job.py_env.console else ""
        assert "Error: no code submitted." in console_text

    @pytest.mark.asyncio
    async def test_comment_only_response_prints_error(self, db0_fixture):  # pylint: disable=unused-argument
        """LLM responding with only comments prints error to console."""
        agent = Agent(role="comment_resp", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.STARTED,
        )
        from tests.conftest import create_chat_log_item
        comment_code = "# just a comment\n# another comment"
        job.chat_log.append(create_chat_log_item(
            console_pos=0, llm_resp=comment_code))

        mock_response = LLM_Response(
            text="exit('done')", session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )
        mock_api = MagicMock()
        mock_api.process_request = AsyncMock(return_value=mock_response)

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            mock_llm_api_cls.get.return_value = mock_api
            await run_job_step(job)

        console_text = "\n".join(job.py_env.console) if job.py_env.console else ""
        assert "Error: no code submitted." in console_text

    @pytest.mark.asyncio
    async def test_block_comment_only_response_prints_error(self, db0_fixture):  # pylint: disable=unused-argument
        """LLM responding with only block comments (docstrings) prints error to console."""
        agent = Agent(role="block_comment_resp", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.STARTED,
        )
        from tests.conftest import create_chat_log_item
        block_comment = '"""This is a block comment"""'
        job.chat_log.append(create_chat_log_item(
            console_pos=0, llm_resp=block_comment))

        mock_response = LLM_Response(
            text="exit('done')", session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )
        mock_api = MagicMock()
        mock_api.process_request = AsyncMock(return_value=mock_response)

        mock_harness = MagicMock()
        mock_harness.check_before_step.return_value = None
        mock_harness.check_after_step.return_value = None

        with patch("statek.executors.utils.LLM_API") as mock_llm_api_cls, \
             patch("statek.executors.utils.get_llm_harness", return_value=mock_harness):
            mock_llm_api_cls.get.return_value = mock_api
            await run_job_step(job)

        console_text = "\n".join(job.py_env.console) if job.py_env.console else ""
        assert "Error: no code submitted." in console_text

    @pytest.mark.asyncio
    async def test_codeblock_with_tool_calls_no_error(self, db0_fixture):  # pylint: disable=unused-argument
        """CodeBlock with tool calls but no code should NOT print error."""
        call_spec = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_blocks = [CodeBlock(code=None, tool_calls=[call_spec]), 'exit("ok")']
        agent = Agent(role="tool_no_error", _system_prompt="Test", _tools=[])
        agent.context["my_tool"] = lambda: "tool_result"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_blocks)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.READY,
        )

        await run_job_step(job)

        console_text = "\n".join(job.py_env.console) if job.py_env.console else ""
        assert "Error: no code submitted." not in console_text

    @pytest.mark.asyncio
    async def test_valid_code_no_error(self, db0_fixture):  # pylint: disable=unused-argument
        """Normal code response should NOT print error."""
        agent = Agent(role="valid_code", _system_prompt="Test", _tools=[])
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.STARTED,
        )
        from tests.conftest import create_chat_log_item
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp='exit("done")'))

        await run_job_step(job)

        console_text = "\n".join(job.py_env.console) if job.py_env.console else ""
        assert "Error: no code submitted." not in console_text


class TestRunJobStepEmptyCodeBlock:
    """Tests that run_job_step handles CodeBlock with None/empty code correctly."""

    @pytest.mark.asyncio
    async def test_code_none_does_not_produce_type_error(self, db0_fixture):  # pylint: disable=unused-argument
        """CodeBlock with code=None should not generate a TypeError in the console."""
        call_spec = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        # Two-block warmup: first block is tool-calls-only (code=None), second exits
        warmup_blocks = [CodeBlock(code=None, tool_calls=[call_spec]), 'exit("ok")']
        agent = Agent(role="role_empty_code", _system_prompt="Test", _tools=[])
        agent.context["my_tool"] = lambda: "tool_result"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_blocks)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.READY,
        )

        result = await run_job_step(job)

        assert result is False
        console_text = "\n".join(job.py_env.console) if job.py_env.console else ""
        assert "TypeError" not in console_text

    @pytest.mark.asyncio
    async def test_tool_result_stored_when_code_is_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool results are stored in tool_log even when CodeBlock.code is None."""
        call_spec = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup_blocks = [CodeBlock(code=None, tool_calls=[call_spec]), 'exit("ok")']
        agent = Agent(role="role_empty_tool", _system_prompt="Test", _tools=[])
        agent.context["my_tool"] = lambda: "my_result"
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_blocks)
        job = Job(
            job_def=job_def, model_family="test", model="test-model",
            job_status=JobStatus.READY,
        )

        await run_job_step(job)

        assert job.py_env.tool_log is not None
        assert 0 in job.py_env.tool_log
        assert "'my_result'" in job.py_env.tool_log[0]
