"""Tests for ToolError storage in tool_log and integration."""
# pylint: disable=no-member,redefined-outer-name

import pytest

from statek.executors.chat_log_item import ChatLogItem, LLM_LogItem, ToolError, WarmupLogItem
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.utils import exec_tool, run_job_step
from statek.agents.agent import Agent
from statek.prompt_config import make_system_prompt
from statek.utils import CallSpec, CodeBlock


class TestToolErrorInToolLog:
    """ToolError entries are stored positionally in tool_log alongside str results."""

    def test_error_stored_as_single_entry(self, db0_fixture):  # pylint: disable=unused-argument
        item = ChatLogItem(console_pos=0)
        item.push_tool_result(ToolError(err_message="ValueError: boom"))
        assert isinstance(item.tool_log, ToolError)
        assert item.tool_log.err_message == "ValueError: boom"

    def test_error_and_success_stored_positionally(self, db0_fixture):  # pylint: disable=unused-argument
        item = ChatLogItem(console_pos=0)
        item.push_tool_result("ok result")
        item.push_tool_result(ToolError(err_message="err"))
        result = item.get_tool_result(0)
        assert isinstance(result, str)
        assert result == "ok result"
        result = item.get_tool_result(1)
        assert isinstance(result, ToolError)
        assert result.err_message == "err"

    def test_multiple_errors(self, db0_fixture):  # pylint: disable=unused-argument
        item = ChatLogItem(console_pos=0)
        item.push_tool_result(ToolError(err_message="e1"))
        item.push_tool_result(ToolError(err_message="e2"))
        assert isinstance(item.get_tool_result(0), ToolError)
        assert isinstance(item.get_tool_result(1), ToolError)

    def test_subclasses_support_tool_error(self, db0_fixture):  # pylint: disable=unused-argument
        llm = LLM_LogItem(console_pos=0, llm_resp="x = 1")
        llm.push_tool_result(ToolError(err_message="RuntimeError: oops"))
        assert isinstance(llm.get_tool_result(0), ToolError)

        warmup = WarmupLogItem(console_pos=1, warmup_block_num=0)
        warmup.push_tool_result(ToolError(err_message="err"))
        assert isinstance(warmup.get_tool_result(0), ToolError)


def _make_job(role, tools=None, context_extras=None):
    agent = Agent(
        role=role,
        _system_prompt=make_system_prompt("Test"),
        _metadata={"MODEL": "test-model"},
        _tools=tools or [],
    )
    if context_extras:
        agent.context.update(context_extras)
    job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
    return Job(
        job_def=job_def, model_family="test", model="test-model",
        job_status=JobStatus.READY,
    )


def _call_spec(func_name, args=None, kwargs=None):
    return CallSpec(id="TEST-001", func_name=func_name, args=args or [], kwargs=kwargs or {})


class TestExecToolReturnsError:
    """exec_tool returns (result, error_message) tuple; error_message is None on success."""

    @pytest.mark.asyncio
    async def test_success_returns_none_error(self, db0_fixture):  # pylint: disable=unused-argument
        def ok():
            return "fine"

        job = _make_job("et_ok", context_extras={"ok": ok})
        result, error = await exec_tool(_call_spec("ok"), job)
        assert "fine" in result
        assert error is None

    @pytest.mark.asyncio
    async def test_exception_returns_error_message(self, db0_fixture):  # pylint: disable=unused-argument
        def failing():
            raise ValueError("boom")

        job = _make_job("et_exc", context_extras={"failing": failing})
        result, error = await exec_tool(_call_spec("failing"), job)
        assert "ValueError" in result
        assert error is not None
        assert "ValueError" in error
        assert "boom" in error

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_error(self, db0_fixture):  # pylint: disable=unused-argument
        job = _make_job("et_nf")
        result, error = await exec_tool(_call_spec("nonexistent"), job)
        assert "NameError" in result
        assert error is not None
        assert "NameError" in error

    @pytest.mark.asyncio
    async def test_async_exception_returns_error(self, db0_fixture):  # pylint: disable=unused-argument
        async def async_fail():
            raise RuntimeError("async boom")

        job = _make_job("et_async_exc", context_extras={"async_fail": async_fail})
        result, error = await exec_tool(_call_spec("async_fail"), job)
        assert "RuntimeError" in result
        assert error is not None


class TestRunJobStepRecordsToolErrors:
    """run_job_step records tool errors as ToolError in tool_log."""

    @staticmethod
    def _make_warmup_job(role, warmup_code, context_extras=None):
        agent = Agent(
            role=role,
            _system_prompt=make_system_prompt("Test"),
            _metadata={"MODEL": "test-model"},
            _tools=[],
        )
        if context_extras:
            agent.context.update(context_extras)
        job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
        return Job(job_def=job_def, model_family="test", model="test-model",
                   job_status=JobStatus.READY)

    @pytest.mark.asyncio
    async def test_failing_tool_stored_as_tool_error(self, db0_fixture):  # pylint: disable=unused-argument
        def failing_tool():
            raise RuntimeError("warmup tool error")

        cs = CallSpec(id="S-001", func_name="failing_tool", args=[], kwargs={})
        job = self._make_warmup_job(
            "te_warmup",
            warmup_code=[CodeBlock(code=None, tool_calls=[cs]), 'exit("ok")'],
            context_extras={"failing_tool": failing_tool},
        )

        await run_job_step(job)

        warmup_items = [item for item in job.chat_log if isinstance(item, WarmupLogItem)]
        assert len(warmup_items) >= 1
        entry = warmup_items[0].get_tool_result(0)
        assert isinstance(entry, ToolError)
        assert "RuntimeError" in entry.err_message

    @pytest.mark.asyncio
    async def test_successful_tool_stored_as_str(self, db0_fixture):  # pylint: disable=unused-argument
        cs = CallSpec(id="S-001", func_name="ok_tool", args=[], kwargs={})
        job = self._make_warmup_job(
            "te_warmup_ok",
            warmup_code=[CodeBlock(code=None, tool_calls=[cs]), 'exit("ok")'],
            context_extras={"ok_tool": lambda: "ok"},
        )

        await run_job_step(job)

        warmup_items = [item for item in job.chat_log if isinstance(item, WarmupLogItem)]
        assert len(warmup_items) >= 1
        entry = warmup_items[0].get_tool_result(0)
        assert isinstance(entry, str)

    @pytest.mark.asyncio
    async def test_mixed_tools_positional(self, db0_fixture):  # pylint: disable=unused-argument
        """With 2 tools (one OK, one failing), entries are positional in tool_log."""
        def failing_tool():
            raise ValueError("fail")

        cs_ok = CallSpec(id="S-001", func_name="ok_tool", args=[], kwargs={})
        cs_fail = CallSpec(id="S-002", func_name="failing_tool", args=[], kwargs={})
        job = self._make_warmup_job(
            "te_mixed",
            warmup_code=[CodeBlock(code=None, tool_calls=[cs_ok, cs_fail]), 'exit("ok")'],
            context_extras={"ok_tool": lambda: "ok", "failing_tool": failing_tool},
        )

        await run_job_step(job)

        warmup_items = [item for item in job.chat_log if isinstance(item, WarmupLogItem)]
        assert len(warmup_items) >= 1
        ok_entry = warmup_items[0].get_tool_result(0)
        assert isinstance(ok_entry, str)
        fail_entry = warmup_items[0].get_tool_result(1)
        assert isinstance(fail_entry, ToolError)
        assert "ValueError" in fail_entry.err_message
