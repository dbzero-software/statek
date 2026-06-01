"""Tests for exec_tool function."""

from typing import Any, Callable
from unittest.mock import patch
import pytest

from statek.executors.utils import exec_tool
from statek.utils import CallSpec, get_current_job
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent
from statek.prompt_config import make_system_prompt
from statek.system import tool, docstr as docs_tool, brief as brief_tool
from statek.future import FutureResult
from tests.test_exec_step import (
    create_future_ready, create_future_not_ready, MemoObject,
    _check_condition_true,
)


# Module-level @tool function (db0 does not allow nested/decorated functions as members)
@tool
def _module_agent_tool(value: str, **kwargs):  # pylint: disable=unused-argument
    """A simple agent tool used in tests."""
    return f"tool: {value}"


# Tools for string-name resolution tests (union type annotation like docs/brief)
@tool
def _resolver_tool(what: type | Callable | Any, **kwargs):  # pylint: disable=unused-argument
    """Accepts a callable or type; returns its name if resolved, else the raw string."""
    if callable(what):
        return f"resolved: {what.__name__}"
    return f"string: {what}"


@tool
def _lookup_target(**kwargs):  # pylint: disable=unused-argument
    """Target tool used as a name-resolution lookup target."""
    return "target"


# Async @tool functions — wrapped by @tool via nest_asyncio (NOT via exec_tool's iscoroutine check)
@tool
async def _async_agent_tool(value: str, **kwargs):  # pylint: disable=unused-argument
    """Async agent tool that returns a value."""
    return f"async_tool: {value}"


@tool
async def _async_agent_tool_error(**kwargs):  # pylint: disable=unused-argument
    """Async agent tool that raises an exception."""
    raise ValueError("async agent tool error")


async def _async_fetch_result_from_deps(future_result):
    """Async complement that returns the value from deps."""
    return future_result.deps.value


class _LLMReprToolObject:  # pylint: disable=too-few-public-methods
    """Simple object with custom LLM representation for exec_tool tests."""

    def __init__(self, name: str):
        self.name = name

    def __llm_repr__(self):
        return f"custom:{self.name}"


def _call_spec(func_name, args=None, kwargs=None):
    return CallSpec(id="TEST-001", func_name=func_name, args=args or [], kwargs=kwargs or {})


def _make_job(role, tools=None, context_extras=None):
    """Create a job with the given agent tools and context additions."""
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
        job_def=job_def,
        model_family="test",
        model="test-model",
        job_status=JobStatus.READY,  # pylint: disable=no-member
    )


class TestExecTool:
    """Tests for exec_tool."""

    @pytest.mark.asyncio
    async def test_return_value_appended_with_llm_formatting(self, db0_fixture):  # pylint: disable=unused-argument
        """Non-None return values are serialized with format_default_llm_repr."""
        def add(a, b):
            return a + b

        job = _make_job("role_return", context_extras={"add": add})
        result, _ = await exec_tool(_call_spec("add", kwargs={"a": 3, "b": 4}), job)

        assert "7" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_return_value_list_uses_llm_repr_for_items(self, db0_fixture):  # pylint: disable=unused-argument
        """List return values format nested objects with __llm_repr__."""
        def list_staff():
            return [_LLMReprToolObject("anna"), _LLMReprToolObject("jan")]

        job = _make_job("role_return_list", context_extras={"list_staff": list_staff})
        result, _ = await exec_tool(_call_spec("list_staff"), job)

        assert result == "[custom:anna,custom:jan]"
        assert "object at" not in result

    @pytest.mark.asyncio
    async def test_none_return_not_added_to_output(self, db0_fixture):  # pylint: disable=unused-argument
        """None return values are silently ignored."""
        def noop():
            return None

        job = _make_job("role_none_ret", context_extras={"noop": noop})
        result, _ = await exec_tool(_call_spec("noop"), job)

        assert result == ""

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_error(self, db0_fixture):  # pylint: disable=unused-argument
        """Unknown func_name yields a NameError description in the output."""
        job = _make_job("role_notfound")
        result, _ = await exec_tool(_call_spec("nonexistent_func"), job)

        assert "NameError" in result
        assert "nonexistent_func" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_exception_not_reraised(self, db0_fixture):  # pylint: disable=unused-argument
        """Exceptions are printed to the private console, not re-raised."""
        def failing():
            raise ValueError("something went wrong")

        job = _make_job("role_exc", context_extras={"failing": failing})
        result, _ = await exec_tool(_call_spec("failing"), job)

        assert "ValueError" in result
        assert "something went wrong" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_no_output_returns_empty_string(self, db0_fixture):  # pylint: disable=unused-argument
        """No return value and no exception → empty string."""
        def silent():
            pass

        job = _make_job("role_silent", context_extras={"silent": silent})
        result, _ = await exec_tool(_call_spec("silent"), job)

        assert result == ""

    @pytest.mark.asyncio
    async def test_positional_args_passed(self, db0_fixture):  # pylint: disable=unused-argument
        """Positional args from CallSpec are forwarded correctly."""
        def multiply(x, y):
            return x * y

        job = _make_job("role_args", context_extras={"multiply": multiply})
        result, _ = await exec_tool(_call_spec("multiply", args=[3, 5]), job)

        assert "15" in result

    @pytest.mark.asyncio
    async def test_kwargs_passed(self, db0_fixture):  # pylint: disable=unused-argument
        """Keyword args from CallSpec are forwarded correctly."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        job = _make_job("role_kwargs", context_extras={"greet": greet})
        result, _ = await exec_tool(
            _call_spec("greet", kwargs={"name": "World", "greeting": "Hi"}), job
        )

        assert 'Hi, World!' in result

    @pytest.mark.asyncio
    async def test_tool_found_in_agent_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool callable stored in agent._tools is found and invoked."""
        job = _make_job("role_agent_tools", tools=[_module_agent_tool])
        result, _ = await exec_tool(
            _call_spec("_module_agent_tool", kwargs={"value": "ok"}), job
        )

        assert 'tool: ok' in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_job_console_not_touched(self, db0_fixture):  # pylint: disable=unused-argument
        """exec_tool never writes to the job's console."""
        def compute(x):
            return x ** 2

        job = _make_job("role_no_console", context_extras={"compute": compute})
        await exec_tool(_call_spec("compute", args=[7]), job)

        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_existing_job_console_untouched(self, db0_fixture):  # pylint: disable=unused-argument
        """Pre-existing job console items are not modified by exec_tool."""
        def compute(x):
            return x * 2

        job = _make_job("role_existing_console", context_extras={"compute": compute})
        job.console_append("existing line")

        await exec_tool(_call_spec("compute", args=[5]), job)

        assert len(job.py_env.console) == 1
        assert job.py_env.console[0] == "existing line"

    @pytest.mark.asyncio
    async def test_async_tool_return_value_captured(self, db0_fixture):  # pylint: disable=unused-argument
        """Async callables are awaited and their return value is captured."""
        async def async_double(x):
            return x * 2

        job = _make_job("role_async", context_extras={"async_double": async_double})
        result, _ = await exec_tool(_call_spec("async_double", args=[21]), job)

        assert "42" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_async_tool_exception_captured(self, db0_fixture):  # pylint: disable=unused-argument
        """Exceptions from async callables are caught and returned."""
        async def async_fail():
            raise RuntimeError("async error")

        job = _make_job("role_async_exc", context_extras={"async_fail": async_fail})
        result, _ = await exec_tool(_call_spec("async_fail"), job)

        assert "RuntimeError" in result
        assert "async error" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_string_tool_name_resolved_to_callable(self, db0_fixture):  # pylint: disable=unused-argument
        """String tool name resolved to callable when annotation is a non-str union type.

        Reproduces the bug where docstr(what='get_user_calendar') returns str's docstring
        because _bind_by_name skips union-type annotations and docstr falls back to
        type(what)=str.
        """
        job = _make_job("role_resolve", tools=[_resolver_tool, _lookup_target])
        result, _ = await exec_tool(
            _call_spec("_resolver_tool", kwargs={"what": "_lookup_target"}), job
        )
        assert "resolved:" in result
        assert "_lookup_target" in result

    @pytest.mark.asyncio
    async def test_async_agent_tool_return_value_captured(self, db0_fixture):  # pylint: disable=unused-argument
        """Async @tool in agent._tools is resolved and its return value captured.

        The @tool decorator wraps async functions via nest_asyncio/run_until_complete,
        so the result is already resolved when exec_tool receives it — the
        asyncio.iscoroutine branch is NOT exercised here.
        """
        job = _make_job("role_async_agent_tool", tools=[_async_agent_tool])
        result, _ = await exec_tool(
            _call_spec("_async_agent_tool", kwargs={"value": "ok"}), job
        )

        assert 'async_tool: ok' in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_async_agent_tool_exception_captured(self, db0_fixture):  # pylint: disable=unused-argument
        """Exception from async @tool in agent._tools is caught and returned as a string."""
        job = _make_job("role_async_agent_tool_err", tools=[_async_agent_tool_error])
        result, _ = await exec_tool(_call_spec("_async_agent_tool_error"), job)

        assert "ValueError" in result
        assert "async agent tool error" in result
        assert job.py_env.console is None


# Module-level @tool for _STATEK_CTX tests (must be at module level due to db0 constraints)
@tool
def _ctx_checker_tool(**kwargs):
    """Tool that checks if _STATEK_CTX has an 'agent' key."""
    ctx = kwargs.get('_local_context', {}).get('_STATEK_CTX', {})
    return 'agent' in ctx


@tool
def _current_job_checker_tool(**kwargs):  # pylint: disable=unused-argument
    """Tool that checks whether framework helpers can resolve the current job."""
    return get_current_job() is not None


class TestExecToolStatekCtx:  # pylint: disable=too-few-public-methods
    """Tests for _STATEK_CTX injection in exec_tool."""

    @pytest.mark.asyncio
    async def test_statek_ctx_agent_available_in_exec_tool(self, db0_fixture):  # pylint: disable=unused-argument
        """_STATEK_CTX with 'agent' key is available to tools via _local_context."""
        agent_obj = Agent(
            role="ctx_tool_test",
            _system_prompt=make_system_prompt("Test"),
            _metadata={"MODEL": "test-model"},
            _tools=[_ctx_checker_tool],
        )
        job_def = JobDef(agent=agent_obj, job_params=None, warmup_code=None)
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,  # pylint: disable=no-member
        )

        result, _ = await exec_tool(_call_spec("_ctx_checker_tool"), job)

        assert result == "True"

    @pytest.mark.asyncio
    async def test_current_job_available_in_exec_tool(self, db0_fixture):  # pylint: disable=unused-argument
        """get_current_job resolves through the active Statek context inside tools."""
        agent_obj = Agent(
            role="ctx_current_job_tool_test",
            _system_prompt=make_system_prompt("Test"),
            _metadata={"MODEL": "test-model"},
            _tools=[_current_job_checker_tool],
        )
        job_def = JobDef(agent=agent_obj, job_params=None, warmup_code=None)
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,  # pylint: disable=no-member
        )

        result, _ = await exec_tool(_call_spec("_current_job_checker_tool"), job)

        assert result == "True"


class _SimpleDocClass:  # pylint: disable=too-few-public-methods
    """A simple class for docstring testing."""


class TestDocsAgentIntegration:
    """Tests that docstr and brief pass the current agent name to format_docstring."""

    @pytest.mark.asyncio
    async def test_docs_passes_agent_type_name_to_format_docstring(self, db0_fixture):  # pylint: disable=unused-argument
        """docstr tool passes current agent's type name as agent= to format_docstring."""
        job = _make_job("doc_test_agent", tools=[docs_tool],
                        context_extras={"_SimpleDocClass": _SimpleDocClass})

        with patch('statek.system.format_docstring', return_value="") as mock_fmt:
            await exec_tool(_call_spec("docstr", kwargs={"what": "_SimpleDocClass"}), job)

        assert mock_fmt.called
        _, call_kwargs = mock_fmt.call_args
        assert call_kwargs.get('agent') == 'Agent'

    @pytest.mark.asyncio
    async def test_brief_passes_agent_type_name_to_format_docstring(self, db0_fixture):  # pylint: disable=unused-argument
        """brief tool passes current agent's type name as agent= to format_docstring."""
        job = _make_job("doc_test_agent", tools=[brief_tool],
                        context_extras={"_SimpleDocClass": _SimpleDocClass})

        with patch('statek.system.format_docstring', return_value="") as mock_fmt:
            await exec_tool(_call_spec("brief", kwargs={"what": "_SimpleDocClass"}), job)

        assert mock_fmt.called
        _, call_kwargs = mock_fmt.call_args
        assert call_kwargs.get('agent') == 'Agent'

    @pytest.mark.asyncio
    async def test_docs_output_does_not_start_with_extra_quote(self, db0_fixture):  # pylint: disable=unused-argument
        """docstr tool output must not be wrapped in extra double quotes."""
        job = _make_job("doc_format_test", tools=[docs_tool],
                        context_extras={"_SimpleDocClass": _SimpleDocClass})

        result, _ = await exec_tool(_call_spec("docstr", kwargs={"what": "_SimpleDocClass"}), job)

        assert result.startswith("class "), (
            f"Output should start with 'class ', got: {result[:60]!r}"
        )

    @pytest.mark.asyncio
    async def test_docs_output_does_not_end_with_extra_quote(self, db0_fixture):  # pylint: disable=unused-argument
        """docstr tool output closing triple-quote must not gain an extra trailing quote."""
        job = _make_job("doc_format_test2", tools=[docs_tool],
                        context_extras={"_SimpleDocClass": _SimpleDocClass})

        result, _ = await exec_tool(_call_spec("docstr", kwargs={"what": "_SimpleDocClass"}), job)

        assert not result.endswith('""""'), (
            f"Output should end with '\"\"\"', not '\"\"\"\"', got tail: {result[-20:]!r}"
        )


class TestExecToolFutureResult:
    """Tests for FutureResult/FutureElement unwrapping in exec_tool."""

    @pytest.mark.asyncio
    async def test_sync_future_result_unwrapped(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool returning a ready FutureResult has .value unwrapped."""
        def tool_returning_future():
            return create_future_ready(99)

        job = _make_job("role_future_sync", context_extras={
            "tool_returning_future": tool_returning_future})
        result, error = await exec_tool(_call_spec("tool_returning_future"), job)

        assert error is None
        assert "99" in result

    @pytest.mark.asyncio
    async def test_async_complement_future_result_unwrapped(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool returning a FutureResult with async complement has .value awaited."""
        def tool_returning_async_future():
            future = FutureResult(deps=MemoObject(value=77), state_num=0)
            future.set_complement_functions(complement=_async_fetch_result_from_deps,
                                            condition=_check_condition_true)
            return future

        job = _make_job("role_future_async", context_extras={
            "tool_returning_async_future": tool_returning_async_future})
        result, error = await exec_tool(_call_spec("tool_returning_async_future"), job)

        assert error is None
        assert "77" in result

    @pytest.mark.asyncio
    async def test_not_ready_future_result_raises_error(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool returning a not-ready FutureResult produces a FutureError in output."""
        def tool_returning_not_ready():
            return create_future_not_ready()

        job = _make_job("role_future_err", context_extras={
            "tool_returning_not_ready": tool_returning_not_ready})
        result, error = await exec_tool(_call_spec("tool_returning_not_ready"), job)

        assert error is not None
        assert "FutureError" in result
