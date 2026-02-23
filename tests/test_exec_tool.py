"""Tests for exec_tool function."""

import pytest

from statek.executors.utils import exec_tool
from statek.utils import CallSpec
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent
from statek.system import tool


# Module-level @tool function (db0 does not allow nested/decorated functions as members)
@tool
def _module_agent_tool(value: str, **kwargs):  # pylint: disable=unused-argument
    """A simple agent tool used in tests."""
    return f"tool: {value}"


def _call_spec(func_name, args=None, kwargs=None):
    return CallSpec(id="TEST-001", func_name=func_name, args=args or [], kwargs=kwargs or {})


def _make_job(role, tools=None, context_extras=None):
    """Create a job with the given agent tools and context additions."""
    agent = Agent(
        role=role,
        _system_prompt="Test",
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
    async def test_return_value_appended_as_repr(self, db0_fixture):  # pylint: disable=unused-argument
        """Non-None return value appears in the output as repr()."""
        def add(a, b):
            return a + b

        job = _make_job("role_return", context_extras={"add": add})
        result = await exec_tool(_call_spec("add", kwargs={"a": 3, "b": 4}), job)

        assert "7" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_none_return_not_added_to_output(self, db0_fixture):  # pylint: disable=unused-argument
        """None return values are silently ignored."""
        def noop():
            return None

        job = _make_job("role_none_ret", context_extras={"noop": noop})
        result = await exec_tool(_call_spec("noop"), job)

        assert result == ""

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_error(self, db0_fixture):  # pylint: disable=unused-argument
        """Unknown func_name yields a NameError description in the output."""
        job = _make_job("role_notfound")
        result = await exec_tool(_call_spec("nonexistent_func"), job)

        assert "NameError" in result
        assert "nonexistent_func" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_exception_not_reraised(self, db0_fixture):  # pylint: disable=unused-argument
        """Exceptions are printed to the private console, not re-raised."""
        def failing():
            raise ValueError("something went wrong")

        job = _make_job("role_exc", context_extras={"failing": failing})
        result = await exec_tool(_call_spec("failing"), job)

        assert "ValueError" in result
        assert "something went wrong" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_no_output_returns_empty_string(self, db0_fixture):  # pylint: disable=unused-argument
        """No return value and no exception → empty string."""
        def silent():
            pass

        job = _make_job("role_silent", context_extras={"silent": silent})
        result = await exec_tool(_call_spec("silent"), job)

        assert result == ""

    @pytest.mark.asyncio
    async def test_positional_args_passed(self, db0_fixture):  # pylint: disable=unused-argument
        """Positional args from CallSpec are forwarded correctly."""
        def multiply(x, y):
            return x * y

        job = _make_job("role_args", context_extras={"multiply": multiply})
        result = await exec_tool(_call_spec("multiply", args=[3, 5]), job)

        assert "15" in result

    @pytest.mark.asyncio
    async def test_kwargs_passed(self, db0_fixture):  # pylint: disable=unused-argument
        """Keyword args from CallSpec are forwarded correctly."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        job = _make_job("role_kwargs", context_extras={"greet": greet})
        result = await exec_tool(
            _call_spec("greet", kwargs={"name": "World", "greeting": "Hi"}), job
        )

        assert "'Hi, World!'" in result

    @pytest.mark.asyncio
    async def test_tool_found_in_agent_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Tool callable stored in agent._tools is found and invoked."""
        job = _make_job("role_agent_tools", tools=[_module_agent_tool])
        result = await exec_tool(
            _call_spec("_module_agent_tool", kwargs={"value": "ok"}), job
        )

        assert "'tool: ok'" in result
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
        result = await exec_tool(_call_spec("async_double", args=[21]), job)

        assert "42" in result
        assert job.py_env.console is None

    @pytest.mark.asyncio
    async def test_async_tool_exception_captured(self, db0_fixture):  # pylint: disable=unused-argument
        """Exceptions from async callables are caught and returned."""
        async def async_fail():
            raise RuntimeError("async error")

        job = _make_job("role_async_exc", context_extras={"async_fail": async_fail})
        result = await exec_tool(_call_spec("async_fail"), job)

        assert "RuntimeError" in result
        assert "async error" in result
        assert job.py_env.console is None
