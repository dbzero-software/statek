"""Tests for Statek's default RestrictedPython sandbox."""

from pathlib import Path

import pytest

import statek
import statek.python_sandbox as sandbox_module
from statek.agents.agent import Agent
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.utils import exec_cli_step, exec_step, exec_tool
from statek.python_sandbox import (
    SandboxPolicy,
    SandboxViolation,
    validate_selltime_agent_config,
)
from statek.prompt_config import make_system_prompt
from statek.settings import StatekSettings, get_statek_settings, set_statek_settings
from statek.system import tool
from statek.utils import CallSpec


@pytest.fixture(autouse=True)
def _reset_statek_settings(monkeypatch):
    monkeypatch.delenv("STATEK_PYTHON_SANDBOX_MODE", raising=False)
    set_statek_settings(None)
    get_statek_settings.cache_clear()
    yield
    set_statek_settings(None)
    get_statek_settings.cache_clear()


def _call_spec(func_name, args=None, kwargs=None):
    return CallSpec(id="SANDBOX-001", func_name=func_name, args=args or [], kwargs=kwargs or {})


@tool
def visible_tool(value: str, **kwargs):  # pylint: disable=unused-argument
    return f"visible: {value}"


@tool(hidden=True)
def hidden_tool(**kwargs):  # pylint: disable=unused-argument
    return "hidden"


@tool
def _internal_tool(**kwargs):  # pylint: disable=unused-argument
    return "internal"


def _job_with_tools(tools=None):
    agent = Agent(
        role="sandbox-agent",
        _system_prompt=make_system_prompt("Test"),
        _metadata={"MODEL": "test-model"},
        _tools=tools or [],
    )
    return Job(
        job_def=JobDef(agent=agent, job_params=None, warmup_code=None),
        model_family="test",
        model="test-model",
        job_status=JobStatus.READY,  # pylint: disable=no-member
    )


def test_statek_init_enables_restricted_mode_by_default():
    statek.init(StatekSettings(prompt_defs={}))

    assert get_statek_settings().python_sandbox_mode == "restricted"


def test_statek_init_restricted_false_disables_mode():
    statek.init(StatekSettings(prompt_defs={}), restricted=False)

    assert get_statek_settings().python_sandbox_mode == "off"


def test_env_off_disables_mode(monkeypatch):
    monkeypatch.setenv("STATEK_PYTHON_SANDBOX_MODE", "off")

    statek.init()

    assert get_statek_settings().python_sandbox_mode == "off"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "__import__('os')",
        "open('/tmp/x')",
        "eval('1 + 1')",
        "exec('x = 1')",
        "compile('x = 1', '<x>', 'exec')",
        "globals()",
        "locals()",
        "type(1)",
        "getattr(1, 'real')",
        "(1).__class__",
    ],
)
async def test_restricted_mode_blocks_host_escapes(job_factory, source):
    job = job_factory()

    with pytest.raises(SandboxViolation):
        await exec_step(source, job)


@pytest.mark.asyncio
async def test_allowed_imports_work(job_factory):
    job = job_factory()

    await exec_step(
        "from datetime import date, timedelta\n"
        "import calendar as cal\n"
        "target = date(2026, 5, 1) + timedelta(days=1)\n"
        "days = cal.monthrange(target.year, target.month)[1]\n"
        "print(days)",
        job,
    )

    assert job.py_env.console == ["31"]


@pytest.mark.asyncio
async def test_exec_cli_step_uses_restricted_mode(job_factory):
    job = job_factory()
    outputs = []

    with pytest.raises(SandboxViolation):
        await exec_cli_step("__import__('os')", job, outputs.append)

    assert any("__import__" in output for output in outputs)


@pytest.mark.asyncio
async def test_allowed_tool_call_executes(db0_fixture):  # pylint: disable=unused-argument
    job = _job_with_tools([visible_tool])

    result, error = await exec_tool(_call_spec("visible_tool", kwargs={"value": "ok"}), job)

    assert error is None
    assert result == "visible: ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["hidden_tool", "missing_tool"])
async def test_hidden_internal_and_unknown_tools_fail(db0_fixture, tool_name):  # pylint: disable=unused-argument
    job = _job_with_tools([visible_tool, hidden_tool, _internal_tool])

    result, error = await exec_tool(_call_spec(tool_name), job)

    assert error is not None
    assert tool_name in result


@pytest.mark.asyncio
async def test_direct_hidden_tool_call_in_code_fails(db0_fixture):  # pylint: disable=unused-argument
    job = _job_with_tools([hidden_tool])

    with pytest.raises(SandboxViolation, match="hidden_tool"):
        await exec_step("hidden_tool()", job)


def test_selltime_warmups_and_examples_validate():
    root = Path("/src/selltime/agent_config")
    if not root.exists():
        pytest.skip("SellTime agent config is not available in this workspace")

    errors = validate_selltime_agent_config(root, SandboxPolicy())

    assert not errors


def test_validate_source_cache_reuses_sanitized_ast(monkeypatch):
    policy = SandboxPolicy()
    calls = []
    original_compile_restricted = sandbox_module.compile_restricted

    def tracking_compile_restricted(*args, **kwargs):
        calls.append(args[0])
        return original_compile_restricted(*args, **kwargs)

    monkeypatch.setattr(
        "statek.python_sandbox.compile_restricted",
        tracking_compile_restricted,
    )

    first = policy.validate_source("x = 1\nx")
    second = policy.validate_source("x = 1\nx")

    assert len(calls) == 1
    assert first is not second
    assert first.body[0] is not second.body[0]
