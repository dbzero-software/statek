"""Tests for Statek's default RestrictedPython sandbox."""

from pathlib import Path

import pytest
import dbzero as db0

import statek
import statek.python_sandbox as sandbox_module
from statek.dbzero_restricted import DbzeroRestrictedModeError
from statek.agents.agent import Agent
from statek.pyenv import PyEnv
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.utils import exec_cli_step, exec_step, exec_tool
from statek.python_sandbox import (
    DEFAULT_ALLOWED_IMPORTS,
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


def test_statek_init_restricted_mode_requires_restricted_dbzero(monkeypatch):
    monkeypatch.setattr(
        "statek.dbzero_restricted.db0.get_config",
        lambda: {"restricted": False},
    )

    with pytest.raises(DbzeroRestrictedModeError, match="restricted=True"):
        statek.init(StatekSettings(prompt_defs={}))


def test_statek_init_restricted_false_does_not_validate_dbzero(monkeypatch):
    def fail_if_called():
        raise AssertionError("dbzero validation should not run")

    monkeypatch.setattr("statek.dbzero_restricted.db0.get_config", fail_if_called)

    statek.init(StatekSettings(prompt_defs={}), restricted=False)

    assert get_statek_settings().python_sandbox_mode == "off"


def test_open_prefix_forces_restricted_prefix(tmp_path):
    settings = StatekSettings(prompt_defs={})
    db0.init(str(tmp_path), restricted=True)
    try:
        if "restricted" not in db0.get_config():  # pylint: disable=no-member
            pytest.skip("installed dbzero does not expose restricted mode")
        statek.init(settings)

        statek.open_prefix("statek-prefix", "rw")
        env = PyEnv()
        env.console = ["ok"]

        assert db0.get_prefix_stats()["restricted"] is True  # pylint: disable=no-member
        assert list(db0.find(PyEnv))[0].console == ["ok"]  # pylint: disable=no-member
    finally:
        db0.close()  # pylint: disable=no-member


def test_open_prefix_rejects_unrestricted_prefix_in_restricted_mode():
    statek.init(StatekSettings(prompt_defs={}))

    with pytest.raises(DbzeroRestrictedModeError, match="restricted=False"):
        statek.open_prefix("statek-prefix", "rw", restricted=False)


def test_dbzero_restricted_mode_allows_statek_memo_properties(tmp_path):
    settings = StatekSettings(prompt_defs={})
    db0.init(str(tmp_path), restricted=True)
    try:
        if "restricted" not in db0.get_config():  # pylint: disable=no-member
            pytest.skip("installed dbzero does not expose restricted mode")
        statek.init(settings)
        statek.open_prefix("statek-prefix", "rw")

        env = PyEnv(local_state={"_PERM_CTX": {"last_example_id": 1}})

        assert env.perm_ctx == {"last_example_id": 1}
    finally:
        db0.close()  # pylint: disable=no-member


def test_env_off_disables_mode(monkeypatch):
    monkeypatch.setenv("STATEK_PYTHON_SANDBOX_MODE", "off")

    statek.init()

    assert get_statek_settings().python_sandbox_mode == "off"


def test_default_allowed_imports_match_supported_stdlib_surface():
    expected = (
        "datetime,calendar,re,math,decimal,fractions,statistics,collections,"
        "itertools,functools,operator,json"
    )

    assert DEFAULT_ALLOWED_IMPORTS == expected
    assert StatekSettings(prompt_defs={}).python_sandbox_allowed_imports == expected
    assert SandboxPolicy().allowed_imports == set(expected.split(","))


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
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "import re\n"
            "matches = re.findall(r'[A-Z]+', 'ab CD ef GH')\n"
            "stdlib_result = [matches, re.sub(r'\\d+', '#', 'a12b34')]",
            [["CD", "GH"], "a#b#"],
        ),
        (
            "import math\n"
            "stdlib_result = [math.sqrt(81), math.factorial(5), round(math.pi, 2)]",
            [9.0, 120, 3.14],
        ),
        (
            "from decimal import Decimal, ROUND_HALF_UP\n"
            "value = Decimal('1.235').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\n"
            "stdlib_result = str(value)\n"
            "del value",
            "1.24",
        ),
        (
            "from fractions import Fraction\n"
            "stdlib_result = str(Fraction(1, 3) + Fraction(1, 6))",
            "1/2",
        ),
        (
            "import statistics\n"
            "stdlib_result = [statistics.mean([1, 2, 3]), statistics.median([1, 7, 2])]",
            [2, 2],
        ),
        (
            "from collections import Counter, deque, defaultdict, OrderedDict\n"
            "counts = Counter('banana')\n"
            "items = deque([1, 2])\n"
            "items.append(3)\n"
            "groups = defaultdict(list)\n"
            "groups['x'].append(4)\n"
            "ordered = OrderedDict([('a', 1), ('b', 2)])\n"
            "stdlib_result = [counts['a'], list(items), groups['x'], list(ordered.items())]\n"
            "del counts\n"
            "del items\n"
            "del groups\n"
            "del ordered",
            [3, [1, 2, 3], [4], [("a", 1), ("b", 2)]],
        ),
        (
            "import itertools\n"
            "pairs = list(itertools.combinations([1, 2, 3], 2))\n"
            "window = list(itertools.islice(itertools.chain([1], [2, 3]), 3))\n"
            "stdlib_result = [pairs, window, list(itertools.pairwise([1, 2, 4]))]",
            [[(1, 2), (1, 3), (2, 3)], [1, 2, 3], [(1, 2), (2, 4)]],
        ),
        (
            "from functools import reduce\n"
            "stdlib_result = reduce(lambda a, b: a + b, [1, 2, 3, 4])",
            10,
        ),
        (
            "import operator\n"
            "stdlib_result = [operator.add(2, 3), operator.mul(4, 5), operator.lt(1, 2)]",
            [5, 20, True],
        ),
        (
            "import json\n"
            "payload = json.loads(json.dumps({'a': [1, 2], 'b': True}))\n"
            "stdlib_result = [payload['a'], payload['b']]",
            [[1, 2], True],
        ),
    ],
)
async def test_restricted_mode_allows_sanitized_stdlib_helpers(job_factory, source, expected):
    job = job_factory()

    await exec_step(source, job)

    assert job.py_env.local_state["stdlib_result"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "import statek",
        "from statek import init",
        "from statek.settings import set_statek_settings",
        "from statek import dbzero_restricted",
        "import dbzero",
        "import os",
        "import sys",
        "import subprocess",
    ],
)
async def test_restricted_mode_blocks_non_sandbox_imports(job_factory, source):
    job = job_factory()

    with pytest.raises(SandboxViolation):
        await exec_step(source, job)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "import calendar as cal\ncal.__dict__",
        "import calendar as cal\ncal.__class__",
        "import calendar as cal\ncal._monthlen",
        "import calendar as cal\ncal.HTMLCalendar",
        "from calendar import __dict__",
        "from calendar import _monthlen",
        "from datetime import __spec__",
        "dir(__import__('calendar'))",
        "getattr(__import__('datetime'), 'date')",
    ],
)
async def test_sandbox_import_wrappers_block_introspection(job_factory, source):
    job = job_factory()

    with pytest.raises(SandboxViolation):
        await exec_step(source, job)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name",
    [
        "re",
        "math",
        "decimal",
        "fractions",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "operator",
        "json",
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "import {module_name} as mod\nmod.__dict__",
        "import {module_name} as mod\nmod.__class__",
        "import {module_name} as mod\nmod._private",
        "import {module_name} as mod\nmod.missing_name",
        "from {module_name} import __spec__",
    ],
)
async def test_new_sandbox_import_wrappers_block_internals(
    job_factory,
    module_name,
    template,
):
    job = job_factory()

    with pytest.raises(SandboxViolation):
        await exec_step(template.format(module_name=module_name), job)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["attrgetter", "methodcaller", "getitem"])
async def test_operator_wrapper_omits_reflection_and_raw_item_helpers(job_factory, name):
    job = job_factory()

    with pytest.raises(SandboxViolation):
        await exec_step(f"from operator import {name}", job)


@pytest.mark.asyncio
async def test_sandbox_import_wrappers_are_not_persisted(job_factory):
    job = job_factory()

    await exec_step(
        "import re\n"
        "import math\n"
        "primitive_result = math.floor(2.9) + len(re.findall('a', 'banana'))",
        job,
    )

    assert job.py_env.local_state["primitive_result"] == 5
    assert "re" not in job.py_env.local_state
    assert "math" not in job.py_env.local_state


@pytest.mark.asyncio
async def test_unrestricted_mode_uses_normal_imports(job_factory):
    set_statek_settings(StatekSettings(prompt_defs={}, python_sandbox_mode="off"))
    job = job_factory()

    await exec_step(
        "import os\n"
        "import statek.settings as statek_settings\n"
        "off_mode_result = [os.path.basename('/tmp/example.txt'), "
        "statek_settings.StatekSettings(prompt_defs={}).python_sandbox_mode]\n"
        "del os\n"
        "del statek_settings",
        job,
    )

    assert job.py_env.local_state["off_mode_result"] == ["example.txt", "restricted"]
    assert "os" not in job.py_env.local_state
    assert "statek_settings" not in job.py_env.local_state


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
