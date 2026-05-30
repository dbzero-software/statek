# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable,protected-access
from typing import Tuple
import asyncio
import pytest
import dbzero as db0
from statek.utils import (
    find_locals,
    get_current_agent,
    get_current_agent_name,
    get_current_job,
    _statek_ctx_scope,
)
from statek.system import inject_context, tool
from statek.executors.utils import exec_step, _smart_call
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent
from statek.prompt_config import make_system_prompt
from statek.future import FutureResult, temporal
from statek.exceptions import FutureError


# --- Helpers for FutureResult-based tests ---

@db0.memo
class _MockFuture(FutureResult):
    """Mock FutureResult for find_locals tests."""

    def __init__(self, result_value, ready: bool):
        super().__init__(deps=None, state_num=1)
        self._result_value = result_value
        self._ready = ready


def _mock_complement(fut: _MockFuture):
    if not fut._ready:
        raise FutureError(fut)
    return fut._result_value


def _mock_condition(fut: _MockFuture):
    return fut._ready


@temporal(_mock_complement, _mock_condition)
def _make_future(result_value, ready: bool = True):
    return _MockFuture(result_value, ready)


# --- Helpers for FutureElement (tuple) tests ---

@db0.memo
class _TupleFuture(FutureResult):
    """Mock FutureResult returning a Tuple[str, int] for find_locals tests."""

    def __init__(self, val, ready: bool):
        super().__init__(deps=None, state_num=1)
        self._val = val
        self._ready = ready


def _tcomp(fut: _TupleFuture) -> Tuple[str, int]:
    if not fut._ready:
        raise FutureError(fut)
    return fut._val


def _tcond(fut: _TupleFuture):
    return fut._ready


@temporal(_tcomp, _tcond)
def _make_tuple_future(val, ready: bool = True):
    return _TupleFuture(val, ready)


# Module-level class used for clean class-name assertions
class _FakeAgentClass:
    pass


class _FakeJob:
    """Minimal job stub exposing py_env.perm_ctx for find_locals tests."""

    class _FakePyEnv:
        def __init__(self, perm_ctx=None):
            self.local_state = {} if perm_ctx is None else {"_PERM_CTX": perm_ctx}

        @property
        def perm_ctx(self):
            return self.local_state.get("_PERM_CTX")

    def __init__(self, perm_ctx=None):
        self.py_env = self._FakePyEnv(perm_ctx)


class _PermCtxRaisesJob:
    """Job stub whose perm_ctx should not be touched for local name hits."""

    class _FakePyEnv:
        @property
        def perm_ctx(self):
            raise AssertionError("perm_ctx should not be accessed")

    def __init__(self):
        self.py_env = self._FakePyEnv()


def _run_with_job(job, func):
    """Run func while Statek context exposes a fake current job."""
    with _statek_ctx_scope({"job": job}):
        return func()


class TestFindLocals:
    """Tests for find_locals function."""

    def test_find_all_locals(self):
        """Test finding all local variables when no filters are specified."""
        x = 10
        y = 'hello'
        z = [1, 2, 3]

        results = list(find_locals())

        # Should find at least x, y, z (may include self from test method)
        assert x in results
        assert y in results
        assert z in results

    def test_find_by_type_int(self):
        """Test finding variables by type (int)."""
        x = 10
        y = 20
        z = 'hello'

        results = list(find_locals(var_type=int))

        assert x in results
        assert y in results
        assert z not in results

    def test_find_by_type_str(self):
        """Test finding variables by type (str)."""
        x = 10
        y = 'hello'
        z = 'world'

        results = list(find_locals(var_type=str))

        assert x not in results
        assert y in results
        assert z in results

    def test_find_by_type_list(self):
        """Test finding variables by type (list)."""
        x = [1, 2, 3]
        y = [4, 5]
        z = 'not a list'

        results = list(find_locals(var_type=list))

        assert x in results
        assert y in results
        assert z not in results

    def test_find_by_name(self):
        """Test finding variables by name."""
        user = {'name': 'Alice'}
        admin = {'name': 'Bob'}

        results = list(find_locals(var_name='user'))

        assert len(results) == 1
        assert results[0] == user

    def test_find_by_name_not_found(self):
        """Test finding variables by name when it doesn't exist."""
        x = 10
        y = 20

        results = list(find_locals(var_name='nonexistent'))

        assert len(results) == 0

    def test_find_by_type_and_name(self):
        """Test finding variables by both type and name."""
        user = 42
        admin = 'Alice'

        results = list(find_locals(var_type=int, var_name='user'))

        assert len(results) == 1
        assert results[0] == 42

    def test_find_by_type_and_name_no_match(self):
        """Test finding variables by type and name when type doesn't match."""
        user = 'Alice'
        admin = 42

        results = list(find_locals(var_type=int, var_name='user'))

        assert len(results) == 0

    def test_find_custom_class(self):
        """Test finding variables by custom class type."""
        class User:
            def __init__(self, name):
                self.name = name

        user1 = User('Alice')
        user2 = User('Bob')
        admin = 'not a user'

        results = list(find_locals(var_type=User))

        assert user1 in results
        assert user2 in results
        assert admin not in results
        assert len(results) == 2

    def test_find_with_none_values(self):
        """Test finding variables that have None value."""
        x = None
        y = 10

        results = list(find_locals(var_type=type(None)))

        assert x in results
        assert y not in results

    def test_find_empty_context(self):
        """Test finding variables when only self exists (in test method)."""
        # Only 'self' and 'results' should exist
        results = list(find_locals(var_name='nonexistent'))

        assert len(results) == 0

    def test_find_with_local_context(self):
        """Test finding variables with __local_context extending the scope."""
        # Create additional context
        context_data = {
            'context_var': 42,
            'message': 'from context'
        }

        def function_with_context(**kwargs):
            _local_context = kwargs.get('_local_context')
            x = 10
            y = 'local'
            # Should find both local vars and context vars
            return list(find_locals(var_type=int))

        # Use inject_context wrapper to inject context_data
        f = inject_context(function_with_context, context_data)
        results = f()

        assert 10 in results  # x from local scope
        assert 42 in results  # context_var from __local_context

    def test_find_by_name_scans_perm_ctx_by_default(self):
        """find_locals finds named values in the current job's perm_ctx by default."""
        job = _FakeJob({"message": "from perm"})

        def exercise():
            return list(find_locals(var_name="message"))

        assert _run_with_job(job, exercise) == ["from perm"]

    def test_find_by_name_can_disable_perm_ctx_scan(self):
        """ext_scan=False limits find_locals to stack/local context."""
        job = _FakeJob({"message": "from perm"})

        def exercise():
            return list(find_locals(var_name="message", ext_scan=False))

        assert not _run_with_job(job, exercise)

    def test_find_by_type_scans_perm_ctx_by_default(self):
        """find_locals includes perm_ctx values when matching by type."""
        job = _FakeJob({"answer": 42, "message": "from perm"})

        def exercise():
            return list(find_locals(var_type=int))

        results = _run_with_job(job, exercise)
        assert 42 in results
        assert "from perm" not in results

    def test_local_value_shadows_perm_ctx_value(self):
        """A stack local with the same name takes precedence over perm_ctx."""
        job = _FakeJob({"message": "from perm"})

        def exercise():
            message = "from local"
            return list(find_locals(var_name="message"))

        assert _run_with_job(job, exercise) == ["from local"]

    def test_named_local_hit_does_not_access_perm_ctx(self):
        """find_locals does not fetch perm_ctx when the named local exists."""
        job = _PermCtxRaisesJob()

        def exercise():
            message = "from local"
            return list(find_locals(var_name="message"))

        assert _run_with_job(job, exercise) == ["from local"]


class TestJobFindLocals:
    """Tests for Job.find_locals."""

    def test_finds_values_from_job_local_state(self, job_factory):
        """Job.find_locals scans the job's py_env.local_state."""
        job = job_factory()
        job.py_env.local_state = {"answer": 42, "message": "hello"}

        assert list(job.find_locals(var_name="answer")) == [42]
        assert list(job.find_locals(var_type=str)) == ["hello"]

    def test_scans_perm_ctx_by_default(self, job_factory):
        """Job.find_locals includes perm_ctx when ext_scan is enabled."""
        job = job_factory()
        job.py_env.local_state = {"_PERM_CTX": {"answer": 42}}

        assert list(job.find_locals(var_name="answer")) == [42]
        assert 42 in list(job.find_locals(var_type=int))

    def test_ext_scan_false_ignores_perm_ctx(self, job_factory):
        """Job.find_locals can be limited to py_env.local_state."""
        job = job_factory()
        job.py_env.local_state = {"_PERM_CTX": {"answer": 42}}

        assert not list(job.find_locals(var_name="answer", ext_scan=False))

    def test_local_value_shadows_perm_ctx(self, job_factory):
        """A py_env local with the same name takes precedence over perm_ctx."""
        job = job_factory()
        job.py_env.local_state = {
            "message": "from local",
            "_PERM_CTX": {"message": "from perm"},
        }

        assert list(job.find_locals(var_name="message")) == ["from local"]

    def test_resolves_futures_like_find_locals(self, job_factory, db0_fixture):
        """Job.find_locals uses the same FutureResult matching behavior."""
        job = job_factory()
        job.py_env.local_state = {
            "ready": _make_future("Alice", ready=True),
            "blocked": _make_future("Bob", ready=False),
        }

        assert list(job.find_locals(var_name="ready")) == ["Alice"]
        assert "Alice" in list(job.find_locals(var_type=str))
        assert "Bob" not in list(job.find_locals(var_type=str))
        with pytest.raises(FutureError):
            list(job.find_locals(var_name="blocked"))


class TestGetCurrentAgent:
    """Tests for get_current_agent() helper."""

    def test_returns_none_without_context(self):
        """get_current_agent returns None when no _STATEK_CTX is available."""
        assert get_current_agent() is None

    def test_returns_agent_via_local_context(self):
        """get_current_agent retrieves the agent object from _STATEK_CTX."""
        mock_agent = object()
        ctx = {"agent": mock_agent}

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_agent()

        result = inject_context(fn, {"_STATEK_CTX": ctx})()
        assert result is mock_agent

    def test_returns_none_when_ctx_has_no_agent_key(self):
        """get_current_agent returns None when _STATEK_CTX exists but has no 'agent' key."""
        ctx = {}

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_agent()

        result = inject_context(fn, {"_STATEK_CTX": ctx})()
        assert result is None


class TestGetCurrentAgentName:
    """Tests for get_current_agent_name() helper."""

    def test_returns_none_without_context(self):
        """get_current_agent_name returns None when no _STATEK_CTX is available."""
        assert get_current_agent_name() is None

    def test_returns_class_name_via_local_context(self):
        """get_current_agent_name returns the class name without module qualifiers."""
        agent = _FakeAgentClass()
        ctx = {"agent": agent}

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_agent_name()

        result = inject_context(fn, {"_STATEK_CTX": ctx})()
        assert result == "_FakeAgentClass"

    def test_returns_none_when_no_agent_in_ctx(self):
        """get_current_agent_name returns None when _STATEK_CTX has no agent."""
        ctx = {}

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_agent_name()

        result = inject_context(fn, {"_STATEK_CTX": ctx})()
        assert result is None


class TestGetCurrentJob:
    """Tests for get_current_job() helper."""

    def test_returns_none_without_context(self):
        """get_current_job returns None when no _STATEK_CTX is available."""
        assert get_current_job() is None

    def test_returns_job_via_local_context(self):
        """get_current_job retrieves the job object from _STATEK_CTX."""
        mock_job = object()
        ctx = {"job": mock_job}

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_job()

        result = inject_context(fn, {"_STATEK_CTX": ctx})()
        assert result is mock_job

    def test_returns_none_when_ctx_has_no_job_key(self):
        """get_current_job returns None when _STATEK_CTX exists but has no 'job' key."""
        ctx = {}

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_job()

        result = inject_context(fn, {"_STATEK_CTX": ctx})()
        assert result is None

    def test_inject_context_restores_outer_context(self):
        """inject_context temporarily overrides and then restores current Statek context."""
        outer_job = object()
        inner_job = object()

        def fn(**kwargs):  # pylint: disable=unused-argument
            return get_current_job()

        with _statek_ctx_scope({"job": outer_job}):
            result = inject_context(fn, {"_STATEK_CTX": {"job": inner_job}})()
            assert result is inner_job
            assert get_current_job() is outer_job


class TestFindLocalsWithFutures:
    """Tests for find_locals resolution of FutureResult / FutureElement."""

    def test_by_name_resolves_ready_future(self, db0_fixture):
        """find_locals resolves a FutureResult when searching by name."""
        user = _make_future("Alice", ready=True)  # noqa: F841

        results = list(find_locals(var_name='user'))

        assert results == ["Alice"]

    def test_by_name_raises_when_future_not_ready(self, db0_fixture):
        """find_locals raises FutureError when the named future is not ready."""
        user = _make_future("Alice", ready=False)  # noqa: F841

        with pytest.raises(FutureError):
            list(find_locals(var_name='user'))

    def test_by_name_and_type_resolves_future_type_matches(self, db0_fixture):
        """find_locals resolves FutureResult by name+type when types match."""
        user = _make_future("Alice", ready=True)  # noqa: F841

        results = list(find_locals(var_type=str, var_name='user'))

        assert results == ["Alice"]

    def test_by_name_and_type_resolves_future_type_mismatch(self, db0_fixture):
        """find_locals returns nothing when resolved type does not match var_type."""
        user = _make_future(42, ready=True)  # noqa: F841

        results = list(find_locals(var_type=str, var_name='user'))

        assert not results

    def test_by_name_and_type_raises_when_future_not_ready(self, db0_fixture):
        """find_locals raises FutureError by-name even when var_type is specified."""
        user = _make_future("Alice", ready=False)  # noqa: F841

        with pytest.raises(FutureError):
            list(find_locals(var_type=str, var_name='user'))

    def test_by_type_finds_value_inside_ready_future(self, db0_fixture):
        """find_locals by type scans FutureResult instances and yields resolved values."""
        future_user = _make_future("Alice", ready=True)  # noqa: F841

        results = list(find_locals(var_type=str))

        assert "Alice" in results

    def test_by_type_skips_unresolved_future_silently(self, db0_fixture):
        """find_locals by type silently skips FutureResult instances not ready yet."""
        future_user = _make_future("Alice", ready=False)  # noqa: F841

        # Must not raise; just skip
        results = list(find_locals(var_type=str))

        assert "Alice" not in results

    def test_by_type_does_not_yield_future_wrapper(self, db0_fixture):
        """find_locals by type does not yield the FutureResult wrapper itself."""
        future_user = _make_future("Alice", ready=True)  # noqa: F841

        results = list(find_locals(var_type=str))

        # Only the resolved string value should appear, not the future wrapper
        assert all(isinstance(r, str) for r in results)

    def test_by_type_resolves_future_element(self, db0_fixture):
        """find_locals by type resolves FutureElement instances."""
        fut = _make_tuple_future(("hello", 42), ready=True)
        elem_str, elem_int = fut  # unpacks to FutureElement instances

        results = list(find_locals(var_type=str))

        assert "hello" in results

    def test_by_type_skips_unresolved_future_element_silently(self, db0_fixture):
        """find_locals by type silently skips unresolved FutureElement instances."""
        fut = _make_tuple_future(("hello", 42), ready=False)
        elem_str, elem_int = fut  # noqa: F841

        results = list(find_locals(var_type=str))

        assert "hello" not in results

    def test_by_type_does_not_yield_future_element_wrapper(self, db0_fixture):
        """find_locals by type does not yield the FutureElement wrapper itself."""
        from statek.future import FutureElement  # pylint: disable=import-outside-toplevel
        fut = _make_tuple_future(("hello", 42), ready=True)
        elem_str, elem_int = fut  # noqa: F841

        results = list(find_locals(var_type=str))

        assert not any(isinstance(r, FutureElement) for r in results)

    def test_by_name_resolves_ready_future_element(self, db0_fixture):
        """find_locals resolves a FutureElement when searching by name."""
        fut = _make_tuple_future(("Alice", 99), ready=True)
        elem_name, elem_age = fut  # noqa: F841

        results = list(find_locals(var_name='elem_name'))

        assert results == ["Alice"]

    def test_by_name_raises_when_future_element_not_ready(self, db0_fixture):
        """find_locals raises FutureError when the named FutureElement is not ready."""
        fut = _make_tuple_future(("Alice", 99), ready=False)
        elem_name, elem_age = fut  # noqa: F841

        with pytest.raises(FutureError):
            list(find_locals(var_name='elem_name'))

    def test_by_name_and_type_resolves_future_element_type_matches(self, db0_fixture):
        """find_locals resolves FutureElement by name+type when types match."""
        fut = _make_tuple_future(("Alice", 99), ready=True)
        elem_name, elem_age = fut  # noqa: F841

        results = list(find_locals(var_type=str, var_name='elem_name'))

        assert results == ["Alice"]

    def test_by_name_and_type_resolves_future_element_type_mismatch(self, db0_fixture):
        """find_locals returns nothing when resolved FutureElement type doesn't match."""
        fut = _make_tuple_future(("Alice", 99), ready=True)
        elem_name, elem_age = fut  # noqa: F841

        results = list(find_locals(var_type=int, var_name='elem_name'))

        assert not results

    def test_by_name_and_type_raises_when_future_element_not_ready(self, db0_fixture):
        """find_locals raises FutureError by-name even when var_type is specified."""
        fut = _make_tuple_future(("Alice", 99), ready=False)
        elem_name, elem_age = fut  # noqa: F841

        with pytest.raises(FutureError):
            list(find_locals(var_type=str, var_name='elem_name'))


# --- Async tool using find_locals, for exec_step integration tests ---

@tool
async def _async_find_local_tool(var_name: str, **kwargs):
    """Async tool that uses find_locals to locate a variable by name."""
    results = list(find_locals(var_name=var_name))
    if results:
        return results[0]
    return None


def _make_job_with_tools(role, tools):
    """Create a job with the given agent tools."""
    agent = Agent(
        role=role,
        _system_prompt=make_system_prompt("Test"),
        _metadata={"MODEL": "test-model"},
        _tools=tools,
    )
    job_def = JobDef(agent=agent, job_params={"goal": "Test"}, warmup_code=None)
    return Job(
        job_def=job_def,
        model_family="test",
        model="test-model",
        job_status=JobStatus.READY,  # pylint: disable=no-member
    )


class TestFindLocalsFromAsyncToolInExecStep:
    """Tests for find_locals working inside an async @tool called from exec_step."""

    @pytest.mark.asyncio
    async def test_async_tool_finds_variable_from_prior_exec_step(self, db0_fixture):
        """Async tool using find_locals locates a variable initialized in a prior exec_step."""
        job = _make_job_with_tools("async_find_local", [_async_find_local_tool])

        await exec_step('greeting = "hello world"', job)
        await exec_step('result = _async_find_local_tool(var_name="greeting")', job)

        assert job.py_env.local_state['result'] == "hello world"

    @pytest.mark.asyncio
    async def test_async_tool_finds_variable_from_same_exec_step(self, db0_fixture):
        """Async tool using find_locals locates a variable initialized in the same exec_step."""
        job = _make_job_with_tools("async_find_local_same", [_async_find_local_tool])

        await exec_step(
            'greeting = "hello world"\nresult = _async_find_local_tool(var_name="greeting")',
            job,
        )

        assert job.py_env.local_state['result'] == "hello world"

    @pytest.mark.asyncio
    async def test_async_tool_returns_none_for_missing_variable(self, db0_fixture):
        """Async tool using find_locals returns None when the variable does not exist."""
        job = _make_job_with_tools("async_find_local_miss", [_async_find_local_tool])

        await exec_step('result = _async_find_local_tool(var_name="nonexistent")', job)

        assert job.py_env.local_state['result'] is None

class TestFindLocalsInAsyncTools:
    """Tests that find_locals works correctly in both sync and async @tool functions.

    The key scenario: a variable is defined in the calling frame (e.g., inside LLM-executed
    code), but the @tool async function cannot reach that frame via stack traversal because
    asyncio internals consume ~5 extra frames between the coroutine and the calling code.

    The fix ensures inject_context captures the live call-stack context before the coroutine
    is launched, so find_locals can retrieve variables regardless of asyncio frame overhead.
    """

    def _make_sync_tool(self):
        @tool
        def sync_tool(**kwargs):  # pylint: disable=unused-variable
            return list(find_locals(var_name='message'))
        return inject_context(sync_tool, {})

    def _make_async_tool(self):
        @tool
        async def async_tool(**kwargs):  # pylint: disable=unused-variable
            return list(find_locals(var_name='message'))
        return inject_context(async_tool, {})

    def test_sync_tool_finds_variable_from_calling_frame(self):
        """find_locals in a sync @tool finds a variable defined in the calling async frame.

        Baseline: sync tools work via direct frame traversal.
        """
        wrapped = self._make_sync_tool()

        async def run():
            message = 'Hello'  # noqa: F841
            def do_call():
                # intermediate frame — message is one level up in run()
                return _smart_call(wrapped)
            return do_call()

        result = asyncio.run(run())
        assert result == ['Hello']

    def test_async_tool_finds_variable_from_calling_frame(self):
        """find_locals in an async @tool finds a variable defined in the calling async frame.

        This is the key regression test. Without the fix, asyncio consumes ~5 extra frames
        via run_until_complete, pushing the calling frame beyond max_frames=10.
        """
        wrapped = self._make_async_tool()

        async def run():
            message = 'Hello'  # noqa: F841
            def do_call():
                # intermediate frame — message is one level up in run()
                return _smart_call(wrapped)
            return do_call()

        result = asyncio.run(run())
        assert result == ['Hello']

    def test_sync_tool_finds_named_parameter(self):
        """find_locals finds a named parameter passed directly to a sync @tool."""
        @tool
        def param_sync_tool(message: str, **kwargs):  # pylint: disable=unused-variable
            return list(find_locals(var_name='message'))

        result = inject_context(param_sync_tool, {})("Hello")
        assert result == ["Hello"]

    def test_async_tool_finds_named_parameter(self):
        """find_locals finds a named parameter passed directly to an async @tool."""
        @tool
        async def param_async_tool(message: str, **kwargs):  # pylint: disable=unused-variable
            return list(find_locals(var_name='message'))

        async def run():
            return inject_context(param_async_tool, {})("Hello")

        result = asyncio.run(run())
        assert result == ["Hello"]

    def test_sync_tool_finds_variable_from_injected_context(self):
        """find_locals finds a variable pre-captured in the injected context (sync tool)."""
        @tool
        def ctx_sync_tool(**kwargs):  # pylint: disable=unused-variable
            return list(find_locals(var_name='message'))

        result = inject_context(ctx_sync_tool, {"message": "from_context"})()
        assert result == ["from_context"]

    def test_async_tool_finds_variable_from_injected_context(self):
        """find_locals finds a variable pre-captured in the injected context (async tool)."""
        @tool
        async def ctx_async_tool(**kwargs):  # pylint: disable=unused-variable
            return list(find_locals(var_name='message'))

        async def run():
            return inject_context(ctx_async_tool, {"message": "from_context"})()

        result = asyncio.run(run())
        assert result == ["from_context"]
