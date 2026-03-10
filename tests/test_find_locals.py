# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable,protected-access
from typing import Tuple
import pytest
import dbzero as db0
from statek.utils import find_locals, get_current_agent, get_current_agent_name, get_current_job
from statek.system import inject_context
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
