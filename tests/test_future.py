"""Tests for statek.future module."""

# pylint: disable=unused-argument

from typing import Tuple
import pytest
import dbzero as db0
from statek.future import (
    temporal,
    FutureResult,
    FutureElement,
    FutureError,
    get_any_future,
    get_all_future,
    get_unpack_size,
)

@db0.memo
class MockFutureResult(FutureResult):
    """Mock subclass of FutureResult for testing."""

    def __init__(self, result_value, condition=True):
        """Initialize mock with default values."""
        super().__init__(deps=None, state_num=1)
        self.result_value = result_value
        self.condition = condition

def complement_function(fut: MockFutureResult):
    return fut.result_value

def condition_function(fut: MockFutureResult):
    return fut.condition

@temporal(complement_function, condition_function)
def make_mock_future(result_value, condition=True):
    return MockFutureResult(result_value, condition)


@db0.memo
class MockTupleFutureResult(FutureResult):
    """Mock subclass of FutureResult for testing tuple returns."""

    def __init__(self, result_value: Tuple, condition=True):
        """Initialize mock with tuple value."""
        super().__init__(deps=None, state_num=1)
        self.result_value = result_value
        self.condition = condition

def tuple_complement_function(fut: MockTupleFutureResult) -> Tuple[str, int]:
    if not fut.condition:
        raise FutureError(fut)
    return fut.result_value

def tuple_condition_function(fut: MockTupleFutureResult):
    return fut.condition

@temporal(tuple_complement_function, tuple_condition_function)
def make_mock_tuple_future(result_value: Tuple, condition=True):
    return MockTupleFutureResult(result_value, condition)


@db0.memo
class MockTripleFutureResult(FutureResult):
    """Mock subclass of FutureResult for testing triple tuple returns."""

    def __init__(self, result_value: Tuple[str, int, float], condition=True):
        """Initialize mock with triple tuple value."""
        super().__init__(deps=None, state_num=1)
        self.result_value = result_value
        self.condition = condition

def triple_complement_function(fut: MockTripleFutureResult) -> Tuple[str, int, float]:
    if not fut.condition:
        raise FutureError(fut)
    return fut.result_value

def triple_condition_function(fut: MockTripleFutureResult):
    return fut.condition

@temporal(triple_complement_function, triple_condition_function)
def make_mock_triple_future(result_value: Tuple[str, int, float], condition=True):
    return MockTripleFutureResult(result_value, condition)


class TestTemporalDecorator:
    """Test cases for temporal decorator."""

    def test_temporal_with_regular_function_simple_result(self, db0_fixture):
        """Test temporal decorator with regular function returning simple result."""
        exp_vals = (42, MockFutureResult("result_value"))
        exp_vals_seq = iter(exp_vals)

        @temporal(complement_function, condition_function)
        def temporal_function():
            return next(exp_vals_seq)

        assert temporal_function() == exp_vals[0]
        result = temporal_function()
        assert result is exp_vals[1]
        assert result.value == "result_value"
        assert result.check_condition() is True

    @pytest.mark.asyncio
    async def test_temporal_with_async_function_simple_result(self, db0_fixture):
        """Test temporal decorator with async function returning simple result."""
        exp_vals = (42, MockFutureResult("result_value", False))
        exp_vals_seq = iter(exp_vals)

        @temporal(complement_function, condition_function)
        async def temporal_function():
            return next(exp_vals_seq)

        assert await temporal_function() == exp_vals[0]
        result = await temporal_function()
        assert result is exp_vals[1]
        assert result.value == "result_value"
        assert result.check_condition() is False

    def test_temporal_with_args_and_kwargs(self, db0_fixture):
        """Test temporal decorator preserves function arguments."""
        @temporal(complement_function, condition_function)
        def function_with_params(a, b, c=10):
            return a + b + c

        assert function_with_params(5, 15, c=20) == 40

    @pytest.mark.asyncio
    async def test_temporal_with_async_args_and_kwargs(self, db0_fixture):
        """Test temporal decorator preserves async function arguments."""
        @temporal(complement_function, condition_function)
        async def function_with_params(a, b, c=10):
            return a + b + c

        assert await function_with_params(5, 15, c=20) == 40


class TestGetAny:
    """Test cases for get_any_future and related functions."""

    def test_get_any_result_value_when_ready(self, db0_fixture):
        """Test accessing value property of get_any_future result when a future is ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_any_future(fut1, fut2)

        # The temporal decorator should have set the complement functions
        assert combined.value == "value2"

    def test_get_any_result_value_raises_when_not_ready(self, db0_fixture):
        """Test accessing get_any_future result raises when no futures are ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_any_future(fut1, fut2)

        with pytest.raises(FutureError):
            _ = combined.value

    def test_get_any_check_condition_returns_true(self, db0_fixture):
        """Test check_condition on get_any_future result returns True when any future is ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_any_future(fut1, fut2)

        assert combined.check_condition() is True

    def test_get_any_check_condition_returns_false(self, db0_fixture):
        """Test check_condition on get_any_future result returns False when no futures are ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_any_future(fut1, fut2)

        assert combined.check_condition() is False

    def test_get_any_raises_on_empty_args(self, db0_fixture):
        """Test get_any_future raises TypeError when called with no arguments."""
        with pytest.raises(TypeError):
            get_any_future()


class TestGetAll:
    """Test cases for get_all_future and related functions."""

    def test_get_all_result_value_when_ready(self, db0_fixture):
        """Test accessing value property of get_all_future result when all futures are ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_all_future(fut1, fut2)

        # The temporal decorator should have set the complement functions
        assert set(combined.value) == set(("value1", "value2"))

    def test_get_all_result_value_raises_when_not_ready(self, db0_fixture):
        """Test accessing get_all_future result raises when any future is not ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_all_future(fut1, fut2)

        with pytest.raises(FutureError):
            _ = combined.value

    def test_get_all_check_condition_returns_true(self, db0_fixture):
        """Test check_condition on get_all_future result returns True when all futures are ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_all_future(fut1, fut2)

        assert combined.check_condition() is True

    def test_get_all_check_condition_returns_false(self, db0_fixture):
        """Test check_condition on get_all_future returns False when any future is not ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_all_future(fut1, fut2)

        assert combined.check_condition() is False

    def test_get_all_raises_on_empty_args(self, db0_fixture):
        """Test get_all_future raises TypeError when called with no arguments."""
        with pytest.raises(TypeError):
            get_all_future()


class TestGetUnpackSize:
    """Test cases for get_unpack_size function."""

    def test_get_unpack_size_with_tuple_annotation(self):
        """Test with a function returning a tuple with explicit size."""
        def returns_pair() -> Tuple[int, str]:
            return (1, "hello")

        assert get_unpack_size(returns_pair) == 2

    def test_get_unpack_size_with_triple(self):
        """Test with a function returning a tuple of three elements."""
        def returns_triple() -> Tuple[int, str, float]:
            return (1, "hello", 3.14)

        assert get_unpack_size(returns_triple) == 3

    def test_get_unpack_size_with_no_annotation(self):
        """Test with a function that has no return type annotation."""
        def no_annotation():
            return (1, 2)

        assert get_unpack_size(no_annotation) is None

    def test_get_unpack_size_with_non_tuple_annotation(self):
        """Test with a function returning a non-tuple type."""
        def returns_int() -> int:
            return 42

        assert get_unpack_size(returns_int) is None


class TestFutureElement:
    """Test cases for FutureElement class and unpacking functionality."""

    def test_unpack_tuple_future(self, db0_fixture):
        """Test unpacking a FutureResult that returns a tuple."""
        fut = make_mock_tuple_future(("hello", 42), condition=True)

        # Unpack the future
        elem1, elem2 = fut

        # Check that we got FutureElement instances
        assert isinstance(elem1, FutureElement)
        assert isinstance(elem2, FutureElement)

        # Check that values are correct
        assert elem1.value == "hello"
        assert elem2.value == 42

    def test_unpack_future_check_condition(self, db0_fixture):
        """Test that FutureElement check_condition delegates to parent."""
        fut = make_mock_tuple_future(("hello", 42), condition=True)

        elem1, elem2 = fut

        # Both elements should have the same condition as parent
        assert elem1.check_condition() is True
        assert elem2.check_condition() is True

    def test_unpack_future_not_ready(self, db0_fixture):
        """Test that FutureElement raises error when parent is not ready."""
        fut = make_mock_tuple_future(("hello", 42), condition=False)

        elem1, elem2 = fut

        # Accessing value should raise FutureError when not ready
        assert elem1.check_condition() is False
        assert elem2.check_condition() is False

        with pytest.raises(FutureError):
            _ = elem1.value

    def test_unpack_non_tuple_future_raises(self, db0_fixture):
        """Test that unpacking non-tuple FutureResult raises FutureError."""
        fut = make_mock_future("single_value", condition=True)

        # Attempting to unpack should raise FutureError
        with pytest.raises(FutureError):
            _, _ = fut

    def test_future_element_preserves_parent_deps(self, db0_fixture):
        """Test that FutureElement preserves parent dependencies."""
        fut = make_mock_tuple_future(("hello", 42), condition=True)

        elem1, elem2 = fut

        # Elements should have same deps as parent
        assert elem1.deps == fut.deps
        assert elem2.deps == fut.deps
        assert elem1.state_num == fut.state_num
        assert elem2.state_num == fut.state_num

    def test_unpack_triple_future(self, db0_fixture):
        """Test unpacking a FutureResult with three elements."""
        fut = make_mock_triple_future(("hello", 42, 3.14), condition=True)

        elem1, elem2, elem3 = fut

        assert elem1.value == "hello"
        assert elem2.value == 42
        assert elem3.value == 3.14
