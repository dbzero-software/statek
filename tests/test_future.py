"""Tests for statek.future module."""

# pylint: disable=unused-argument

import pytest
import dbzero as db0
from statek.future import (
    temporal,
    FutureResult,
    FutureError,
    get_any,
    get_all,
)

@db0.memo
class MockFutureResult(FutureResult):
    """Mock subclass of FutureResult for testing."""

    def __init__(self, value, condition=True):
        """Initialize mock with default values."""
        super().__init__(deps=None, state_num=1)
        self.value = value
        self.condition = condition

def complement_function(fut: MockFutureResult):
    return fut.value

def condition_function(fut: MockFutureResult):
    return fut.condition

@temporal(complement_function, condition_function)
def make_mock_future(value, condition=True):
    return MockFutureResult(value, condition)


class TestTemporalDecorator:
    """Test cases for temporal decorator."""

    @staticmethod
    def _complement_function(fut: MockFutureResult):
        return fut.value

    @staticmethod
    def _condition_function(fut: MockFutureResult):
        return fut.condition


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
    """Test cases for get_any and related functions."""

    def test_get_any_result_value_when_ready(self, db0_fixture):
        """Test accessing value property of get_any result when a future is ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_any(fut1, fut2)

        # The temporal decorator should have set the complement functions
        assert combined.value == "value2"

    def test_get_any_result_value_raises_when_not_ready(self, db0_fixture):
        """Test accessing value property of get_any result raises when no futures are ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_any(fut1, fut2)

        with pytest.raises(FutureError):
            _ = combined.value

    def test_get_any_check_condition_returns_true(self, db0_fixture):
        """Test check_condition on get_any result returns True when any future is ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_any(fut1, fut2)

        assert combined.check_condition() is True

    def test_get_any_check_condition_returns_false(self, db0_fixture):
        """Test check_condition on get_any result returns False when no futures are ready."""
        fut1 = make_mock_future("value1", condition=False)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_any(fut1, fut2)

        assert combined.check_condition() is False

    def test_get_any_raises_on_empty_args(self, db0_fixture):
        """Test get_any raises TypeError when called with no arguments."""
        with pytest.raises(TypeError):
            get_any()


class TestGetAll:
    """Test cases for get_all and related functions."""

    def test_get_all_result_value_when_ready(self, db0_fixture):
        """Test accessing value property of get_all result when all futures are ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_all(fut1, fut2)

        # The temporal decorator should have set the complement functions
        assert set(combined.value) == set(("value1", "value2"))

    def test_get_all_result_value_raises_when_not_ready(self, db0_fixture):
        """Test accessing value property of get_all result raises when any future is not ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_all(fut1, fut2)

        with pytest.raises(FutureError):
            _ = combined.value

    def test_get_all_check_condition_returns_true(self, db0_fixture):
        """Test check_condition on get_all result returns True when all futures are ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=True)

        combined = get_all(fut1, fut2)

        assert combined.check_condition() is True

    def test_get_all_check_condition_returns_false(self, db0_fixture):
        """Test check_condition on get_all result returns False when any future is not ready."""
        fut1 = make_mock_future("value1", condition=True)
        fut2 = make_mock_future("value2", condition=False)

        combined = get_all(fut1, fut2)

        assert combined.check_condition() is False

    def test_get_all_raises_on_empty_args(self, db0_fixture):
        """Test get_all raises TypeError when called with no arguments."""
        with pytest.raises(TypeError):
            get_all()
