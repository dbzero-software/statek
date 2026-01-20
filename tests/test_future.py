"""Tests for statek.future module."""

# pylint: disable=unused-argument

import pytest
import dbzero as db0
from statek.future import temporal, FutureResult

@db0.memo
class MockFutureResult(FutureResult):
    """Mock subclass of FutureResult for testing."""

    def __init__(self, value, condition=True):
        """Initialize mock with default values."""
        super().__init__(deps=None, state_num=1)
        self.value = value
        self.condition = condition


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

        @temporal(self._complement_function, self._condition_function)
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

        @temporal(self._complement_function, self._condition_function)
        async def temporal_function():
            return next(exp_vals_seq)

        assert await temporal_function() == exp_vals[0]
        result = await temporal_function()
        assert result is exp_vals[1]
        assert result.value == "result_value"
        assert result.check_condition() is False

    def test_temporal_with_args_and_kwargs(self, db0_fixture):
        """Test temporal decorator preserves function arguments."""
        @temporal(self._complement_function, self._condition_function)
        def function_with_params(a, b, c=10):
            return a + b + c

        assert function_with_params(5, 15, c=20) == 40

    @pytest.mark.asyncio
    async def test_temporal_with_async_args_and_kwargs(self, db0_fixture):
        """Test temporal decorator preserves async function arguments."""
        @temporal(self._complement_function, self._condition_function)
        async def function_with_params(a, b, c=10):
            return a + b + c

        assert await function_with_params(5, 15, c=20) == 40
