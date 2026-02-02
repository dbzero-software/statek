# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable
from dataclasses import dataclass
import pytest
import dbzero as db0
from statek.future import FutureResult, temporal
from statek.utils import format_callable_decl


# Module-level test data classes and functions
@db0.memo
@dataclass
class SimpleData:
    value: int = 0


def get_simple_data(future: FutureResult) -> int:
    return future.deps.value


def is_simple_data_ready(future: FutureResult) -> bool:
    return future.deps.value > 0


@temporal(complement=get_simple_data, condition=is_simple_data_ready)
def base_simple_func(data: SimpleData) -> FutureResult:
    return FutureResult(deps=data, state_num=0)


@temporal(extends=base_simple_func)
def extended_simple_func(x: int) -> FutureResult:
    """Extended version that takes int instead of SimpleData."""
    return base_simple_func(SimpleData(value=x))


@temporal(extends=extended_simple_func)
def chained_extended_func(x: str) -> FutureResult:
    """Chained extension that takes str."""
    return extended_simple_func(int(x))


class TestTemporalExtends:
    """Tests for temporal function extension mechanism."""

    def test_basic_extension(self, db0_fixture):
        """Test basic extension of a temporal function."""
        result = extended_simple_func(42)
        assert isinstance(result, FutureResult)
        assert result.check_condition()
        assert result.value == 42

    def test_extension_chain(self, db0_fixture):
        """Test chaining multiple extensions."""
        result = chained_extended_func("100")
        assert isinstance(result, FutureResult)
        assert result.check_condition()
        assert result.value == 100

    def test_extension_preserves_complement_condition(self, db0_fixture):
        """Test that extension uses the base function's complement and condition."""
        # The extended function should use the same complement/condition as base
        assert extended_simple_func.__temporal_complement__ is get_simple_data
        assert extended_simple_func.__temporal_condition__ is is_simple_data_ready

        # Test it works correctly
        result = extended_simple_func(50)
        assert result.value == 50

    def test_extension_error_cannot_extend_non_temporal(self, db0_fixture):
        """Test that extending a non-temporal function raises an error."""
        def regular_function(x):
            return x * 2

        with pytest.raises(ValueError, match="not a temporal function"):
            @temporal(extends=regular_function)
            def bad_extension(x: int) -> FutureResult:
                return regular_function(x)

    def test_extension_error_cannot_specify_both_extends_and_complement(self, db0_fixture):
        """Test that specifying both extends and complement raises an error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            @temporal(extends=base_simple_func, complement=get_simple_data,
                      condition=is_simple_data_ready)
            def bad_extension(x: int) -> FutureResult:
                return base_simple_func(SimpleData(value=x))

    def test_extension_error_missing_arguments(self, db0_fixture):
        """Test that not providing complement/condition or extends raises an error."""
        with pytest.raises(ValueError, match="Must specify either"):
            @temporal()
            def bad_func(x: int) -> FutureResult:
                return FutureResult(deps=None, state_num=0)

    def test_extension_reports_correct_return_type(self, db0_fixture):
        """Test that extended functions report the correct return type via format_callable_decl."""
        # Extended function should report the complement's return type (int), not FutureResult
        result = format_callable_decl(extended_simple_func)
        assert result == "def extended_simple_func(x: int) -> int"

        # Chained extension should also report the same return type
        result = format_callable_decl(chained_extended_func)
        assert result == "def chained_extended_func(x: str) -> int"

        # Base function should also report the complement's return type
        # Note: dbzero wraps the class name with "Memo_" prefix
        result = format_callable_decl(base_simple_func)
        assert result == "def base_simple_func(data: Memo_SimpleData) -> int"
