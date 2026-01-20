from dataclasses import dataclass
from typing import Any, Set, Tuple, Callable
import functools
import inspect
import dbzero as db0

@db0.memo
@dataclass
class FutureResult:
    """
    Either a single or a set of alternative memo class instances
    Set - awaits change in ANY of the instances
    Tuple - awaits change in ALL of the instances
    """
    deps: Any | Set[Any] | Tuple[Any]
    """
    The state number from which the modifications are tracked from
    (state before change)
    The state is associated with the prefix of `deps` objects
    """
    state_num: int
    """Continuation condition"""
    __check_condition: Callable = None
    __fetch_result: Callable = None

    def set_complement_functions(self, complement: Callable, condition: Callable):
        self.__check_condition = condition
        self.__fetch_result = complement

    @property
    def value(self):
        """
        Retrieve result (if available) or raise FutureError
        """
        return self.__fetch_result(self)

    def check_condition(self):
        return self.__check_condition(self)


def _handle_temporal_function_result(
    result: FutureResult | Any,
    complement: Callable,
    condition: Callable
):

    if isinstance(result, FutureResult):
        # Inject complement functions for returned FutureResult
        result.set_complement_functions(complement, condition)
    return result


def temporal(complement: Callable[[FutureResult], Any], condition: Callable[[FutureResult], bool]):
    """
    Decorates a temporal function to properly handle future results.
    This decorator is mandatory for marking temporal functions.

    Args:
        complement: Function to retrieve the result value when temporal function completes.
        condition: Function to check if completion criteria was met.

    Returns:
        Decorated temporal function
    """

    def decorator(f):
        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                result = await f(*args, **kwargs)
                return _handle_temporal_function_result(result, complement, condition)
            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            return _handle_temporal_function_result(result, complement, condition)
        return wrapper

    return decorator
