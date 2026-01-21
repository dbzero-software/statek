from dataclasses import dataclass
from typing import Any, Optional, Set, Tuple, Callable, Sequence
import functools
import inspect
import dbzero as db0
from . import tool

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


@dataclass
class FutureError(Exception):
    """
    Raised by a temporal function when trying to retrieve a response which is not available yet.

    Attributes:
        future_result: The awaited result.
        instr_num: The instruction number (to continue from).
    """
    future_result: FutureResult
    instr_num: Optional[int] = None


@db0.memo
class CombinedFutureResult(FutureResult):
    """
    A FutureResult that aggregates multiple FutureResult instances.
    
    This class allows combining multiple futures and checking their completion
    status collectively. It's used to create composite futures that can be
    evaluated based on whether any or all of the constituent futures have completed.
    
    Args:
        futures: A sequence of FutureResult instances to combine.
        *args: Positional arguments passed to the parent FutureResult.
        **kwargs: Peyword arguments passed to the parent FutureResult.
    
    Attributes:
        futures: The sequence of FutureResult instances being tracked.
    """
    def __init__(self, futures: Sequence[FutureResult], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.futures = futures


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


def check_any_completed(combined: CombinedFutureResult) -> bool:
    """
    Check if any of the futures in a combined future have completed.
    
    Args:
        combined: A CombinedFutureResult containing multiple futures.
    
    Returns:
        True if at least one future has completed, False otherwise.
    """
    return any(fut.check_condition() for fut in combined.futures)


def get_any_result(combined: CombinedFutureResult) -> Any:
    """
    Retrieve the value of the first completed future from a combined future.
    
    Args:
        combined: A CombinedFutureResult containing multiple futures.
    
    Returns:
        The value of the first future that has completed.
    
    Raises:
        FutureError: If none of the futures have completed yet.
    """
    for fut in combined.futures:
        if fut.check_condition():
            return fut.value
    # None of combined future results is ready
    raise FutureError(combined)


@temporal(complement=get_any_result, condition=check_any_completed)
@tool
def get_any(*args: FutureResult) -> CombinedFutureResult:
    """
    Create a combined future that will yield a result when any of the input futures complete.
    
    This function combines multiple futures into a single CombinedFutureResult.
    When the result's value is accessed, it will return the value of the first
    completed future, or raise FutureError if none have completed yet.
    
    Args:
        *args: Variable number of FutureResult instances to combine.
    
    Returns:
        A CombinedFutureResult that tracks all input futures.
    
    Raises:
        TypeError: If no futures are provided.
    """
    if not args:
        raise TypeError("get_any requires at least one FutureResult argument")
    return CombinedFutureResult(args, deps=None, state_num=0)


def check_all_completed(combined: CombinedFutureResult) -> bool:
    """
    Check if all futures in a combined future have completed.
    
    Args:
        combined: A CombinedFutureResult containing multiple futures.
    
    Returns:
        True if all futures have completed, False otherwise.
    """
    return all(fut.check_condition() for fut in combined.futures)


def get_all_result(combined: CombinedFutureResult) -> Tuple[Any]:
    """
    Retrieve the values of all futures from a combined future.
    
    Args:
        combined: A CombinedFutureResult containing multiple futures.
    
    Returns:
        A tuple containing the values of all completed futures, in order.
    
    Raises:
        FutureError: If any of the futures have not completed yet.
    """
    results = []
    for fut in combined.futures:
        if not fut.check_condition():
            # One of combined future results is not ready
            raise FutureError(combined)
        results.append(fut.value)
    return tuple(results)


@temporal(complement=get_all_result, condition=check_all_completed)
@tool
def get_all(*args: FutureResult) -> CombinedFutureResult:
    """
    Create a combined future that will yield a result when all input futures complete.
    
    This function combines multiple futures into a single CombinedFutureResult.
    When the result's value is accessed, it will return a tuple of all futures' values
    if all have completed, or raise FutureError if any have not completed yet.
    
    Args:
        *args: Variable number of FutureResult instances to combine.
    
    Returns:
        A CombinedFutureResult that tracks all input futures.
    
    Raises:
        TypeError: If no futures are provided.
    """
    if not args:
        raise TypeError("get_all requires at least one FutureResult argument")
    return CombinedFutureResult(args, deps=None, state_num=0)
