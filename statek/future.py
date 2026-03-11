from dataclasses import dataclass
from typing import Any, Set, Tuple, Callable, Sequence, get_origin, get_args
import functools
import inspect
import dbzero as db0
from .exceptions import FutureError


def get_unpack_size(func: Callable) -> int | None:
    """Determines the declared size (element count) of a tuple returned by a callable.

    Args:
        func: The function whose return type is to be analyzed.

    Returns:
        The tuple size if the return type is a tuple, None otherwise.

    Examples:
        def returns_pair() -> Tuple[int, str]: ...
        get_unpack_size(returns_pair)  # Returns 2

        def returns_int() -> int: ...
        get_unpack_size(returns_int)  # Returns None
    """
    try:
        # Get the function signature
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        # If there's no annotation or it's empty, return None
        if return_annotation is inspect.Signature.empty:
            return None

        # Get the origin of the type (e.g., tuple from Tuple[int, str])
        origin = get_origin(return_annotation)

        # Check if it's a tuple type
        if origin is tuple:
            # Get the tuple element types
            args = get_args(return_annotation)

            # Handle Tuple[int, ...] (variable length tuple)
            if len(args) == 2 and args[1] is Ellipsis:
                return None

            # Return the number of elements in the tuple
            return len(args) if args else None

        return None
    except (ValueError, TypeError):
        return None

@db0.memo
@dataclass
class FutureResult:
    """Represents a future result that tracks dependencies and their state changes.

    This class is used to represent asynchronous or pending operations that depend on
    changes to memo class instances. It can track single instances or multiple instances
    with different completion criteria.

    Attributes:
        deps: Either a single or a set of alternative memo class instances.
              Set - awaits change in ANY of the instances.
              Tuple - awaits change in ALL of the instances.
        state_num: The state number from which modifications are tracked from
                   (state before change). The state is associated with the prefix
                   of `deps` objects.
    """
    deps: Any | Set[Any] | Tuple[Any]
    state_num: int
    """Continuation condition"""
    __check_condition: Callable = None
    __fetch_result: Callable = None
    """Deferred error-handler registration (set by bind_error_handler)"""
    __pending_error_handler: Callable = None
    __pending_error_handler_job: Any = None

    def set_complement_functions(self, complement: Callable, condition: Callable):
        self.__check_condition = condition
        self.__fetch_result = complement

    def bind_error_handler(self, error_handler: Callable, job: Any) -> None:
        """Attach an error handler to be registered when value is first retrieved.

        The handler is registered on *job* (via ``job.add_error_handler``) with the
        resolved value as context the first time ``.value`` returns successfully.
        Subsequent accesses to ``.value`` do not re-register the handler.

        Args:
            error_handler: A callable satisfying the error-handler protocol.
            job: The Job instance on which to register the handler.
        """
        self.__pending_error_handler = error_handler
        self.__pending_error_handler_job = job

    @property
    def value(self):
        """Retrieve result (if available) or raise FutureError.

        On the first successful retrieval, any pending error handler bound via
        :meth:`bind_error_handler` is registered on the associated job with the
        resolved value as context.
        """
        result = self.__fetch_result(self)
        if self.__pending_error_handler is not None:
            self.__pending_error_handler_job.add_error_handler(
                self.__pending_error_handler, result
            )
            self.__pending_error_handler = None
            self.__pending_error_handler_job = None
        return result

    def check_condition(self):
        """Check if completion condition was met"""
        return self.__check_condition(self)

    def __iter__(self):
        """Allow unpacking of FutureResult if it returns a tuple.

        Yields:
            FutureElement instances for each element in the tuple.

        Raises:
            FutureError: If the result cannot be unpacked.
        """
        # Get the size of the tuple to unpack
        unpack_size = get_unpack_size(self.__fetch_result)

        if unpack_size is None:
            raise FutureError(self, "Cannot unpack FutureResult: return type is not a tuple")

        # Yield FutureElement for each element
        for elem_id in range(unpack_size):
            yield FutureElement(self, elem_id)


def _future_element_get_value(elem: 'FutureElement'):
    """Get the value of a FutureElement by accessing its parent's value."""
    return elem.get_result()

def _future_element_check_condition(elem: 'FutureElement'):
    """Check if a FutureElement's parent condition is met."""
    return elem.check_condition()

@db0.memo
class FutureElement(FutureResult):
    """Represents a single element of a tuple returned by a FutureResult.

    FutureElement instances allow the use of Python's unpacking syntax for
    FutureResult objects without having to await them immediately.

    Attributes:
        __parent_result: The parent FutureResult instance.
        __elem_id: The index of this element in the parent's tuple result.
    """
    def check_condition(self):
        return self.__parent_result.check_condition()

    def get_result(self):
        return self.__parent_result.value[self.__elem_id]

    def __init__(self, parent_result: FutureResult, elem_id: int):
        """Initialize a FutureElement.

        Args:
            parent_result: The parent FutureResult instance.
            elem_id: The index of this element in the parent's tuple result.
        """
        super().__init__(deps=parent_result.deps, state_num=parent_result.state_num)
        self.__parent_result = parent_result
        self.__elem_id = elem_id
        # Set complement functions using global functions
        self.set_complement_functions(
            _future_element_get_value,
            _future_element_check_condition
        )


@db0.memo
class CombinedFutureResult(FutureResult):
    """A FutureResult that aggregates multiple FutureResult instances.

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


def temporal(complement: Callable[[FutureResult], Any] = None,
             condition: Callable[[FutureResult], bool] = None,
             extends: Callable = None):
    """Decorates a temporal function to properly handle future results.

    This decorator is mandatory for marking temporal functions. A temporal function
    can either define its own complement/condition functions, or extend another
    temporal function by forwarding to it with modified arguments.

    Args:
        complement: Function to retrieve the result value when temporal function completes.
                   Not required when extending another function.
        condition: Function to check if completion criteria was met.
                  Not required when extending another function.
        extends: Another temporal function to extend. When provided, this function
                should forward to the extended function and return its result.

    Returns:
        Decorated temporal function

    Examples:
        # Define a new temporal function
        @temporal(complement=get_result, condition=is_complete)
        def my_func(x: int) -> FutureResult:
            return FutureResult(...)

        # Extend an existing temporal function
        @temporal(extends=my_func)
        def extended_func(x: str) -> FutureResult:
            return my_func(int(x))
    """
    # Validate arguments
    if extends is not None:
        # When extending, we don't need complement/condition
        if complement is not None or condition is not None:
            raise ValueError("Cannot specify both 'extends' and 'complement'/'condition'")
        # Get complement/condition from extended function
        if not hasattr(extends, '__temporal_complement__') or \
           not hasattr(extends, '__temporal_condition__'):
            raise ValueError(
                f"Cannot extend {extends.__name__}: it is not a temporal function")
        complement = extends.__temporal_complement__
        condition = extends.__temporal_condition__
    else:
        # When not extending, complement and condition are required
        if complement is None or condition is None:
            raise ValueError("Must specify either 'extends' or both 'complement' and 'condition'")

    def decorator(f):
        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                result = await f(*args, **kwargs)
                return _handle_temporal_function_result(result, complement, condition)
            async_wrapper.__is_temporal__ = True
            async_wrapper.__temporal_complement__ = complement
            async_wrapper.__temporal_condition__ = condition
            if extends is not None:
                async_wrapper.__extends__ = extends
            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            return _handle_temporal_function_result(result, complement, condition)
        wrapper.__is_temporal__ = True
        wrapper.__temporal_complement__ = complement
        wrapper.__temporal_condition__ = condition
        if extends is not None:
            wrapper.__extends__ = extends
        return wrapper

    return decorator


def check_any_completed(combined: CombinedFutureResult) -> bool:
    """Check if any of the futures in a combined future have completed.

    Args:
        combined: A CombinedFutureResult containing multiple futures.

    Returns:
        True if at least one future has completed, False otherwise.
    """
    return any(fut.check_condition() for fut in combined.futures)


def get_any_result(combined: CombinedFutureResult) -> Any:
    """Retrieve the value of the first completed future from a combined future.

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
def get_any_future(*args: FutureResult) -> CombinedFutureResult:
    """Create a combined future that will yield a result when any of the input futures complete.

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
        raise TypeError("get_any_future requires at least one argument")
    return CombinedFutureResult(args, deps=None, state_num=0)


def check_all_completed(combined: CombinedFutureResult) -> bool:
    """Check if all futures in a combined future have completed.

    Args:
        combined: A CombinedFutureResult containing multiple futures.

    Returns:
        True if all futures have completed, False otherwise.
    """
    return all(fut.check_condition() for fut in combined.futures)


def get_all_result(combined: CombinedFutureResult) -> Tuple[Any]:
    """Retrieve the values of all futures from a combined future.

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
def get_all_future(*args: FutureResult) -> CombinedFutureResult:
    """Create a combined future that will yield a result when all input futures complete.

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
        raise TypeError("get_all_future requires at least one argument")
    return CombinedFutureResult(args, deps=None, state_num=0)
