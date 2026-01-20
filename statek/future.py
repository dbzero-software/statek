from dataclasses import dataclass
from typing import Any, Set, Tuple, Callable
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
    __check_condition: Callable
    __fetch_result: Callable

    @property
    def value(self):
        """
        Retrieve result (if available) or raise FutureError
        """
        return self.__fetch_result(self)

    def check_condition(self):
        return self.__check_condition(self)
