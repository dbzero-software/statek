"""Shared context data structures for Statek jobs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import dbzero as db0


@db0.memo
@dataclass
class ContextVar:
    """A value stored in shared context with usage tracking metadata."""

    category: Any
    value: Any
    description: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    use_count: int = 0


@db0.memo
@dataclass
class SharedContext:
    """Named shared context variables available to future job integrations."""

    __context_vars: Dict[str, ContextVar] = field(default_factory=dict)

    def set_var(self, category: Any, key: str, value: Any, description: str) -> None:
        """Store or replace a named context variable.

        Args:
            category: Category identifier for the context variable.
            key: Name used to retrieve the variable.
            value: Stored context value.
            description: Human-readable description of the variable.
        """
        self.__context_vars[key] = ContextVar(
            category=category,
            value=value,
            description=description,
        )

    def get_var(self, key: str) -> Optional[ContextVar]:
        """Return a stored context variable and increment its usage count.

        Args:
            key: Name of the variable to retrieve.

        Returns:
            The stored context variable, or None when the key is absent.
        """
        var = self.__context_vars.get(key)
        if var is None:
            return None
        var.use_count += 1
        return var

    def _peek_var(self, key: str) -> Optional[ContextVar]:
        """Return a stored context variable without incrementing usage."""
        return self.__context_vars.get(key)

    def __contains__(self, key: str) -> bool:
        """Return whether *key* exists in the shared context."""
        return key in self.__context_vars


ReadableContexts = Optional[Union[SharedContext, Iterable[SharedContext]]]


@db0.memo
class SharedContextProxy:
    """Merged shared-context view with one writable context."""

    def __init__(self, writable: SharedContext, readables: ReadableContexts = None):
        self.__writable = writable
        self.__readables = self.__normalize_readables(readables)

    @staticmethod
    def __normalize_readables(readables: ReadableContexts) -> Tuple[SharedContext, ...]:
        if readables is None:
            return ()
        if isinstance(readables, SharedContext):
            return (readables,)
        return tuple(readables)

    def __contexts(self) -> Tuple[SharedContext, ...]:
        return (self.__writable, *self.__readables)

    def set_var(self, category: Any, key: str, value: Any, description: str) -> None:
        """Store a variable in the writable context."""
        self.__writable.set_var(category, key, value, description)

    def get_var(self, key: str) -> Optional[ContextVar]:
        """Return the newest matching variable and increment only that variable."""
        latest = None
        for context in self.__contexts():
            var = context._peek_var(key)  # pylint: disable=protected-access
            if var is not None and (latest is None or var.created_at > latest.created_at):
                latest = var
        if latest is None:
            return None
        latest.use_count += 1
        return latest

    def __contains__(self, key: str) -> bool:
        """Return whether *key* exists in any proxied context."""
        return any(
            context._peek_var(key) is not None  # pylint: disable=protected-access
            for context in self.__contexts()
        )


__all__ = ["ContextVar", "SharedContext", "SharedContextProxy"]
