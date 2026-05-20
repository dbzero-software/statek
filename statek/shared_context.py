"""Shared context data structures for Statek jobs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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

    def __contains__(self, key: str) -> bool:
        """Return whether *key* exists in the shared context."""
        return key in self.__context_vars


__all__ = ["ContextVar", "SharedContext"]
