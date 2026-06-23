"""Shared context data structures for Statek jobs."""

# pylint: disable=no-member,protected-access

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import dbzero as db0

from .system import tool
from .utils import get_current_job


@db0.memo
@dataclass(init=False)
class ContextCategory:
    """A category accepted for variables in shared contexts."""

    def __init__(self, name: str, prefix=None):
        db0.set_prefix(self, prefix)
        self.name = name


def _default_context_categories(prefix=None) -> Dict[str, ContextCategory]:
    """Create the initial canonical category mapping."""
    return {
        name: ContextCategory(name=name, prefix=prefix)
        for name in ("PREFERENCE", "ENTITY", "VOCABULARY")
    }


@db0.memo(singleton=True)
class ContextCategoryDict:
    """Process-wide registry of accepted shared-context categories."""

    def __init__(
        self,
        prefix=None,
        categories: Optional[Dict[str, ContextCategory]] = None,
    ):
        db0.set_prefix(self, prefix)
        self.categories = (
            categories if categories is not None
            else _default_context_categories(prefix=prefix)
        )

    def get(self, name: str) -> Optional[ContextCategory]:
        """Return a registered category, accepting names case-insensitively."""
        if not isinstance(name, str):
            return None
        return self.categories.get(name) or self.categories.get(name.upper())


@db0.memo
@dataclass
class ContextVar:
    """A value stored in shared context with usage tracking metadata."""

    category: Any
    value: Any
    description: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    use_count: int = 0


def _category_name(category: Any) -> Optional[str]:
    """Return a normalized category name when one can be resolved."""
    name = category.name if isinstance(category, ContextCategory) else category
    return name.upper() if isinstance(name, str) else None


@db0.memo
@dataclass
class SharedContext:
    """Named shared context variables available to future job integrations."""

    __context_vars: Dict[str, ContextVar] = field(default_factory=dict)
    _binding_keys: Tuple[str, ...] = field(default_factory=tuple)

    def set_var(self, category: Any, key: str, value: Any, description: str) -> None:
        """Store or replace a named context variable.

        Args:
            category: Category identifier for the context variable.
            key: Name used to retrieve the variable.
            value: Stored context value.
            description: Human-readable description of the variable.
        """
        category_name = (
            category.name if isinstance(category, ContextCategory) else category
        )
        if ContextCategoryDict().get(category_name) is None:
            return
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

    def _items(self) -> Tuple[Tuple[str, ContextVar], ...]:
        """Return named variables without changing usage counters."""
        return tuple(self.__context_vars.items())

    def select(
        self,
        category: ContextCategory,
        filter: Optional[Callable[[ContextVar], bool]] = None,  # pylint: disable=redefined-builtin
    ) -> Iterable[ContextVar]:
        """Iterate variables in *category* that satisfy an optional predicate."""
        for _, var in self._select_items(category, filter):
            yield var

    def _select_items(
        self,
        category: ContextCategory,
        filter: Optional[Callable[[ContextVar], bool]] = None,  # pylint: disable=redefined-builtin
    ) -> Iterable[Tuple[str, ContextVar]]:
        """Iterate named variables selected by category and predicate."""
        category_name = _category_name(category)
        for key, var in self.__context_vars.items():
            if _category_name(var.category) != category_name:
                continue
            if filter is not None and not filter(var):
                continue
            var.use_count += 1
            yield key, var

    def __contains__(self, key: str) -> bool:
        """Return whether *key* exists in the shared context."""
        return key in self.__context_vars


ReadableContexts = Optional[Union[SharedContext, Iterable[SharedContext]]]


@db0.memo
class SharedContextProxy:
    """Merged shared-context view with one writable context."""

    def __init__(self, writable: Optional[SharedContext],
                 readables: ReadableContexts = None, read_only: bool = False):
        self.__writable = writable
        self.__readables = self.__normalize_readables(readables)
        self.__read_only = read_only

    @staticmethod
    def __normalize_readables(readables: ReadableContexts) -> Tuple[SharedContext, ...]:
        if readables is None:
            return ()
        if isinstance(readables, SharedContext):
            return (readables,)
        return tuple(readables)

    def __contexts(self) -> Tuple[SharedContext, ...]:
        if self.__writable is None:
            return self.__readables
        return (self.__writable, *self.__readables)

    def set_var(self, category: Any, key: str, value: Any, description: str) -> None:
        """Store a variable in the writable context."""
        if self.__read_only or self.__writable is None:
            raise PermissionError("Shared context is read-only")
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

    def select(
        self,
        category: ContextCategory,
        filter: Optional[Callable[[ContextVar], bool]] = None,  # pylint: disable=redefined-builtin
    ) -> Iterable[ContextVar]:
        """Iterate newest resolvable variables in the requested category."""
        for _, var in self._select_items(category, filter):
            yield var

    def _select_items(
        self,
        category: ContextCategory,
        filter: Optional[Callable[[ContextVar], bool]] = None,  # pylint: disable=redefined-builtin
    ) -> Iterable[Tuple[str, ContextVar]]:
        """Iterate newest named variables selected by category and predicate."""
        latest_by_key: Dict[str, ContextVar] = {}
        for context in self.__contexts():
            for key, var in context._items():
                latest = latest_by_key.get(key)
                if latest is None or var.created_at > latest.created_at:
                    latest_by_key[key] = var

        category_name = _category_name(category)
        for key, var in latest_by_key.items():
            if _category_name(var.category) != category_name:
                continue
            if filter is not None and not filter(var):
                continue
            var.use_count += 1
            yield key, var


def _binding_key(specifier: Any) -> str:
    """Return a stable key for a supported context specifier."""
    if isinstance(specifier, str):
        return f"string:{specifier}"
    if db0.is_memo(specifier):
        return f"memo:{db0.uuid(specifier)}"
    raise TypeError("Context specifier must be a string or dbzero memo object")


def _binding_keys(specifiers: Iterable[Any]) -> Tuple[str, ...]:
    """Normalize context specifiers as an order-independent identity."""
    return tuple(sorted({_binding_key(specifier) for specifier in specifiers}))


def _find_bound_contexts(keys: Tuple[str, ...]) -> Tuple[SharedContext, ...]:
    """Return contexts whose bindings are subsets of *keys*."""
    requested = set(keys)
    return tuple(
        context for context in db0.find(SharedContext)
        if set(context._binding_keys).issubset(requested)
    )


@tool(system=True, hidden=True)
def init_shared_context(*args, read_only: bool = False, **kwargs) -> None:
    """Bind the current job to shared contexts selected by *args*.

    Positional arguments must be strings or dbzero memo objects. Broader
    contexts are merged for reads, while writes target the exact binding.

    Args:
        read_only: When True, reject writes through the active context.
    """
    del kwargs  # Framework-managed execution context arguments.
    job = get_current_job()
    if job is None:
        raise RuntimeError("init_shared_context requires a current job")

    keys = _binding_keys(args)
    contexts = _find_bound_contexts(keys)
    exact = next(
        (context for context in contexts if context._binding_keys == keys),
        None,
    )
    if exact is None:
        exact = SharedContext(_binding_keys=keys)
        contexts = (*contexts, exact)

    readables = tuple(context for context in contexts if context is not exact)
    if read_only:
        active_context = SharedContextProxy(
            writable=None,
            readables=(exact, *readables),
            read_only=True,
        )
    elif readables:
        active_context = SharedContextProxy(writable=exact, readables=readables)
    else:
        active_context = exact
    job._set_shared_context(active_context)  # pylint: disable=protected-access


@tool(system=True)
def shared_context_set_var(
    category: ContextCategory,
    key: str,
    value: Any,
    description: str,
    **kwargs,
) -> None:
    """Store a variable in the current job's initialized shared context.

    Args:
        category: Registered category for the context variable.
        key: Name used to retrieve the variable.
        value: Value to store.
        description: Human-readable description of the variable.
    """
    del kwargs  # Framework-managed execution context arguments.
    job = get_current_job()
    if job is None:
        return
    context = job._get_shared_context()  # pylint: disable=protected-access
    if context is not None:
        context.set_var(category, key, value, description)


@tool(system=True)
def print_locals(category: str, *args, **kwargs) -> None:
    """Print shared-context variables, optionally filtered by value type.

    Args:
        category: Registered context-category name.
        args: Optional types; values matching any requested type are included.
    """
    del kwargs  # Framework-managed execution context arguments.
    resolved_category = ContextCategoryDict().get(category)
    if resolved_category is None:
        raise ValueError(f"Unknown context category: {category}")
    if any(not isinstance(value_type, type) for value_type in args):
        raise TypeError("print_locals positional filters must be types")

    header = f"Locals from the category: {resolved_category.name}"
    if args:
        type_names = ", ".join(value_type.__name__ for value_type in args)
        header += f", type: {type_names}"
    print(header)

    job = get_current_job()
    if job is None:
        return
    context = job._get_shared_context()
    if context is None:
        return
    predicate = (lambda var: isinstance(var.value, args)) if args else None
    for name, var in context._select_items(resolved_category, predicate):
        suffix = f": {var.description}" if var.description else ""
        print(f"{name}  # {resolved_category.name}{suffix}")


def _context_var_has_registered_category(var: ContextVar) -> bool:
    """Return whether *var* belongs to a currently registered category."""
    name = _category_name(var.category)
    return ContextCategoryDict().get(name) is not None


class ContextFallbackProxy(dict):
    """Local namespace with lazy shared-context fallback for missing names."""

    _NOT_RESOLVED = object()

    def __init__(self, local_context: Dict[str, Any], shared_context=None):
        super().__init__(local_context)
        self.__shared_context = shared_context
        self.__resolved: Dict[str, Optional[ContextVar]] = {}

    def __resolve(self, key: str) -> Optional[ContextVar]:
        cached = self.__resolved.get(key, self._NOT_RESOLVED)
        if cached is not self._NOT_RESOLVED:
            return cached
        var = None
        if self.__shared_context is not None:
            candidate = self.__shared_context.get_var(key)
            if candidate is not None and _context_var_has_registered_category(candidate):
                var = candidate
        self.__resolved[key] = var
        return var

    def __missing__(self, key: str) -> Any:
        var = self.__resolve(key)
        if var is None:
            raise KeyError(key)
        return var.value

    def __setitem__(self, key: str, value: Any) -> None:
        self.ensure_writable(key)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs) -> None:
        """Add locals after validating every key against shared context."""
        incoming = dict(*args, **kwargs)
        for key in incoming:
            self.ensure_writable(key)
        super().update(incoming)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set a default only when it cannot shadow a shared variable."""
        if key not in self:
            self.ensure_writable(key)
        return super().setdefault(key, default)

    def __ior__(self, other):
        self.update(other)
        return self

    def ensure_writable(self, key: str) -> None:
        """Raise when *key* resolves to a protected shared variable."""
        if key not in self and self.__resolve(key) is not None:
            raise PermissionError(
                f"Shared context variable is read-only: {key}. "
                "Use shared_context_set_var to update shared context."
            )


__all__ = [
    "ContextCategory",
    "ContextCategoryDict",
    "ContextVar",
    "SharedContext",
    "ContextFallbackProxy",
    "SharedContextProxy",
    "init_shared_context",
    "print_locals",
    "shared_context_set_var",
]
