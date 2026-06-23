"""Shared context data structures for Statek jobs."""

# pylint: disable=no-member,protected-access

import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import dbzero as db0

from .system import tool
from .utils import get_current_job


@db0.memo
@dataclass
class ContextCategory:
    """A category accepted for variables in shared contexts."""

    name: str


def _default_context_categories() -> Dict[str, ContextCategory]:
    """Create the initial canonical category mapping."""
    return {
        name: ContextCategory(name=name)
        for name in ("PREFERENCE", "ENTITY", "VOCABULARY")
    }


@db0.memo(singleton=True)
@dataclass
class ContextCategoryDict:
    """Process-wide registry of accepted shared-context categories."""

    categories: Dict[str, ContextCategory] = field(
        default_factory=_default_context_categories
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


def _referenced_names(code: str) -> Tuple[str, ...]:
    """Return names loaded, assigned, or deleted by an execution step."""
    tree = ast.parse(code)
    return tuple({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})


def _context_var_has_registered_category(var: ContextVar) -> bool:
    """Return whether *var* belongs to a currently registered category."""
    category = var.category
    name = category.name if isinstance(category, ContextCategory) else category
    return ContextCategoryDict().get(name) is not None


def _changed_from_injected(original: Any, current: Any) -> bool:
    """Return whether an injected value was reassigned during execution."""
    if original is current:
        return False
    try:
        return original != current
    except Exception:  # pylint: disable=broad-except
        return True


@contextmanager
def _feed_shared_context(
    code: str,
    job: Any,
    local_context: Dict[str, Any],
    global_context: Optional[Dict[str, Any]] = None,
):
    """Temporarily expose referenced shared variables to an execution step."""
    context = job._get_shared_context()
    if context is None:
        yield
        return

    injected = {}
    for name in _referenced_names(code):
        if name in local_context:
            continue
        var = context.get_var(name)
        if var is not None and _context_var_has_registered_category(var):
            injected[name] = var.value
            local_context[name] = var.value

    missing = object()
    global_originals = {
        name: global_context.get(name, missing) for name in injected
    } if global_context is not None else {}

    def cleanup() -> None:
        for name in injected:
            local_context.pop(name, None)
        if global_context is not None:
            for name, original in global_originals.items():
                if original is missing:
                    global_context.pop(name, None)
                else:
                    global_context[name] = original

    try:
        yield
    except BaseException:
        cleanup()
        raise
    changed = [
        name for name, original in injected.items()
        if name not in local_context
        or _changed_from_injected(original, local_context[name])
    ]
    cleanup()
    if changed:
        names = ", ".join(sorted(changed))
        raise PermissionError(
            f"Shared context variable(s) are read-only: {names}. "
            "Use shared_context_set_var to update shared context."
        )


__all__ = [
    "ContextCategory",
    "ContextCategoryDict",
    "ContextVar",
    "SharedContext",
    "SharedContextProxy",
    "init_shared_context",
    "shared_context_set_var",
]
