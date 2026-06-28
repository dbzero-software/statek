# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RestrictedPython policy for LLM-authored Statek code."""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import operator
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import TYPE_CHECKING, Iterable, Iterator, Optional

from RestrictedPython import compile_restricted
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_delattr,
    guarded_iter_unpack_sequence,
    guarded_setattr,
    guarded_unpack_sequence,
    safe_builtins,
    safer_getattr,
)

if TYPE_CHECKING:
    from statek.settings import StatekSettings


class SandboxViolation(RuntimeError):
    """Raised when Python source violates the Statek sandbox policy."""


_SANDBOX_POLICY_CONFIGURED = False
_SANDBOX_POLICY: Optional["SandboxPolicy"] = None
_VALIDATED_SOURCE_CACHE_SIZE = 256
_VALIDATED_SOURCE_CACHE: "OrderedDict[tuple, ast.Module]" = OrderedDict()


_DENIED_NAMES = {
    "__import__",
    "open",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "type",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "dir",
}

_ALLOWED_BUILTIN_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "BaseException",
    "Exception",
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "LookupError",
    "NameError",
    "StopIteration",
}

_INPLACE_OPERATORS = {
    "+=": operator.add,
    "-=": operator.sub,
    "*=": operator.mul,
    "/=": operator.truediv,
    "//=": operator.floordiv,
    "%=": operator.mod,
    "**=": operator.pow,
    "<<=": operator.lshift,
    ">>=": operator.rshift,
    "&=": operator.and_,
    "^=": operator.xor,
    "|=": operator.or_,
}

_PYTHON_FENCE_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+.-]*)[^\n]*\n(?P<code>.*?)```",
    re.DOTALL,
)


def _csv_set(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _setting_int(settings: StatekSettings, name: str, default: int) -> int:
    value = getattr(settings, name, default)
    return value if isinstance(value, int) else default


def _setting_str(settings: StatekSettings, name: str, default: str) -> str:
    value = getattr(settings, name, default)
    return value if isinstance(value, str) else default


def _inplacevar(op: str, left, right):
    try:
        func = _INPLACE_OPERATORS[op]
    except KeyError as exc:
        raise SandboxViolation(f"operator {op!r} is not allowed") from exc
    return func(left, right)


def _blocked_import(name: str, allowed_imports: set[str], level: int):
    if level:
        raise SandboxViolation("relative imports are not allowed")
    root = name.split(".", 1)[0]
    if root not in allowed_imports:
        raise SandboxViolation(f"import '{root}' is not allowed")


def _is_private_name_only_restrictedpython_error(exc: SyntaxError) -> bool:
    messages = [str(arg) for arg in exc.args if arg]
    if not messages:
        return False
    return all("invalid variable name because it starts with \"_\"" in msg for msg in messages)


def _freeze_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(names)))


@dataclass(frozen=True)
class SandboxPolicy:
    """Static and runtime policy for sandboxed Python snippets."""

    max_source_bytes: int = 200_000
    max_ast_nodes: int = 20_000
    allowed_imports: set[str] = field(default_factory=lambda: {"datetime", "calendar", "statek"})
    allowed_tools: set[str] = field(default_factory=set)
    blocked_tools: set[str] = field(default_factory=set)
    _globals_template: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_settings(
        cls,
        settings: StatekSettings,
        *,
        allowed_tools: Optional[Iterable[str]] = None,
        blocked_tools: Optional[Iterable[str]] = None,
    ) -> "SandboxPolicy":
        configured_tools = _csv_set(
            _setting_str(settings, "python_sandbox_allowed_tools", "")
        )
        return cls(
            max_source_bytes=_setting_int(settings, "python_sandbox_max_source_bytes", 200_000),
            max_ast_nodes=_setting_int(settings, "python_sandbox_max_ast_nodes", 20_000),
            allowed_imports=_csv_set(
                _setting_str(settings, "python_sandbox_allowed_imports", "datetime,calendar,statek")
            ),
            allowed_tools=set(allowed_tools or set()) | configured_tools,
            blocked_tools=set(blocked_tools or set()) - configured_tools,
        )

    def cache_key(self, source: str) -> tuple:
        return (
            self.max_source_bytes,
            self.max_ast_nodes,
            _freeze_names(self.allowed_imports),
            _freeze_names(self.allowed_tools),
            _freeze_names(self.blocked_tools),
            source,
        )

    def validate_source(self, source: str) -> ast.Module:
        cache_key = self.cache_key(source)
        cached = _VALIDATED_SOURCE_CACHE.get(cache_key)
        if cached is not None:
            _VALIDATED_SOURCE_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached)

        source_bytes = len(source.encode("utf-8"))
        if source_bytes > self.max_source_bytes:
            raise SandboxViolation(
                f"source is {source_bytes} bytes; limit is {self.max_source_bytes}"
            )
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise SandboxViolation(f"invalid Python syntax: {exc}") from exc

        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.max_ast_nodes:
            raise SandboxViolation(
                f"source has {node_count} AST nodes; limit is {self.max_ast_nodes}"
            )
        _PolicyVisitor(self).visit(tree)
        self.compile_restricted(ast.parse(source), "<sandbox>", "exec")
        _VALIDATED_SOURCE_CACHE[cache_key] = copy.deepcopy(tree)
        _VALIDATED_SOURCE_CACHE.move_to_end(cache_key)
        while len(_VALIDATED_SOURCE_CACHE) > _VALIDATED_SOURCE_CACHE_SIZE:
            _VALIDATED_SOURCE_CACHE.popitem(last=False)
        return tree

    def compile_restricted(self, tree: ast.AST, filename: str, mode: str) -> CodeType:
        try:
            return compile_restricted(tree, filename, mode)
        except SyntaxError as exc:
            message = "; ".join(str(arg) for arg in exc.args if arg)
            if _is_private_name_only_restrictedpython_error(exc):
                return compile(tree, filename, mode)
            raise SandboxViolation(message or str(exc)) from exc

    def globals(self, base_globals: dict) -> dict:
        result = dict(base_globals)
        if not self._globals_template:
            self._globals_template.update(self._build_globals_template())
        result.update(self._globals_template)
        return result

    def _build_globals_template(self) -> dict:
        builtins_dict = {
            name: value
            for name, value in safe_builtins.items()
            if name in _ALLOWED_BUILTIN_NAMES and name not in _DENIED_NAMES
        }
        for name in _ALLOWED_BUILTIN_NAMES:
            if name in _DENIED_NAMES:
                continue
            if name not in builtins_dict and hasattr(builtins, name):
                builtins_dict[name] = getattr(builtins, name)
        builtins_dict["__import__"] = self._guarded_import

        return {
            "__builtins__": builtins_dict,
            "_getattr_": safer_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_write_": full_write_guard,
            "_setattr_": guarded_setattr,
            "_delattr_": guarded_delattr,
            "_inplacevar_": _inplacevar,
        }

    def _guarded_import(
        self,
        name: str,
        globals=None,  # pylint: disable=redefined-builtin,unused-argument
        locals=None,  # pylint: disable=redefined-builtin,unused-argument
        fromlist=(),  # pylint: disable=unused-argument
        level: int = 0,
    ):
        _blocked_import(name, self.allowed_imports, level)
        return importlib.import_module(name)


def _settings_sandbox_mode(settings: StatekSettings) -> str:
    mode = getattr(settings, "python_sandbox_mode", "restricted")
    return mode if isinstance(mode, str) else "restricted"


def configure_sandbox(
    settings: "StatekSettings",
    blocked_tools: Optional[Iterable[str]] = None,
) -> None:
    """Configure the process-global Python sandbox policy from settings."""
    global _SANDBOX_POLICY_CONFIGURED, _SANDBOX_POLICY  # pylint: disable=global-statement

    if _settings_sandbox_mode(settings).lower() == "off":
        _SANDBOX_POLICY = None
    else:
        _SANDBOX_POLICY = SandboxPolicy.from_settings(
            settings,
            blocked_tools=blocked_tools,
        )
    _SANDBOX_POLICY_CONFIGURED = True


def reset_sandbox_policy() -> None:
    """Clear the process-global sandbox policy so it can be rebuilt lazily."""
    global _SANDBOX_POLICY_CONFIGURED, _SANDBOX_POLICY  # pylint: disable=global-statement

    _SANDBOX_POLICY_CONFIGURED = False
    _SANDBOX_POLICY = None
    _VALIDATED_SOURCE_CACHE.clear()


def get_sandbox_policy(blocked_tools: Optional[Iterable[str]] = None) -> Optional[SandboxPolicy]:
    """Return the process-global sandbox policy, lazily configured if needed."""
    if not _SANDBOX_POLICY_CONFIGURED:
        from statek.settings import get_statek_settings  # pylint: disable=import-outside-toplevel

        configure_sandbox(get_statek_settings(), blocked_tools=blocked_tools)
    return _SANDBOX_POLICY


class _PolicyVisitor(ast.NodeVisitor):
    """Statek-specific checks that supplement RestrictedPython's transformer."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy

    def visit_Name(self, node: ast.Name):  # pylint: disable=invalid-name
        if isinstance(node.ctx, ast.Load) and node.id in _DENIED_NAMES:
            raise SandboxViolation(f"name '{node.id}' is not allowed")
        if node.id.startswith("__") and node.id.endswith("__"):
            raise SandboxViolation(f"dunder name '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):  # pylint: disable=invalid-name
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise SandboxViolation(f"dunder attribute '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):  # pylint: disable=invalid-name
        for alias in node.names:
            _blocked_import(alias.name, self.policy.allowed_imports, level=0)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):  # pylint: disable=invalid-name
        if node.module is None:
            raise SandboxViolation("relative imports are not allowed")
        _blocked_import(node.module, self.policy.allowed_imports, node.level)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # pylint: disable=invalid-name
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in _DENIED_NAMES:
                raise SandboxViolation(f"call to '{name}' is not allowed")
            if name in self.policy.blocked_tools and name not in self.policy.allowed_tools:
                raise SandboxViolation(f"tool '{name}' is not exposed to this job")
        self.generic_visit(node)


def extract_python_fenced_blocks(markdown: str) -> Iterator[tuple[int, str]]:
    """Yield executable Python fenced blocks from a Markdown document."""
    block_index = 0
    for match in _PYTHON_FENCE_RE.finditer(markdown):
        lang = match.group("lang").lower()
        if lang not in {"python", "py"}:
            continue
        yield block_index, match.group("code").strip("\n")
        block_index += 1


def validate_python_source(source: str, policy: SandboxPolicy) -> None:
    """Validate one Python source snippet with Statek's sandbox policy."""
    policy.validate_source(source)


def validate_python_files(paths: Iterable[Path], policy: SandboxPolicy) -> list[str]:
    """Validate plain Python files and return actionable diagnostics."""
    errors = []
    for path in paths:
        try:
            validate_python_source(path.read_text(encoding="utf-8"), policy)
        except SandboxViolation as exc:
            errors.append(f"{path}: {exc}")
    return errors


def validate_markdown_python_blocks(paths: Iterable[Path], policy: SandboxPolicy) -> list[str]:
    """Validate Python fenced blocks in Markdown files and return diagnostics."""
    errors = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for block_index, code in extract_python_fenced_blocks(text):
            try:
                validate_python_source(code, policy)
            except SandboxViolation as exc:
                errors.append(f"{path}: python block {block_index}: {exc}")
    return errors


def validate_selltime_agent_config(root: Path, policy: SandboxPolicy) -> list[str]:
    """Validate SellTime warmup files and Markdown Python examples."""
    warmups = sorted((root / "warmup_code").glob("*.py"))
    examples = sorted((root / "examples").glob("**/*.md"))
    return (
        validate_python_files(warmups, policy)
        + validate_markdown_python_blocks(examples, policy)
    )
