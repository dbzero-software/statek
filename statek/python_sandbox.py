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
import calendar
import collections
import copy
import datetime
import decimal
import fractions
import functools
import itertools
import json
import math
import operator
import re
import statistics
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

DEFAULT_ALLOWED_IMPORTS = (
    "datetime,calendar,re,math,decimal,fractions,statistics,collections,"
    "itertools,functools,operator,json"
)

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
    if name != root:
        raise SandboxViolation(f"import '{name}' is not available in the sandbox")


def _blocked_import_name(module_name: str, imported_name: str, allowed_imports: set[str]) -> None:
    _blocked_import(module_name, allowed_imports, level=0)
    if imported_name.startswith("_"):
        raise SandboxViolation(f"import '{imported_name}' is not allowed")
    try:
        getattr(_SAFE_IMPORTS[module_name], imported_name)
    except KeyError as exc:
        raise SandboxViolation(
            f"import '{module_name}' is not available in the sandbox"
        ) from exc
    except SandboxViolation as exc:
        raise SandboxViolation(f"import '{imported_name}' is not allowed") from exc


class _SandboxModule:
    """Minimal attribute wrapper for modules exposed to sandboxed code."""

    __slots__ = ("_name", "_exports")

    def __init__(self, name: str, exports: dict[str, object]):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_exports", dict(exports))

    def __getattribute__(self, name: str):
        if name == "__class__":
            return type(self)
        if name in {"_name", "_exports"}:
            return object.__getattribute__(self, name)
        if name.startswith("_") or (name.startswith("__") and name.endswith("__")):
            module_name = object.__getattribute__(self, "_name")
            raise SandboxViolation(f"attribute '{module_name}.{name}' is not allowed")
        exports = object.__getattribute__(self, "_exports")
        try:
            return exports[name]
        except KeyError as exc:
            module_name = object.__getattribute__(self, "_name")
            raise SandboxViolation(f"attribute '{module_name}.{name}' is not allowed") from exc

    def __setattr__(self, name: str, value: object) -> None:
        raise SandboxViolation("sandbox module attributes are read-only")

    def __delattr__(self, name: str) -> None:
        raise SandboxViolation("sandbox module attributes are read-only")

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_name")
        return f"<sandbox module {name}>"


_SAFE_IMPORTS = {
    "datetime": _SandboxModule(
        "datetime",
        {
            "date": datetime.date,
            "datetime": datetime.datetime,
            "time": datetime.time,
            "timedelta": datetime.timedelta,
            "timezone": datetime.timezone,
        },
    ),
    "calendar": _SandboxModule(
        "calendar",
        {
            "monthrange": calendar.monthrange,
            "weekday": calendar.weekday,
            "isleap": calendar.isleap,
            "day_name": calendar.day_name,
            "day_abbr": calendar.day_abbr,
            "month_name": calendar.month_name,
            "month_abbr": calendar.month_abbr,
        },
    ),
    "re": _SandboxModule(
        "re",
        {
            "compile": re.compile,
            "search": re.search,
            "match": re.match,
            "fullmatch": re.fullmatch,
            "findall": re.findall,
            "finditer": re.finditer,
            "sub": re.sub,
            "subn": re.subn,
            "split": re.split,
            "escape": re.escape,
            "ASCII": re.ASCII,
            "A": re.A,
            "IGNORECASE": re.IGNORECASE,
            "I": re.I,
            "LOCALE": re.LOCALE,
            "L": re.L,
            "MULTILINE": re.MULTILINE,
            "M": re.M,
            "DOTALL": re.DOTALL,
            "S": re.S,
            "VERBOSE": re.VERBOSE,
            "X": re.X,
            "NOFLAG": re.NOFLAG,
        },
    ),
    "math": _SandboxModule(
        "math",
        {
            "acos": math.acos,
            "acosh": math.acosh,
            "asin": math.asin,
            "asinh": math.asinh,
            "atan": math.atan,
            "atan2": math.atan2,
            "atanh": math.atanh,
            "ceil": math.ceil,
            "comb": math.comb,
            "copysign": math.copysign,
            "cos": math.cos,
            "cosh": math.cosh,
            "degrees": math.degrees,
            "dist": math.dist,
            "erf": math.erf,
            "erfc": math.erfc,
            "exp": math.exp,
            "expm1": math.expm1,
            "fabs": math.fabs,
            "factorial": math.factorial,
            "floor": math.floor,
            "fmod": math.fmod,
            "frexp": math.frexp,
            "fsum": math.fsum,
            "gamma": math.gamma,
            "gcd": math.gcd,
            "hypot": math.hypot,
            "isclose": math.isclose,
            "isfinite": math.isfinite,
            "isinf": math.isinf,
            "isnan": math.isnan,
            "isqrt": math.isqrt,
            "lcm": math.lcm,
            "ldexp": math.ldexp,
            "lgamma": math.lgamma,
            "log": math.log,
            "log10": math.log10,
            "log1p": math.log1p,
            "log2": math.log2,
            "modf": math.modf,
            "nextafter": math.nextafter,
            "perm": math.perm,
            "pow": math.pow,
            "prod": math.prod,
            "radians": math.radians,
            "remainder": math.remainder,
            "sin": math.sin,
            "sinh": math.sinh,
            "sqrt": math.sqrt,
            "tan": math.tan,
            "tanh": math.tanh,
            "trunc": math.trunc,
            "ulp": math.ulp,
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "inf": math.inf,
            "nan": math.nan,
        },
    ),
    "decimal": _SandboxModule(
        "decimal",
        {
            "Decimal": decimal.Decimal,
            "ROUND_CEILING": decimal.ROUND_CEILING,
            "ROUND_FLOOR": decimal.ROUND_FLOOR,
            "ROUND_UP": decimal.ROUND_UP,
            "ROUND_DOWN": decimal.ROUND_DOWN,
            "ROUND_HALF_UP": decimal.ROUND_HALF_UP,
            "ROUND_HALF_DOWN": decimal.ROUND_HALF_DOWN,
            "ROUND_HALF_EVEN": decimal.ROUND_HALF_EVEN,
            "ROUND_05UP": decimal.ROUND_05UP,
        },
    ),
    "fractions": _SandboxModule("fractions", {"Fraction": fractions.Fraction}),
    "statistics": _SandboxModule(
        "statistics",
        {
            "StatisticsError": statistics.StatisticsError,
            "mean": statistics.mean,
            "fmean": statistics.fmean,
            "geometric_mean": statistics.geometric_mean,
            "harmonic_mean": statistics.harmonic_mean,
            "median": statistics.median,
            "median_low": statistics.median_low,
            "median_high": statistics.median_high,
            "median_grouped": statistics.median_grouped,
            "mode": statistics.mode,
            "multimode": statistics.multimode,
            "pstdev": statistics.pstdev,
            "pvariance": statistics.pvariance,
            "stdev": statistics.stdev,
            "variance": statistics.variance,
            "quantiles": statistics.quantiles,
            "correlation": statistics.correlation,
            "covariance": statistics.covariance,
            "linear_regression": statistics.linear_regression,
        },
    ),
    "collections": _SandboxModule(
        "collections",
        {
            "Counter": collections.Counter,
            "deque": collections.deque,
            "defaultdict": collections.defaultdict,
            "OrderedDict": collections.OrderedDict,
        },
    ),
    "itertools": _SandboxModule(
        "itertools",
        {
            "accumulate": itertools.accumulate,
            "chain": itertools.chain,
            "combinations": itertools.combinations,
            "combinations_with_replacement": itertools.combinations_with_replacement,
            "compress": itertools.compress,
            "count": itertools.count,
            "cycle": itertools.cycle,
            "dropwhile": itertools.dropwhile,
            "filterfalse": itertools.filterfalse,
            "groupby": itertools.groupby,
            "islice": itertools.islice,
            "pairwise": itertools.pairwise,
            "permutations": itertools.permutations,
            "product": itertools.product,
            "repeat": itertools.repeat,
            "starmap": itertools.starmap,
            "takewhile": itertools.takewhile,
            "tee": itertools.tee,
            "zip_longest": itertools.zip_longest,
        },
    ),
    "functools": _SandboxModule("functools", {"reduce": functools.reduce}),
    "operator": _SandboxModule(
        "operator",
        {
            "abs": operator.abs,
            "add": operator.add,
            "and_": operator.and_,
            "concat": operator.concat,
            "contains": operator.contains,
            "countOf": operator.countOf,
            "eq": operator.eq,
            "floordiv": operator.floordiv,
            "ge": operator.ge,
            "gt": operator.gt,
            "index": operator.index,
            "indexOf": operator.indexOf,
            "inv": operator.inv,
            "invert": operator.invert,
            "is_": operator.is_,
            "is_not": operator.is_not,
            "le": operator.le,
            "length_hint": operator.length_hint,
            "lshift": operator.lshift,
            "lt": operator.lt,
            "matmul": operator.matmul,
            "mod": operator.mod,
            "mul": operator.mul,
            "ne": operator.ne,
            "neg": operator.neg,
            "not_": operator.not_,
            "or_": operator.or_,
            "pos": operator.pos,
            "pow": operator.pow,
            "rshift": operator.rshift,
            "sub": operator.sub,
            "truediv": operator.truediv,
            "truth": operator.truth,
            "xor": operator.xor,
        },
    ),
    "json": _SandboxModule("json", {"loads": json.loads, "dumps": json.dumps}),
}


def _is_safe_export(value: object) -> bool:
    return any(
        value is exported
        for module in _SAFE_IMPORTS.values()
        for exported in object.__getattribute__(module, "_exports").values()
    )


def _safe_import(name: str, allowed_imports: set[str], level: int):
    _blocked_import(name, allowed_imports, level)
    root = name.split(".", 1)[0]
    try:
        return _SAFE_IMPORTS[root]
    except KeyError as exc:
        raise SandboxViolation(f"import '{root}' is not available in the sandbox") from exc


def is_sandbox_transient_value(value: object) -> bool:
    """Return whether a runtime value should not be persisted after sandbox execution."""
    return isinstance(value, _SandboxModule) or _is_safe_export(value)


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
    allowed_imports: set[str] = field(default_factory=lambda: _csv_set(DEFAULT_ALLOWED_IMPORTS))
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
                _setting_str(settings, "python_sandbox_allowed_imports", DEFAULT_ALLOWED_IMPORTS)
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
        return _safe_import(name, self.allowed_imports, level)


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
        for alias in node.names:
            _blocked_import_name(node.module, alias.name, self.policy.allowed_imports)
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
