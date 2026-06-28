# Python CLI Sandbox Design

## Summary

Statek executes LLM-authored Python through two paths:

- Markdown/code-block execution via `exec_step`.
- Direct-style `python_cli` tool calls via `exec_cli_step`.

Both paths currently execute in the Statek process with access to job locals, agent context,
system tools, helper functions, and Python process state. A useful sandbox must therefore
be applied before either path reaches `exec`/`eval`, and both paths must use the same policy.

The recommended first implementation is an in-process RestrictedPython policy backed by
strict capability exposure. This is not equivalent to OS isolation. It is acceptable only if
Statek exposes no raw Python callables and dbzero objects satisfy the requirements listed
below. A worker-process sandbox remains the stronger future option for high-risk deployments.

## Goals

- Prevent LLM code from importing modules or reaching modules already loaded in the parent
  process, such as `os`, `sys`, `subprocess`, or Statek internals.
- Prevent unauthorized tool access, including hidden, internal, or unrelated system tools.
- Preserve existing Python CLI behavior where possible: local state, `print`, expression
  output, `exit`, `FutureError` continuation, and `python_cli` output routing.
- Allow primitive values and dbzero-backed objects, subject to dbzero-side sandbox guarantees.

## Non-Goals

- Do not design a complete dbzero sandbox facade here. This document lists only the required
  dbzero contract.
- Do not rely on RestrictedPython as a hostile-code boundary by itself.
- Do not expose arbitrary application objects, raw Statek objects, raw dbzero internals, or
  raw Python callables to sandboxed code.
- Do not add explicit `db0.commit()` or transaction handling as part of sandboxing.

## Current Risk

The current execution setup merges broad process-owned capabilities into the execution
namespace. In particular, `_setup_execution_context` merges agent private context, registers
direct tools, adds system tools, injects framework helpers, and monkey-patches process-wide
`builtins.print` and `builtins.exit`.

That makes simple source filtering insufficient. Blocking `import os` does not help if code
can reach a bound method, inspect its `__globals__`, and recover an already-imported module.
A local probe demonstrated this shape with a dbzero memo instance under a weak
RestrictedPython policy that exposed raw `getattr` and raw item access:

```python
m = getattr(obj, "public_method")
g = getattr(m, "__globals__")
result = g["os"]
```

The strict policy must therefore block reflection and avoid exposing objects that can return
normal Python functions, bound methods, modules, frames, descriptors, classes, or private
metadata.

## Statek Sandbox Architecture

Introduce one central sandbox execution boundary used by both `exec_step` and `exec_cli_step`.
The boundary should run before Statek's existing AST transformer and before code reaches
`exec`/`eval`.

The boundary should:

- Compile and validate LLM source with RestrictedPython.
- Construct a fresh sandbox namespace per execution.
- Provide a controlled safe builtins map.
- Inject only approved runtime functions: `print`, `exit`, and framework helpers that are
  intentionally part of the sandbox API.
- Convert job locals into sandbox-safe values before exposure.
- Convert sandbox outputs back into parent-owned values after execution.
- Route all tool calls through a broker instead of exposing raw tool callables.

The same sandbox policy must apply to:

- Warmup code.
- Markdown/code-block execution.
- Direct `python_cli` code.
- Continuation after `FutureError`.

## RestrictedPython Policy

RestrictedPython should be configured as a strict policy layer. The policy should deny by
default and grant only the smallest subset needed by agent code.

Disallowed language or runtime access:

- `import` and `from ... import ...`, unless a future policy explicitly allows a pure module.
- `getattr`, `setattr`, `delattr`, `vars`, `dir`, `type`, `object`, `super`.
- `globals`, `locals`, `eval`, `exec`, `compile`, `open`, `input`, `breakpoint`,
  `__import__`.
- Private and dunder names or attributes.
- Direct class/type/object graph traversal.
- Raw descriptors, frames, code objects, function objects, bound methods, and module objects.

Required guards:

- `_getattr_`: strict guard that rejects private/dunder names and delegates only to
  sandbox-safe object access.
- `_getitem_`: guard that blocks private string keys, wraps returned values, and supports only
  approved container/query operations.
- `_write_`: guard that permits writes only to sandbox-approved mutable values and dbzero
  fields marked writable.
- `_getiter_`: guard that iterates only approved iterables and wraps yielded values.

Resource limits should be added where practical even for in-process execution:

- Maximum source length.
- Maximum AST node count.
- Maximum literal size.
- Maximum loop/step budget if a practical instrumentation approach is chosen.

These limits are defense in depth. CPU and memory denial of service are not fully solved
without process isolation.

## Namespace And Local State

The sandbox namespace must not be built from `globals()` or from the full agent context.

Allowed values:

- JSON-like primitives: `None`, `bool`, `int`, `float`, `str`.
- Approved simple containers containing only sandbox-safe values.
- dbzero objects that satisfy the dbzero requirements below.
- Tool capability wrappers created by Statek.

Rejected values:

- Raw Python functions and bound methods.
- Raw modules.
- Raw classes/types, unless represented by a sandbox-safe symbolic value.
- Raw Statek `Job`, `Agent`, context dictionaries, settings objects, adapters, and private
  helpers.
- Raw dbzero internal handles, cache/workspace objects, or native implementation objects.

After execution, Statek should persist only approved local-state deltas. Unsupported values
created inside the sandbox should be rejected or represented through explicit handles rather
than copied into `job.py_env.local_state`.

## Tool Capability Broker

Sandboxed Python must never receive raw tool callables. It should receive lightweight
capability wrappers whose only job is to ask the parent broker to run a tool.

The broker must validate:

- The requested tool name is present in the current job's allowed tool set.
- Hidden, internal, and system tools are denied unless explicitly granted by policy.
- Arguments and keyword arguments are sandbox-safe.
- Returned values are primitives, approved containers, dbzero objects that satisfy the dbzero
  requirements, or explicit opaque handles.

The broker remains responsible for current Statek semantics:

- Tool result formatting.
- Tool error formatting.
- `FutureResult` and `FutureError` handling.
- Error-handler binding.
- Tool-log ordering.
- `python_cli` output remaining in `tool_log`, not `job.py_env.console`.

This is the main authorization boundary for tool access. Prompt-level tool visibility and LLM
request tool selection are not sufficient authorization checks.

## dbzero Requirements

dbzero objects exposed to RestrictedPython must not provide paths to Python reflection,
process globals, or dbzero internals. The exact facade or implementation design belongs to
dbzero and is intentionally out of scope for this document.

Minimum requirements:

- Attribute access never exposes:
  - `__class__`, `__dict__`, `__mro__`, `__subclasses__`.
  - Function objects, bound methods, descriptors, or properties backed by raw Python callables.
  - `__globals__`, `__closure__`, frames, modules, or code objects.
  - Raw native handles, cache objects, workspace objects, prefix objects, or storage internals.
- Field reads are limited to declared persistent fields marked sandbox-readable by dbzero.
- Field writes are limited to declared persistent fields marked sandbox-writable by dbzero.
- Values returned from dbzero reads, queries, collections, indexes, tags, or enums are
  recursively sandbox-safe or rejected.
- Collection, query, and index access exposes only explicit safe operations such as length,
  iteration, indexing/slicing, sorting/range selection, and approved mutators.
- User-defined memo methods are not sandbox-safe by default. Custom behavior must be exposed
  as an explicit Statek/dbzero capability rather than as a raw bound method.
- dbzero provides a reliable way for Statek guards to recognize sandbox-safe dbzero values.
- dbzero provides tests proving that sandbox-visible objects cannot recover modules such as
  `os` through method globals, class metadata, `__dict__`, private attributes, or raw handles.

Until this contract exists, Statek should not expose raw dbzero memo objects to sandboxed code.

## Configuration

Proposed settings:

- `STATEK_PYTHON_SANDBOX_MODE=off|restricted|worker`
- `STATEK_PYTHON_SANDBOX_TIMEOUT_MS`
- `STATEK_PYTHON_SANDBOX_MAX_SOURCE_BYTES`
- `STATEK_PYTHON_SANDBOX_MAX_AST_NODES`
- `STATEK_PYTHON_SANDBOX_ALLOWED_TOOLS`

Mode meanings:

- `off`: current behavior. Suitable only for trusted local development and compatibility.
- `restricted`: RestrictedPython plus Statek capability broker plus dbzero-safe objects.
- `worker`: future stronger isolation using a separate constrained process.

Recommended defaults:

- Local development: `off` until compatibility work is complete.
- Production with trusted agents: `restricted`.
- Production with user-influenced or adversarial agents: `worker` when available.

## Worker Mode Future Hardening

Worker mode should remain in the design as the stronger boundary for deployments that need
host-level isolation. It should run untrusted code in a separate process with:

- Scrubbed environment.
- Empty temporary working directory.
- Closed inherited file descriptors.
- CPU, memory, file-size, process-count, and open-file limits.
- Parent-enforced wall-clock timeout.
- Tool access only through the parent broker.

Avoid raw `fork` memory inheritance for adversarial isolation. It can be faster because the
child sees the parent's warm memory by copy-on-write, but it also exposes the interpreter heap,
cached dbzero objects, settings, modules, and other process state to the child. Worker mode
should prefer `spawn`/`exec` with explicit serialized inputs.

## Test Scenarios

Host access is blocked:

- `import os`
- `from os import system`
- `__import__("os")`
- `open("/etc/passwd")`
- `globals()`, `locals()`, `type(obj)`, `obj.__class__`
- `eval("1 + 1")`, `exec("x = 1")`, `compile("x", "<x>", "eval")`

dbzero escape attempts are blocked:

- `obj.__dict__`
- `obj.__class__`
- `obj.public_method`
- `obj.public_method.__globals__`
- `getattr(obj, "__class__")`
- private-key traversal through item access
- access to raw native handles or workspace/cache internals

Unauthorized tools are blocked:

- Hidden tools.
- Internal tools.
- System tools not explicitly exposed.
- Tools absent from the job policy.
- Tool lookup through local variables or object graphs.

Allowed behavior still works:

- Read sandbox-readable primitive fields.
- Write sandbox-writable primitive fields.
- Iterate approved dbzero collections and query results.
- Call approved tools through the broker.
- Preserve expression output and `print` routing.
- Keep `python_cli` output in `tool_log`.
- Preserve `exit()` behavior.
- Preserve `FutureError` continuation semantics.

Resource abuse tests:

- Oversized source is rejected.
- Oversized AST is rejected.
- Huge literals are rejected.
- Infinite loop behavior is documented as not fully contained until worker mode, unless a
  step budget is implemented.

## Open Questions

- Whether `restricted` mode should be the default once dbzero satisfies the sandbox contract.
- Whether sandbox-readable and sandbox-writable dbzero fields should default to allow or deny.
- Whether Statek should allow any imports in restricted mode, even for pure modules.
- How to represent unsupported non-primitive returned values: reject, stringify, or use opaque
  handles.
- How much Python syntax to support in v1 beyond assignments, expressions, control flow,
  comprehensions, and simple function definitions.

