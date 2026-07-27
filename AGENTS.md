# Codex / GPT-5.4 Working Guide

This file defines repository-specific instructions for agents working in this
codebase, especially Codex-style agents and GPT-5.4.

## Development Process

### Design Document

[Statek Design Document](https://docs.google.com/document/d/1GUI872TsYcSR2xs4PQBrXcl0MkVeM_jti43SDkDeEzc/edit?usp=sharing)

### Canonical Requirements

`AI_Docs/` contains non-canonical implementation plans, summaries, review notes, and completion records. Always follow the assigned task and applicable official specifications instead. Statek's canonical specification is `../statek_design.md`; `../selltime-docs/` is the canonical specification source for Selltime work. If an AI document conflicts with a task or specification, follow the task/specification and report the conflict. Do not modify a canonical specification unless the user explicitly requests it.

### Design Review

Before implementing any non-trivial feature or change, critically review the proposed design:

- Challenge unclear, redundant, or over-engineered requirements. Confirm the problem is worth solving in the proposed way.
- Identify risks early, especially around data integrity, concurrency, performance, persistence, and maintainability.
- Surface ambiguities explicitly. If the design is underspecified, ask a focused question instead of guessing.
- Prefer simpler alternatives when they achieve the same outcome with less complexity, and explain the trade-off.
- Call out anti-patterns directly, including unnecessary abstraction, leaky coupling, hidden state, and premature optimisation.

For Codex / GPT-5.4 specifically:

- Do not jump straight into implementation because the local edit path looks obvious. Validate the design first.
- Keep reasoning concrete and code-facing. Tie concerns to specific modules, data flows, or failure modes.
- When making assumptions, keep them narrow and visible in the final summary.

Do not start implementation until the design is clear and sensible. A short clarification is cheaper than a wrong patch.

## Test-Driven Development

Unless highly infeasible, implementation must start with tests:

1. Write failing tests that define the expected behaviour.
2. Implement the minimum code to make the tests pass.
3. Refactor while keeping tests green.

For Codex / GPT-5.4:

- Prefer adding or updating the smallest test that proves the behaviour change.
- Do not weaken assertions just to make tests pass.
- If TDD is not practical, state why and still add regression coverage before finishing when possible.

## Code Quality

- Keep code DRY. Extract shared logic instead of duplicating it across executors, models, or UI handlers.
- Prefer clear naming, small focused functions, straightforward control flow, and no dead code.
- Avoid speculative abstractions. Add indirection only when it solves an actual problem already present in the code.
- Preserve surrounding style unless there is a strong reason to normalize a wider area.

For Codex / GPT-5.4:

- Read the relevant files first. Do not infer architecture from filenames alone.
- Make the smallest coherent change that fully solves the task.
- Do not revert unrelated user changes in a dirty worktree.
- Keep comments sparse and useful. Explain non-obvious intent, not mechanics.

## Execution Norms

- Prefer direct inspection of the local codebase over broad assumptions.
- When searching the repo, use fast command-line tools when available.
- Before editing, understand the nearby tests and call sites that constrain the change.
- If a task spans backend and UI, verify both sides of the interface instead of patching one side blindly.
- If you encounter unexpected local changes, work around them unless they directly block the task.

## Before Finalising

Always run both standard checks and ensure they pass before considering the work done:

```bash
./scripts/run_lint.sh
./scripts/run_tests.sh
```

If one of these cannot be run, or fails for reasons unrelated to the change, say so clearly in the final handoff.

## dbzero (db0) Reference

Full documentation: https://docs.dbzero.io

### Documentation Pages

- [Classes & @memo decorator](https://docs.dbzero.io/classes) - Persistent objects, singletons, UUIDs, type IDs
- [Collections](https://docs.dbzero.io/collections) - Persistent list, dict, set, tuple (drop-in Python replacements)
- [Tags](https://docs.dbzero.io/tags) - Labeling and discovery of objects
- [Queries](https://docs.dbzero.io/queries) - `db0.find()`, boolean logic, subqueries, filtering
- [Enums](https://docs.dbzero.io/enums) - Persistent enum types via `@db0.enum`
- [Indexes](https://docs.dbzero.io/indexes) - Sorted key-value storage, range queries
- [Relations & References](https://docs.dbzero.io/relations) - Direct refs vs indirect (tag-based) coupling
- [Transactions](https://docs.dbzero.io/transactions) - Commit, autocommit, crash safety, single-writer/multi-reader
- [Atomic Updates](https://docs.dbzero.io/atomic) - Atomic operations
- [Snapshots & Time Travel](https://docs.dbzero.io/snapshots) - Point-in-time isolation
- [Prefixes & Scoping](https://docs.dbzero.io/prefixes) - Data namespace isolation
- [Migrations](https://docs.dbzero.io/migrations) - Schema evolution
- [API Reference](https://docs.dbzero.io/api-reference) - Complete function reference

### Core Patterns

```python
# Persistent objects use @db0.memo decorator
@db0.memo(prefix=DATA_PREFIX)
@dataclass
class MyModel:
    field: str
    ix_items: db0.index = field(default_factory=db0.index)  # sorted index

# Singletons
@db0.memo(prefix=DATA_PREFIX, singleton=True)
class AppState:
    pass

# Enums (two styles)
@db0.enum(values=["VALUE1", "VALUE2"], prefix=DATA_PREFIX)
class MyEnum:
    pass
# or: MyEnum = db0.enum("MyEnum", ["VALUE1", "VALUE2"])
```

### Tags API

```python
db0.tags(obj).add("TAG")          # add single tag
db0.tags(obj).add(["A", "B"])     # add multiple tags
db0.tags(obj).remove("TAG")       # remove tag (NOTE: no .discard method)
db0.tags(obj) -= ["A", "B"]       # remove via operator
db0.tags(*objects).add("TAG")     # bulk tag multiple objects
```

### Queries

```python
db0.find(MyModel)                 # all instances of type
db0.find(MyModel, "tag")          # instances with tag (AND)
db0.find(MyModel, "A", "B")       # AND: must have both tags
db0.find(MyModel, ["A", "B"])     # OR: must have either tag
db0.find(MyModel, db0.no("X"))    # NOT: exclude tag
db0.filter(query, lambda o: ...)  # filter with Python function
```

### Indexes

```python
# Supported key types: int, Decimal, date, datetime, time, None
ix.add(key, obj)                  # add to index
ix.remove(key, obj)               # remove from index
ix.sort(query)                    # sort query results by index key
ix.sort(query, desc=True)         # descending sort
ix.select(low, high)              # range query (inclusive)
db0.find("tag", ix.select(1, 10)) # combine with other queries
```

### Query Composition

`db0.find()` can compose multiple subqueries, but sorting is a terminal operation. Once sorted, the result is evaluated and cannot be composed further.

```python
# WRONG — sort=True (default) evaluates the result; db0.find() raises RuntimeError
candidates = ix.find_active_between(d1, d2)               # evaluated iterable
db0.find(candidates, db0.no("FOR_APPROVAL"))              # RuntimeError: Invalid object iterator

# CORRECT — sort=False keeps the query composable
candidates = ix.find_active_between(d1, d2, sort=False)   # raw db0 query
db0.find(candidates, db0.no("FOR_APPROVAL"))              # works
```

Prefer pushing filters into the db0 query engine rather than filtering in Python loops. Apply `sort()` only after all filtering is done.

### Avoiding `db0.filter`

`db0.filter(lambda obj: ..., query)` scans every object in Python. It is O(n) in the collection size and bypasses the db0 index engine. Avoid it whenever a native db0 query can express the same constraint.

A common case is narrowing a full-collection query down to a known list of objects. Pass the list directly as the first `db0.find()` argument instead:

```python
# WRONG — scans all instances in Python
candidates = db0.filter(
    lambda m: m in objects_set,
    db0.find(MyModel, ["TAG_A", "TAG_B"])
)

# CORRECT — db0 resolves only the provided objects against the tag index
candidates = db0.find(objects, ["TAG_A", "TAG_B"])
```

Use `db0.filter` only when no native db0 query can express the predicate, such as a numeric threshold on a field that has no index.

### Persistence

This application relies on db0 autocommit. Do not add explicit `db0.commit()` calls or transaction handling. Changes are persisted automatically at regular intervals and on `db0.close()`.

```python
db0.uuid(obj)                     # get object's permanent UUID
```
