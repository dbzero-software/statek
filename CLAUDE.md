# Development Process

## Test-Driven Development

Unless highly infeasible, implementation must start with writing tests first (TDD):

1. Write failing tests that define the expected behaviour.
2. Implement the minimum code to make the tests pass.
3. Refactor while keeping tests green.

## Code Quality

- **DRY** — avoid duplication; extract shared logic into reusable components.
- **Clean code** — clear naming, small focused functions, no dead code.

## Before Finalising

Always run both checks and ensure they pass before considering the work done:

```bash
./scripts/run_lint.sh
./scripts/run_tests.sh
```

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
db0.find(MyModel, "A", "B")      # AND: must have both tags
db0.find(MyModel, ["A", "B"])     # OR: must have either tag
db0.find(MyModel, db0.no("X"))   # NOT: exclude tag
db0.filter(query, lambda o: ...) # filter with Python function
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

`db0.find()` can compose multiple subqueries, but **sorting is a terminal operation** — once a query is sorted, it becomes an evaluated sequence that cannot be further composed.

```python
# WRONG — sort=True (default) evaluates the result; db0.find() raises RuntimeError
candidates = ix.find_active_between(d1, d2)          # evaluated iterable
db0.find(candidates, db0.no("FOR_APPROVAL"))         # RuntimeError: Invalid object iterator

# CORRECT — sort=False keeps the query composable
candidates = ix.find_active_between(d1, d2, sort=False)  # raw db0 query
db0.find(candidates, db0.no("FOR_APPROVAL"))             # works — filters at engine level
```

Prefer pushing tag filters into the db0 query engine rather than filtering in Python loops when possible. Only apply `sort()` as the final step after all filtering is done.

### Avoiding `db0.filter`

`db0.filter(lambda obj: ..., query)` scans every object in `query` in Python — it is **O(n)** in the collection size and bypasses the db0 index engine. Avoid it whenever a native db0 query can express the same constraint.

A common case is narrowing a full-collection query down to a known list of objects. Pass the list directly as `db0.find()`'s first argument instead:

```python
# WRONG — scans all instances in Python
candidates = db0.filter(
    lambda m: m in objects_set,
    db0.find(MyModel, ["TAG_A", "TAG_B"])
)

# CORRECT — db0 engine resolves only the provided objects against the tag index
candidates = db0.find(objects, ["TAG_A", "TAG_B"])
```

`db0.filter` is acceptable only when no native query can express the predicate (e.g. a numeric threshold on a field that has no index).

### Persistence

This application relies on db0's **autocommit** — do not add explicit `db0.commit()` calls or transaction handling. Changes are persisted automatically at regular intervals and on `db0.close()`.

```python
db0.uuid(obj)                     # get object's permanent UUID
```
