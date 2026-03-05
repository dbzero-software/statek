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
