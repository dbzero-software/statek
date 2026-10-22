#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Running linters..."

# Run pylint for code quality checks
echo "Running pylint..."
"${SCRIPT_DIR}/uv_run.sh" python -m pylint statek/ tests/

echo "All lint checks passed!"
