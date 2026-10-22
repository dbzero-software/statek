#!/bin/bash
set -e
export PYTHONIOENCODING=utf8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
exec "${SCRIPT_DIR}/uv_run.sh" python -m pytest -m 'not integration_test' -m 'not stress_test' --capture=no "$@" -vv
