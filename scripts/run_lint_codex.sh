#!/bin/bash
set -e

export PYTHONIOENCODING=utf8

VENV_PYTHON="/src/statek/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Expected virtualenv python not found at $VENV_PYTHON" >&2
    exit 1
fi

echo "Starting lint"
exec "$VENV_PYTHON" -m pylint "$@" statek/ tests/
