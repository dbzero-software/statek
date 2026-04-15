#!/bin/bash
set -e

export PYTHONIOENCODING=utf8

VENV_PYTHON="/src/statek/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Expected virtualenv python not found at $VENV_PYTHON" >&2
    exit 1
fi

TEST_FILES_DIR="/src/statek/__test_files"
if [ -d "$TEST_FILES_DIR" ]; then
    rm -rf "$TEST_FILES_DIR"
fi

exec "$VENV_PYTHON" -m pytest "$@"
