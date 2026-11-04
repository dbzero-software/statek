#!/bin/bash
set -e

INSTALL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --system)
            INSTALL_ARGS=(--system)
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--system]" >&2
            exit 1
            ;;
    esac
done

INSTALL_CMD=(uv pip install)
if [[ ${#INSTALL_ARGS[@]} -gt 0 ]]; then
    PYTHON_REQUEST=()
    if [[ -n ${UV_PYTHON:-} ]]; then
        PYTHON_REQUEST=("$UV_PYTHON")
    fi
    SYSTEM_PYTHON="$(uv python find --system "${PYTHON_REQUEST[@]}")"
    INSTALL_ARGS+=(--python "$SYSTEM_PYTHON")
    if ! "$SYSTEM_PYTHON" - <<'PY'
import os
import sysconfig

paths = sysconfig.get_paths()
raise SystemExit(not all(os.access(paths[key], os.W_OK) for key in (
    "purelib", "platlib", "scripts",
)))
PY
    then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "System Python is not writable: $SYSTEM_PYTHON" >&2
            echo "Run with appropriate install permissions or use a virtual environment." >&2
            exit 1
        fi
        echo "System Python requires elevated install permissions: $SYSTEM_PYTHON"
        sudo -v
        INSTALL_CMD=(sudo --preserve-env=UV_BREAK_SYSTEM_PACKAGES -- "$(command -v uv)" pip install)
    fi
fi

echo "Building distribution packages..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source and wheel distributions
uv build --sdist --wheel

echo "Build completed successfully!"
echo "Statek distribution packages are in dist/"

echo "Installing packages..."
"${INSTALL_CMD[@]}" "${INSTALL_ARGS[@]}" dist/*.whl
echo "Packages installed successfully!"
