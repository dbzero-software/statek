#!/bin/bash
set -e

INSTALL_ARGS=()
build_only=false
output=""
python="${PYTHON:-python3}"
python_args=()
while (( $# )); do
    case "$1" in
        --system) INSTALL_ARGS=(--system); shift ;;
        --build-only) build_only=true; shift ;;
        --path|--python)
            if (( $# < 2 )) || [[ -z "$2" ]]; then echo "$1 requires a value" >&2; exit 2; fi
            if [[ "$1" == --path ]]; then output="$2"; else python="$2"; python_args=(--python "$2"); fi
            shift 2 ;;
        *) echo "Usage: $0 [--system] [--build-only] [--path DIR] [--python EXECUTABLE]" >&2; exit 2 ;;
    esac
done
python="$("$python" -c 'import os, sys; print(os.path.abspath(sys.executable))')"
if [[ -n "$output" ]]; then
    output="$("$python" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$output")"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# A separate branch preserves the legacy build-and-install workflow.
if $build_only || [[ -n "$output" ]]; then
    output="${output:-$PWD/dist}"
    uv build --wheel --python "$python" --out-dir "$output"
    if ! $build_only; then uv pip install --python "$python" "${INSTALL_ARGS[@]}" "$output"/*.whl; fi
    exit 0
fi

INSTALL_CMD=(uv pip install)
if [[ ${#INSTALL_ARGS[@]} -gt 0 ]]; then
    PYTHON_REQUEST=()
    if (( ${#python_args[@]} )); then
        PYTHON_REQUEST=("$python")
    elif [[ -n ${UV_PYTHON:-} ]]; then
        PYTHON_REQUEST=("$UV_PYTHON")
    fi
    SYSTEM_PYTHON="$(uv python find --system "${PYTHON_REQUEST[@]}")"
    python_args=(--python "$SYSTEM_PYTHON")
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
uv build --sdist --wheel "${python_args[@]}"

echo "Build completed successfully!"
echo "Statek distribution packages are in dist/"

echo "Installing packages..."
"${INSTALL_CMD[@]}" "${python_args[@]}" "${INSTALL_ARGS[@]}" dist/*.whl
echo "Packages installed successfully!"
