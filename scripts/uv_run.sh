#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${SCRIPT_DIR}/uv_with_codeartifact.sh" run --project "${REPO_ROOT}" --no-sync "$@"
