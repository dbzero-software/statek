#!/usr/bin/env bash
set -euo pipefail

# uv does not read the pip config written by aws codeartifact login.
if [[ -z "${UV_DEFAULT_INDEX:-}" && -z "${UV_INDEX_URL:-}" ]]; then
    if [[ -n "${CODEARTIFACT_PIP_CONFIG:-}" ]]; then
        candidates=("${CODEARTIFACT_PIP_CONFIG}")
    else
        candidates=("${HOME}/.config/pip/pip.conf" "${HOME}/.pip/pip.conf" /etc/pip.conf)
    fi

    for config_file in "${candidates[@]}"; do
        if [[ -f "${config_file}" ]]; then
            index_url="$(sed -n 's/^[[:space:]]*index-url[[:space:]]*=[[:space:]]*//p' "${config_file}" | tail -n 1)"
            if [[ -n "${index_url}" ]]; then
                export UV_DEFAULT_INDEX="${index_url}"
                break
            fi
        fi
    done
fi

exec uv "$@"
