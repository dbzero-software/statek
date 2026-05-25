"""Statek-UI authentication helpers.

Re-exports the shared OIDC setup from the selltime project (loaded by file path
to avoid namespace collision — both projects have a top-level ``web_ui`` package).
"""

import importlib.util
import os
from pathlib import Path
import sys


def _resolve_oidc_module_path() -> Path:
    """Resolve SellTime's shared NiceGUI OIDC helper without importing this package recursively."""
    current_file = Path(__file__).resolve()
    search_paths = [
        path.strip()
        for path in os.environ.get('STATEK_EXTERNAL_PATHS', '').split(':')
        if path.strip()
    ]
    search_paths.extend(path for path in sys.path if path)

    for base_path in search_paths:
        candidate = Path(base_path) / 'web_ui' / 'auth' / 'nicegui_oidc.py'
        if candidate.is_file() and candidate.resolve() != current_file:
            return candidate
    raise ImportError(
        'Could not locate SellTime web_ui.auth.nicegui_oidc; set STATEK_EXTERNAL_PATHS or PYTHONPATH'
    )


_OIDC_MODULE_PATH = _resolve_oidc_module_path()

_spec = importlib.util.spec_from_file_location('selltime_nicegui_oidc', _OIDC_MODULE_PATH)
_oidc_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oidc_module)

OIDCSettingsBase = _oidc_module.OIDCSettingsBase
setup_oidc_auth = _oidc_module.setup_oidc_auth
