"""Statek-UI authentication helpers.

Re-exports the shared OIDC setup from the selltime project (loaded by file path
to avoid namespace collision — both projects have a top-level ``web_ui`` package).
"""

import importlib.util
import os

_SELLTIME_PATH = os.environ.get('STATEK_EXTERNAL_PATHS', '').split(':')[0].strip()
_OIDC_MODULE_PATH = os.path.join(_SELLTIME_PATH, 'web_ui', 'auth', 'nicegui_oidc.py')

_spec = importlib.util.spec_from_file_location('selltime_nicegui_oidc', _OIDC_MODULE_PATH)
_oidc_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oidc_module)

OIDCSettingsBase = _oidc_module.OIDCSettingsBase
setup_oidc_auth = _oidc_module.setup_oidc_auth
