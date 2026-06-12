"""Tests for Statek UI auth helper exports."""

import inspect

from StatekWebUI import auth
from StatekWebUI.auth import nicegui_oidc


def test_statek_auth_exports_local_oidc_helpers():
    """The auth package must use Statek-owned helpers, not project-specific modules."""
    assert auth.OIDCSettingsBase is nicegui_oidc.OIDCSettingsBase
    assert auth.setup_oidc_auth is nicegui_oidc.setup_oidc_auth
    assert nicegui_oidc.__name__ == "StatekWebUI.auth.nicegui_oidc"


def test_statek_auth_module_has_no_selltime_coupling():
    """Statek's OIDC helper must not import or load Selltime internals."""
    source = inspect.getsource(nicegui_oidc)

    assert "selltime" not in source.lower()
