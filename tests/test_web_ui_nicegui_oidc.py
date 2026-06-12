"""Tests for Statek NiceGUI OIDC helpers."""

import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
import jwt as pyjwt
from starlette.middleware.sessions import SessionMiddleware

from StatekWebUI.auth.nicegui_oidc import (
    OIDCSettingsBase,
    setup_oidc_auth,
)

_DEFAULT_TOKEN = object()


class _FakeOIDCConfig:
    """Record OIDC configuration passed by setup_oidc_auth."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeSessionHandler:
    """Minimal shelve session handler replacement for auth setup tests."""

    def __init__(self, mode: str, filename: str) -> None:
        self.mode = mode
        self.filename = filename
        self.reset_count = 0

    def reset_keys(self) -> None:
        self.reset_count += 1


class _FakeNiceGUIOIDClient:
    """Minimal NiceGUIOIDClient replacement used by middleware tests."""

    current_token: dict[str, str] | None = {"access_token": "rpc-token"}

    def __init__(
        self,
        nicegui_app: FastAPI,
        auth_config: _FakeOIDCConfig,
        session_storage: _FakeSessionHandler,
    ) -> None:
        self.nicegui_app = nicegui_app
        self.auth_config = auth_config
        self.session_storage = session_storage
        self.logout_count = 0

    def _logout(self) -> None:
        self.logout_count += 1

    def _get_current_token(self) -> dict[str, str] | None:
        return self.current_token

    def is_authenticated(self) -> bool:
        return self.current_token is not None


def _id_token(groups: list[str]) -> str:
    return pyjwt.encode({"cognito:groups": groups}, key="", algorithm="none")


def _install_fake_easyoidc(monkeypatch, current_token: object = _DEFAULT_TOKEN) -> None:
    if current_token is _DEFAULT_TOKEN:
        current_token = {"access_token": "rpc-token"}
    _FakeNiceGUIOIDClient.current_token = current_token

    easyoidc_module = types.ModuleType("EasyOIDC")
    easyoidc_module.Config = _FakeOIDCConfig
    easyoidc_module.SessionHandler = _FakeSessionHandler

    frameworks_module = types.ModuleType("EasyOIDC.frameworks")
    nicegui_module = types.ModuleType("EasyOIDC.frameworks.nicegui")
    nicegui_module.NiceGUIOIDClient = _FakeNiceGUIOIDClient

    monkeypatch.setitem(sys.modules, "EasyOIDC", easyoidc_module)
    monkeypatch.setitem(sys.modules, "EasyOIDC.frameworks", frameworks_module)
    monkeypatch.setitem(sys.modules, "EasyOIDC.frameworks.nicegui", nicegui_module)


def test_oauth_callback_error_page_uses_statek_branding(monkeypatch):
    """OIDC callback failures render a Statek-owned error screen."""
    _install_fake_easyoidc(monkeypatch)

    app = FastAPI()
    setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(cookie_secret_key="test-secret"),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    client = TestClient(app)
    response = client.get("/auth/callback?error=access_denied&error_description=No+access")

    assert response.status_code == 400
    assert "Statek" in response.text
    assert "#5c6bc0" in response.text
    assert "#26a69a" in response.text
    assert "SellTime" not in response.text
    assert "selltime" not in response.text.lower()


def test_default_oidc_scope_matches_cognito_local_config():
    """Statek should not request provider scopes that the local Cognito client lacks."""
    assert OIDCSettingsBase().oidc_scope == "openid email"


def test_quoted_oidc_scope_is_normalized_for_shell_loaded_env(monkeypatch):
    """Shell-loaded env files keep quotes that must not reach the provider request."""
    _install_fake_easyoidc(monkeypatch)

    app = FastAPI()
    auth = setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(
            oidc_scope='"openid email"',
            cookie_secret_key="test-secret",
        ),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )

    assert getattr(auth, "auth_config").kwargs["scope"] == ["openid", "email"]


def test_access_denied_page_uses_statek_branding(monkeypatch):
    """Authorization denials render a Statek-owned 403 screen."""
    _install_fake_easyoidc(
        monkeypatch,
        current_token={"access_token": "rpc-token", "id_token": _id_token(["reader"])},
    )

    app = FastAPI()
    setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/secure")
    async def secure():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/secure")

    assert response.status_code == 403
    assert "Statek" in response.text
    assert "Access blocked" in response.text
    assert "#5c6bc0" in response.text
    assert "#26a69a" in response.text
    assert "SellTime" not in response.text
    assert "selltime" not in response.text.lower()


def test_unauthenticated_request_reaches_oidc_redirect_layer(monkeypatch):
    """Super-admin authorization must not replace login redirects with a 403 page."""
    _install_fake_easyoidc(monkeypatch, current_token=None)

    app = FastAPI()
    setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/secure")
    async def secure():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/secure")

    assert response.status_code == 200
    assert "Access blocked" not in response.text


def test_super_admin_user_can_access_dashboard(monkeypatch):
    """Users in the required Cognito group can access dashboard routes."""
    _install_fake_easyoidc(
        monkeypatch,
        current_token={
            "access_token": "rpc-token",
            "id_token": _id_token(["super-admin"]),
        },
    )

    app = FastAPI()
    setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/secure")
    async def secure():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/secure")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_configured_cognito_group_can_access_dashboard(monkeypatch):
    """REQUIRED_COGNITO_GROUP controls the single dashboard authorization rule."""
    _install_fake_easyoidc(
        monkeypatch,
        current_token={
            "access_token": "rpc-token",
            "id_token": _id_token(["ops-admin"]),
        },
    )

    app = FastAPI()
    setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(required_cognito_group="ops-admin"),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/secure")
    async def secure():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/secure")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_super_admin_check_does_not_block_nicegui_internal_routes(monkeypatch):
    """NiceGUI websocket and asset routes must stay reachable after page authorization."""
    _install_fake_easyoidc(
        monkeypatch,
        current_token={"access_token": "rpc-token", "id_token": _id_token(["reader"])},
    )

    app = FastAPI()
    setup_oidc_auth(
        nicegui_app=app,
        settings=OIDCSettingsBase(),
        skip_auth=False,
        impersonate=None,
        session_file="/tmp/test_statek_nicegui_oidc_sessions",
    )
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/_nicegui/test")
    async def nicegui_internal_route():
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/_nicegui/test")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
