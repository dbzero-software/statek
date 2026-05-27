"""OIDC authentication helpers for the Statek NiceGUI web UI."""

from __future__ import annotations

from contextvars import ContextVar
from html import escape as html_escape
import logging
from typing import Any, Awaitable, Callable
from urllib.parse import quote as url_quote, urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from statek.multi_source_settings import MultiSourceBaseSettings

log = logging.getLogger(__name__)

_AUTH_PATHS = {"/login", "/auth/callback", "/signout"}
_ACCESS_CHECK_BYPASS_PATHS = _AUTH_PATHS | {"/favicon.ico"}
_ACCESS_CHECK_BYPASS_PREFIXES = ("/_nicegui", "/static/")
_BRAND_PRIMARY = "#5c6bc0"
_BRAND_SECONDARY = "#26a69a"

AccessCheck = Callable[[Any | None, Request], bool]
CallNext = Callable[[Request], Awaitable[Any]]


class OIDCSettingsBase(MultiSourceBaseSettings):
    """Base OIDC settings shared by Statek web UI configuration classes.

    Args:
        oidc_well_known_url: OpenID Connect discovery document URL.
        oidc_client_id: OIDC client identifier.
        oidc_client_secret: OIDC client secret.
        oidc_redirect_uri: Callback URL registered with the provider.
        oidc_post_logout_uri: URL used after provider sign-out.
        oidc_scope: Space-separated OIDC scopes requested by the UI.
        oidc_logout_url: Provider logout endpoint.
        cookie_secret_key: Secret used by the web session middleware.
    """

    oidc_well_known_url: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_redirect_uri: str = ""
    oidc_post_logout_uri: str = "/"
    oidc_scope: str = "openid email"
    oidc_logout_url: str = ""
    cookie_secret_key: str = ""


def _get_rpc_access_token(auth_client: Any | None) -> str | None:
    """Return the current OAuth access token for dbzero RPC authorization."""
    if auth_client is None:
        return None

    token = auth_client._get_current_token()  # pylint: disable=protected-access
    return token.get("access_token") if token is not None else None


async def _call_with_rpc_auth_token(
    rpc_auth_token_var: ContextVar[Any],
    auth_client: Any | None,
    request: Request,
    call_next: CallNext,
) -> Any:
    """Run an HTTP request with the current OAuth token in the RPC context."""
    context_token = rpc_auth_token_var.set(_get_rpc_access_token(auth_client))
    try:
        return await call_next(request)
    finally:
        rpc_auth_token_var.reset(context_token)


def _set_rpc_auth_token_provider(
    rpc_auth_token_var: ContextVar[Any] | None,
    auth_client: Any | None,
) -> Any | None:
    """Set a lazy token provider for NiceGUI callback contexts outside middleware."""
    if rpc_auth_token_var is None or auth_client is None:
        return None

    return rpc_auth_token_var.set(lambda: _get_rpc_access_token(auth_client))


def _render_auth_status_page(
    title: str,
    heading: str,
    message: str,
    action_href: str,
    action_label: str,
    detail: str = "",
) -> str:
    """Render a Statek authentication status page."""
    safe_title = html_escape(title)
    safe_heading = html_escape(heading)
    safe_message = html_escape(message)
    safe_detail = html_escape(detail)
    safe_action_href = html_escape(action_href, quote=True)
    safe_action_label = html_escape(action_label)
    message_html = f'<p class="statek-auth-message">{safe_message}</p>' if message else ""
    detail_html = f'<p class="statek-auth-detail">{safe_detail}</p>' if detail else ""

    return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{safe_title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --statek-primary: {_BRAND_PRIMARY};
            --statek-secondary: {_BRAND_SECONDARY};
            --statek-text: #212121;
            --statek-muted: #607d8b;
            --statek-surface: rgba(255, 255, 255, 0.94);
            --statek-border: rgba(255, 255, 255, 0.3);
        }}

        * {{ box-sizing: border-box; }}

        body {{
            min-height: 100vh;
            margin: 0;
            display: grid;
            place-items: center;
            padding: 2rem;
            background:
                radial-gradient(circle at 18% 18%, rgba(255, 255, 255, 0.24), transparent 28rem),
                linear-gradient(135deg, var(--statek-primary), var(--statek-secondary));
            color: var(--statek-text);
            font-family: 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        .statek-auth-card {{
            width: min(100%, 460px);
            padding: 2.5rem;
            border: 1px solid var(--statek-border);
            border-radius: 18px;
            background: var(--statek-surface);
            box-shadow: 0 24px 70px rgba(33, 33, 33, 0.24);
            text-align: center;
            backdrop-filter: blur(14px);
        }}

        .statek-auth-kicker {{
            margin: 0 0 1rem;
            color: var(--statek-secondary);
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        h1 {{
            margin: 0;
            color: var(--statek-text);
            font-size: clamp(2rem, 6vw, 2.6rem);
            line-height: 1.08;
        }}

        .statek-auth-message {{
            margin: 1.25rem 0 0;
            color: var(--statek-muted);
            font-size: 1rem;
            line-height: 1.65;
        }}

        .statek-auth-detail {{
            margin: 1rem 0 0;
            padding: 0.875rem 1rem;
            border-radius: 10px;
            background: rgba(92, 107, 192, 0.09);
            color: var(--statek-primary);
            font-size: 0.9rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }}

        .statek-auth-action {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            margin-top: 1.75rem;
            padding: 0 1.35rem;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--statek-primary), var(--statek-secondary));
            color: #ffffff;
            font-weight: 700;
            text-decoration: none;
            box-shadow: 0 10px 22px rgba(92, 107, 192, 0.26);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .statek-auth-action:hover {{
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(38, 166, 154, 0.3);
        }}

        @media (max-width: 560px) {{
            body {{ padding: 1rem; }}
            .statek-auth-card {{ padding: 2rem 1.25rem; }}
        }}
    </style>
</head>
<body>
    <main class="statek-auth-card">
        <p class="statek-auth-kicker">Statek Dashboard</p>
        <h1>{safe_heading}</h1>
        {message_html}
        {detail_html}
        <a class="statek-auth-action" href="{safe_action_href}">{safe_action_label}</a>
    </main>
</body>
</html>'''


def _oidc_scope_values(oidc_scope: str) -> list[str]:
    """Return provider scope names from env-file compatible scope text."""
    scope_text = oidc_scope.strip()
    if len(scope_text) >= 2 and scope_text[0] == scope_text[-1] and scope_text[0] in {'"', "'"}:
        scope_text = scope_text[1:-1]

    return scope_text.split()


def _should_skip_access_check(path: str) -> bool:
    """Return whether path belongs to auth callbacks or NiceGUI internals."""
    return path in _ACCESS_CHECK_BYPASS_PATHS or path.startswith(_ACCESS_CHECK_BYPASS_PREFIXES)


def setup_oidc_auth(
    nicegui_app: Any,
    settings: OIDCSettingsBase,
    skip_auth: bool,
    impersonate: str | None,
    session_file: str,
    access_check: AccessCheck | None = None,
    rpc_auth_token_var: ContextVar[Any] | None = None,
) -> Any | None:
    """Set up OIDC authentication for the Statek NiceGUI application.

    Args:
        nicegui_app: The NiceGUI/FastAPI app instance.
        settings: OIDC settings used to configure the provider client.
        skip_auth: When True, no OIDC flow is set up.
        impersonate: Accepted for CLI compatibility; Statek does not resolve app users.
        session_file: Path prefix for shelve session storage.
        access_check: Optional callback used to authorize authenticated requests.
        rpc_auth_token_var: Optional ContextVar populated for dbzero RPC calls.

    Returns:
        The OIDC client instance, or None when authentication is disabled.
    """
    auth = None
    session_storage = None

    def build_signout_url() -> str:
        if not settings.oidc_logout_url:
            return "/"

        post_logout = settings.oidc_post_logout_uri
        if post_logout.startswith("/"):
            parsed = urlparse(settings.oidc_redirect_uri)
            post_logout = f"{parsed.scheme}://{parsed.netloc}{post_logout}"

        return (
            f"{settings.oidc_logout_url}"
            f"?client_id={settings.oidc_client_id}"
            f"&logout_uri={url_quote(post_logout, safe='')}"
        )

    def signout_route(request: Request) -> RedirectResponse:
        if auth is not None:
            try:
                auth._logout()  # pylint: disable=protected-access
            except Exception as error:  # pylint: disable=broad-exception-caught
                log.warning("Error during OIDC logout: %s", error)

        if session_storage is not None:
            try:
                session_storage.reset_keys()
            except Exception as error:  # pylint: disable=broad-exception-caught
                log.warning("Error clearing OIDC session storage: %s", error)

        if "session" in request.scope:
            request.session.clear()

        return RedirectResponse(build_signout_url())

    nicegui_app.add_route("/signout", signout_route)

    if skip_auth:
        log.warning("Statek UI authentication is disabled")
        if impersonate:
            log.warning("Ignoring Statek UI impersonation setting: %s", impersonate)
    else:
        from EasyOIDC import (  # pylint: disable=import-outside-toplevel
            Config as OIDCConfig,
            SessionHandler,
        )
        from EasyOIDC.frameworks.nicegui import (  # pylint: disable=import-outside-toplevel
            NiceGUIOIDClient,
        )

        oidc_config = OIDCConfig(
            well_known_openid_url=settings.oidc_well_known_url,
            client_id=settings.oidc_client_id,
            client_secret=settings.oidc_client_secret,
            redirect_uri=settings.oidc_redirect_uri,
            cookie_secret_key=settings.cookie_secret_key,
            scope=_oidc_scope_values(settings.oidc_scope),
            app_login_route="/login",
            app_logout_route="/signout",
            app_authorize_route="/auth/callback",
            post_logout_uri=settings.oidc_post_logout_uri,
        )
        session_storage = SessionHandler(mode="shelve", filename=session_file)
        session_storage.reset_keys()
        auth = NiceGUIOIDClient(
            nicegui_app,
            auth_config=oidc_config,
            session_storage=session_storage,
        )

    _set_rpc_auth_token_provider(rpc_auth_token_var, auth)

    class OAuthCallbackErrorMiddleware(BaseHTTPMiddleware):
        """Return an error page when the OIDC provider redirects with an error."""

        async def dispatch(self, request: Request, call_next: CallNext) -> Any:
            if request.url.path == "/auth/callback" and "error" in request.query_params:
                error = request.query_params.get("error", "Authentication error")
                description = request.query_params.get("error_description", "")
                return HTMLResponse(
                    _render_auth_status_page(
                        title="Statek - sign-in error",
                        heading="Sign-in failed",
                        message="The identity provider returned an error.",
                        detail=f"{error}: {description}".strip(": "),
                        action_href="/login",
                        action_label="Try again",
                    ),
                    status_code=400,
                )

            return await call_next(request)

    class AccessCheckMiddleware(BaseHTTPMiddleware):
        """Deny access when the configured authorization callback rejects a request."""

        async def dispatch(self, request: Request, call_next: CallNext) -> Any:
            if not _should_skip_access_check(request.url.path):
                is_authenticated = auth is None or (auth and auth.is_authenticated())
                if is_authenticated and access_check is not None and not access_check(auth, request):
                    return HTMLResponse(
                        _render_auth_status_page(
                            title="Statek - access blocked",
                            heading="Access blocked",
                            message="You do not have permission to access this Statek dashboard.",
                            action_href="/signout",
                            action_label="Sign out",
                        ),
                        status_code=403,
                    )

            return await call_next(request)

    class RpcAuthTokenMiddleware(BaseHTTPMiddleware):
        """Populate the dbzero RPC bearer token context for each HTTP request."""

        async def dispatch(self, request: Request, call_next: CallNext) -> Any:
            return await _call_with_rpc_auth_token(
                rpc_auth_token_var,
                auth,
                request,
                call_next,
            )

    if auth is not None:
        nicegui_app.add_middleware(OAuthCallbackErrorMiddleware)
    if access_check is not None:
        nicegui_app.add_middleware(AccessCheckMiddleware)
    if rpc_auth_token_var is not None:
        nicegui_app.add_middleware(RpcAuthTokenMiddleware)

    return auth
