"""
Statek Dashboard — NiceGUI web UI.

Read-only HTTP service for reviewing job processing logs and configuration.
Uses ui.sub_pages for SPA-style navigation: header and left nav render once
and persist across route changes — only the content area is swapped client-side
(no full page reload, no WebSocket reconnect).

Configuration via environment variables:
  DBZERO_ROOT      — path to the db0 data directory (required)
  STATEK_ORG_NAME      — organisation name (default: Selltime)
  STATEK_PROJECT_NAME  — project name (default: selltime)
  STATEK_ENV           — environment (default: dev)
  STATEK_UI_HOST       — listen address (default: 0.0.0.0)
  STATEK_UI_PORT       — listen port (default: 8765)
"""

import argparse
import logging
import os
import sys

log = logging.getLogger(__name__)

# Add external project paths to sys.path so db0 can resolve callables from
# modules outside this package.  Configured via STATEK_EXTERNAL_PATHS in
# .env_statek (colon-separated list of directories, e.g. /src/selltime).
_EXTERNAL_PATHS = os.environ.get('STATEK_EXTERNAL_PATHS', '')
for _ext_path in _EXTERNAL_PATHS.split(':'):
    _ext_path = _ext_path.strip()
    if _ext_path and _ext_path not in sys.path:
        sys.path.insert(0, _ext_path)
        log.debug('Added external path to sys.path: %s', _ext_path)

from nicegui import ui, app
import dbzero as db0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ORG_NAME = os.environ['STATEK_ORG_NAME']
PROJECT_NAME = os.environ['STATEK_PROJECT_NAME']
ENV = os.environ['STATEK_ENV']
DATA_PREFIX = f"/{ORG_NAME}/{PROJECT_NAME}/{ENV}/data"
STATEK_PREFIX = f"/{ORG_NAME}/{PROJECT_NAME}/{ENV}/statek"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Statek Dashboard')
    parser.add_argument('--host', default=os.environ.get('STATEK_UI_HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('STATEK_UI_PORT', '8765')))
    parser.add_argument('--db0-path', default=os.environ.get('DBZERO_ROOT', '.dbzero_data'),
                        help='Path to the db0 data directory')
    parser.add_argument('--import', dest='imports', action='append', default=[],
                        metavar='MODULE',
                        help='Import a module so db0 can deserialize its @db0.memo classes '
                             '(may be repeated, e.g. --import selltime.agents)')
    args, _ = parser.parse_known_args()
    return args


_args = _parse_args()

# Eagerly import modules so db0 can deserialize their @db0.memo classes
# (e.g. agent subclasses defined in external projects).
import importlib  # pylint: disable=wrong-import-position,wrong-import-order

for _mod_name in _args.imports:
    try:
        importlib.import_module(_mod_name)
        log.info('Imported external module: %s', _mod_name)
    except Exception:  # pylint: disable=broad-except
        log.warning('Failed to import external module: %s', _mod_name, exc_info=True)


# ---------------------------------------------------------------------------
# db0 lifecycle
# ---------------------------------------------------------------------------

@app.on_startup
def startup():
    db0.init(_args.db0_path, prefix=STATEK_PREFIX, read_write=False)
    log.info('db0 initialised (read-only): path=%s prefix=%s', _args.db0_path, STATEK_PREFIX)
    log.info('db0 prefixes: %s', list(db0.get_prefixes()))


@app.on_shutdown
def shutdown():
    db0.close()


# ---------------------------------------------------------------------------
# Brand / CSS
# ---------------------------------------------------------------------------

BRAND_PRIMARY   = '#5c6bc0'   # indigo-ish — technical/monitoring feel
BRAND_SECONDARY = '#26a69a'   # teal accent

_BRAND_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

body, .nicegui-content {
    font-family: 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    background-color: #f5f5f5;
}

.q-header {
    background: linear-gradient(135deg, %(primary)s, %(secondary)s) !important;
}

.statek-sidebar {
    background-color: #ffffff !important;
    border-right: 1px solid #e0e0e0 !important;
}

.statek-nav-item {
    border-radius: 8px !important;
    margin: 2px 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #424242 !important;
    justify-content: flex-start !important;
}

.statek-nav-item:hover {
    background: #ede7f6 !important;
    color: %(primary)s !important;
}

.statek-nav-item.active-nav {
    background: #ede7f6 !important;
    color: %(primary)s !important;
}

.statek-nav-section {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #9e9e9e;
    text-transform: uppercase;
    padding: 12px 16px 4px;
}

.nicegui-sub-pages {
    width: 100%%;
    align-items: stretch;
}

.nicegui-sub-pages > * {
    width: 100%%;
    max-width: 100%%;
}

code, .q-badge, pre {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
""" % {'primary': BRAND_PRIMARY, 'secondary': BRAND_SECONDARY}


# ---------------------------------------------------------------------------
# Page imports (after db0 is configured at module level)
# ---------------------------------------------------------------------------

from web_ui.pages.agents import create_agents_page      # noqa: E402
from web_ui.pages.job_defs import create_job_defs_page  # noqa: E402
from web_ui.pages.jobs import create_jobs_page          # noqa: E402


# ---------------------------------------------------------------------------
# Navigation structure
#
# Adding a future page:
#   1. Create web_ui/pages/<name>.py with a create_<name>_page() function
#   2. Import it above
#   3. Add a _NAV_ITEMS entry with the route, label, icon, and builder
#   4. Register the builder in _SUB_PAGE_ROUTES
# ---------------------------------------------------------------------------

_NAV_ITEMS = [
    # (route, label, icon, section)
    ('/agents',   'Agents',           'smart_toy',  'Configuration'),
    ('/job_defs', 'Job Definitions',  'description','Configuration'),
    ('/jobs',     'All Jobs',         'list',       'Jobs'),
    # --- Future views ---
    # ('/stats',    'Job Statistics',   'bar_chart',  'Jobs'),
]


# ---------------------------------------------------------------------------
# Sub-page content builders
# ---------------------------------------------------------------------------

def _home_content() -> None:
    with ui.column().classes('items-center justify-center gap-6 mt-16'):
        ui.icon('monitoring').classes('text-8xl text-indigo-300')
        ui.label('Statek Dashboard').classes('text-3xl font-bold text-gray-800')
        ui.label('Select a view from the sidebar.').classes('text-gray-500')


def _agents_content() -> None:
    create_agents_page()


def _job_defs_content() -> None:
    create_job_defs_page()


def _jobs_content() -> None:
    create_jobs_page()


# Placeholder builders for future pages (imported by route registration only)
def _not_implemented_content(label: str):
    def _inner():
        with ui.column().classes('items-center justify-center gap-4 mt-16'):
            ui.icon('construction').classes('text-6xl text-gray-300')
            ui.label(label).classes('text-xl font-semibold text-gray-600')
            ui.label('Coming soon.').classes('text-gray-400')
    return _inner


_SUB_PAGE_ROUTES = {
    '/':          _home_content,
    '/agents':    _agents_content,
    '/job_defs':  _job_defs_content,
    '/jobs':      _jobs_content,
    # Future:
    # '/stats':   _not_implemented_content('Job Statistics'),
    # '/jobs/{job_uuid}': _not_implemented_content('Job Detail'),
}


# ---------------------------------------------------------------------------
# Shared layout
# ---------------------------------------------------------------------------

def _create_header() -> None:
    ui.add_head_html(f'<style>{_BRAND_CSS}</style>')
    with ui.header().classes('text-white items-center px-4 gap-3'):
        ui.icon('monitoring').classes('text-2xl')
        ui.label('Statek Dashboard').classes('text-lg font-semibold tracking-wide')
        ui.space()
        ui.label('read-only').classes('text-xs opacity-60 font-mono')


def _create_sidebar() -> None:
    with ui.left_drawer(value=True).props('width=220 bordered').classes('statek-sidebar pt-2'):
        # Group nav items by section
        sections: dict[str, list] = {}
        for route, label, icon, section in _NAV_ITEMS:
            sections.setdefault(section, []).append((route, label, icon))

        for section_name, items in sections.items():
            ui.label(section_name).classes('statek-nav-section')
            for route, label, icon in items:
                ui.button(
                    label,
                    icon=icon,
                    on_click=lambda r=route: ui.navigate.to(r),
                ).props('flat no-caps align=left').classes('statek-nav-item w-full')

        # Divider before future sections placeholder
        ui.separator().classes('my-2 mx-4')
        ui.label('Jobs').classes('statek-nav-section')
        for label, icon in [('Job Statistics', 'bar_chart')]:
            ui.button(label, icon=icon).props('flat no-caps align=left disable').classes(
                'statek-nav-item w-full opacity-40'
            ).tooltip('Coming soon')


@ui.page('/')
@ui.page('/{_:path}')
def layout() -> None:
    """Persistent shell — header + sidebar — with sub_pages for the content area."""
    _create_header()
    _create_sidebar()

    with ui.column().classes('flex-grow w-full items-stretch pt-6 px-8 pb-8'):
        ui.sub_pages(_SUB_PAGE_ROUTES).classes('w-full items-stretch')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ui.run(
        title='Statek Dashboard',
        host=_args.host,
        port=_args.port,
        reload=False,
        show_welcome_message=False,
        loop='asyncio',
        storage_secret='statek-dashboard-secret',
    )
