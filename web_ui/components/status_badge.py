"""Reusable UI components for the Statek web UI."""

from ..nicegui_compat import ui

_STATUS_COLORS = {
    'READY':      ('bg-blue-100',   'text-blue-800'),
    'WARMING_UP': ('bg-yellow-100', 'text-yellow-800'),
    'STARTED':    ('bg-green-100',  'text-green-800'),
    'SUSPENDED':  ('bg-orange-100', 'text-orange-800'),
    'DONE':       ('bg-gray-100',   'text-gray-700'),
}

_STATUS_ICONS = {
    'READY':      'radio_button_unchecked',
    'WARMING_UP': 'hourglass_top',
    'STARTED':    'play_circle',
    'SUSPENDED':  'pause_circle',
    'DONE':       'check_circle',
}


def create_status_badge(status_str: str) -> None:
    """Render a coloured inline status badge."""
    bg, text = _STATUS_COLORS.get(status_str, ('bg-gray-100', 'text-gray-700'))
    icon = _STATUS_ICONS.get(status_str, 'help')
    with ui.row().classes(f'items-center gap-1 px-2 py-0.5 rounded-full {bg}'):
        ui.icon(icon).classes(f'text-sm {text}')
        ui.label(status_str).classes(f'text-xs font-medium {text}')
