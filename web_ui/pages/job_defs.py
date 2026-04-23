"""Job Definitions list page for the Statek web UI."""

import logging
from typing import List, Optional

import dbzero as db0

from statek.executors.job import _get_static_task_difficulty
from web_ui.nicegui_compat import ui
from web_ui.model_bindings import get_all_job_defs

log = logging.getLogger(__name__)

PAGE_SIZE = 25


def _paginate(items: list, page: int, page_size: int) -> list:
    """Return the slice of *items* for the given 1-indexed *page*."""
    start = (page - 1) * page_size
    return items[start:start + page_size]


def _job_def_has_errors(job_def) -> bool:
    """Return True if job_def has associated errors; False on any exception."""
    try:
        return job_def.has_errors()
    except Exception:  # pylint: disable=broad-except
        return False


def _job_def_get_errors(job_def) -> list:
    """Return a list of errors for job_def; empty list on any exception."""
    try:
        return list(job_def.get_errors())
    except Exception:  # pylint: disable=broad-except
        return []


def _format_traceback(traceback: Optional[List[str]]) -> str:
    """Join traceback frame strings into a single string."""
    if not traceback:
        return ''
    return ''.join(traceback)


def _get_job_def_system_prompt(job_def) -> str:
    """Return the formatted system prompt for a job definition."""
    agent = job_def.agent
    if agent is None:
        return ''
    try:
        return agent.system_prompt(
            task_difficulty=_get_static_task_difficulty(job_def.metadata),
            job_params=job_def.job_params,
        )
    except Exception:  # pylint: disable=broad-except
        log.exception("Failed to format system prompt for job definition")
        return ''


def _render_warmup_preview(warmup_code) -> None:
    """Render warmup_code — handles str, CodeBlock, or list thereof."""
    if warmup_code is None:
        ui.label('None').classes('text-sm text-gray-500 italic')
        return

    from statek.utils import CodeBlock  # pylint: disable=import-outside-toplevel

    blocks = [warmup_code] if isinstance(warmup_code, (str, CodeBlock)) else list(warmup_code)
    for i, block in enumerate(blocks):
        code_text = block.code if isinstance(block, CodeBlock) else str(block)
        if len(blocks) > 1:
            if i > 0:
                ui.separator().classes('my-2')
            ui.label(f'Block {i + 1} of {len(blocks)}').classes(
                'text-xs text-indigo-600 font-semibold bg-indigo-50 px-2 py-0.5 rounded'
            )
        preview = code_text[:400] + ('…' if len(code_text) > 400 else '')
        ui.code(preview, language='python').classes('w-full text-xs')


def _render_job_def_card(job_def) -> None:
    """Render a single JobDef as an info card."""
    agent = job_def.agent
    agent_role = agent.role if agent is not None else '(no agent)'

    try:
        uuid_str = str(db0.uuid(job_def))
    except Exception:  # pylint: disable=broad-except
        uuid_str = '—'

    has_errors = _job_def_has_errors(job_def)
    border_cls = 'border-l-4 border-red-400' if has_errors else ''

    with ui.card().classes(f'w-full shadow-sm hover:shadow-md transition-shadow {border_cls}'):
        with ui.row().classes('w-full items-start justify-between mb-2'):
            with ui.column().classes('gap-0'):
                ui.label(agent_role).classes('text-lg font-bold text-gray-900')
                ui.label(f'UUID: {uuid_str}').classes('text-xs text-gray-400 font-mono')

            with ui.row().classes('items-center gap-2 self-start'):
                if has_errors:
                    errors = _job_def_get_errors(job_def)
                    ui.badge(f'{len(errors)} error(s)', color='negative').props('rounded')
                if job_def.job_params:
                    ui.badge(f'{len(job_def.job_params)} params', color='secondary')

        ui.separator()

        # Definition errors
        if has_errors:
            errors = _job_def_get_errors(job_def)
            with ui.expansion(f'Errors ({len(errors)})', icon='error').classes('w-full text-red-600'):
                with ui.column().classes('gap-3 pl-2'):
                    for i, err in enumerate(errors):
                        if i > 0:
                            ui.separator()
                        ui.label(err.error_message).classes('text-sm text-red-700 font-mono break-all')
                        tb = _format_traceback(err.traceback)
                        if tb:
                            ui.code(tb, language='text').classes('w-full text-xs')

        # Job parameters
        if job_def.job_params:
            with ui.expansion('Job Parameters', icon='tune').classes('w-full'):
                with ui.column().classes('gap-1 pl-2'):
                    for key, value in job_def.job_params.items():
                        with ui.row().classes('items-start gap-2'):
                            ui.label(f'{key}:').classes('text-xs font-semibold text-gray-600 w-32 shrink-0')
                            ui.label(str(value)).classes('text-xs text-gray-700 font-mono break-all')
        else:
            with ui.row().classes('items-center gap-2 text-gray-500 px-2 py-1'):
                ui.icon('tune').classes('text-sm')
                ui.label('No job parameters.').classes('text-sm italic')

        # Warmup code
        with ui.expansion('Warmup Code', icon='code').classes('w-full'):
            _render_warmup_preview(job_def.warmup_code)

        # System prompt preview
        prompt_text = _get_job_def_system_prompt(job_def)
        if prompt_text:
            with ui.expansion('System Prompt', icon='chat').classes('w-full'):
                preview = prompt_text[:500] + ('…' if len(prompt_text) > 500 else '')
                ui.code(preview, language='text').classes('w-full text-xs')


def create_job_defs_page() -> None:
    """Render the Job Definitions list page."""
    job_defs = list(get_all_job_defs())
    log.info("All job defs: %d", len(job_defs))

    total = len(job_defs)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page_state = {'page': 1}

    with ui.row().classes('w-full items-center justify-between mb-4'):
        ui.label('Job Definitions').classes('text-2xl font-bold text-gray-900')
        ui.badge(str(total), color='primary').classes('text-sm')

    if not job_defs:
        with ui.card().classes('w-full'):
            ui.label('No job definitions found in the database.').classes('text-gray-500 italic')
        return

    job_defs_sorted = sorted(
        job_defs,
        key=lambda jd: (jd.agent.role if jd.agent else '')
    )

    @ui.refreshable
    def job_defs_list() -> None:
        page_items = _paginate(job_defs_sorted, page_state['page'], PAGE_SIZE)
        with ui.column().classes('w-full gap-4'):
            for job_def in page_items:
                _render_job_def_card(job_def)

        if total_pages > 1:
            with ui.row().classes('w-full items-center justify-between mt-3'):
                def _prev(p=page_state):
                    if p['page'] > 1:
                        p['page'] -= 1
                        job_defs_list.refresh()

                def _next(p=page_state):
                    if p['page'] < total_pages:
                        p['page'] += 1
                        job_defs_list.refresh()

                ui.button('Previous', icon='chevron_left', on_click=_prev).props('flat dense')
                ui.label(f'Page {page_state["page"]} of {total_pages}').classes('text-sm text-gray-600')
                ui.button('Next', icon='chevron_right', on_click=_next).props('flat dense')

    job_defs_list()
