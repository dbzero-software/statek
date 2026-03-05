"""Job Definitions list page for the Statek web UI."""

import dbzero as db0
from nicegui import ui

from web_ui.model_bindings import get_all_job_defs


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

    with ui.card().classes('w-full shadow-sm hover:shadow-md transition-shadow'):
        with ui.row().classes('w-full items-start justify-between mb-2'):
            with ui.column().classes('gap-0'):
                ui.label(agent_role).classes('text-lg font-bold text-gray-900')
                ui.label(f'UUID: {uuid_str}').classes('text-xs text-gray-400 font-mono')

            if job_def.job_params:
                ui.badge(f'{len(job_def.job_params)} params', color='secondary').classes('self-start')

        ui.separator()

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

        # Prompt template preview (from agent metadata)
        prompt_text = job_def.prompt()
        if prompt_text:
            with ui.expansion('Prompt Template', icon='chat').classes('w-full'):
                preview = prompt_text[:500] + ('…' if len(prompt_text) > 500 else '')
                ui.code(preview, language='text').classes('w-full text-xs')


def create_job_defs_page() -> None:
    """Render the Job Definitions list page."""
    job_defs = list(get_all_job_defs())

    with ui.row().classes('w-full items-center justify-between mb-4'):
        ui.label('Job Definitions').classes('text-2xl font-bold text-gray-900')
        ui.badge(str(len(job_defs)), color='primary').classes('text-sm')

    if not job_defs:
        with ui.card().classes('w-full'):
            ui.label('No job definitions found in the database.').classes('text-gray-500 italic')
        return

    job_defs_sorted = sorted(
        job_defs,
        key=lambda jd: (jd.agent.role if jd.agent else '')
    )
    with ui.column().classes('w-full gap-4'):
        for job_def in job_defs_sorted:
            _render_job_def_card(job_def)
