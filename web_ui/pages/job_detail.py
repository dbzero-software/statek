"""Job detail view for the Statek web UI."""

from typing import Optional

import dbzero as db0
from nicegui import ui

from statek.utils import CodeBlock
from web_ui.components.status_badge import create_status_badge


# ---------------------------------------------------------------------------
# Pure helper functions (testable without NiceGUI)
# ---------------------------------------------------------------------------

def _get_console_slice(console: Optional[list], from_pos: int, to_pos: int) -> str:
    """Return joined console output from from_pos (inclusive) to to_pos (exclusive).

    Each console item is a separate append — ensure items are newline-separated
    so that two consecutive prints don't appear on the same line.
    """
    if not console:
        return ''
    chunk = console[from_pos:to_pos]
    parts = []
    for item in chunk:
        s = str(item)
        parts.append(s if s.endswith('\n') else s + '\n')
    return ''.join(parts)


def _get_warmup_blocks(job) -> list:
    """Return list of warmup code blocks (str or CodeBlock-like). Empty if no warmup."""
    warmup = job.job_def.warmup_code
    if warmup is None:
        return []
    if isinstance(warmup, str):
        return [warmup]
    if hasattr(warmup, 'code'):
        # Single CodeBlock-like object
        return [warmup]
    # Sequence of blocks (list, tuple, etc.)
    try:
        return list(warmup)
    except TypeError:
        return [warmup]


def _get_warmup_console_ranges(job) -> list[tuple[int, int]]:
    """Return list of (from_pos, to_pos) console ranges for each warmup block."""
    blocks = _get_warmup_blocks(job)
    if not blocks:
        return []

    n = len(blocks)
    positions = job.warmup_console_positions  # recorded after each block completes
    console_len = len(job.py_env.console) if job.py_env.console else 0

    ranges = []
    prev = 0
    for i in range(n):
        if i < len(positions):
            end = positions[i]
        else:
            # Block hasn't finished yet — use full console
            end = console_len
        ranges.append((prev, end))
        prev = end
    return ranges


def _get_turn_console_ranges(job) -> list[tuple[int, int]]:
    """Return list of (from_pos, to_pos) console ranges for each LLM turn in chat_log."""
    if not job.chat_log:
        return []

    warmup_positions = job.warmup_console_positions
    warmup_base = warmup_positions[-1] if warmup_positions else 0

    ranges = []
    prev = warmup_base
    for item in job.chat_log:
        ranges.append((prev, item.console_pos))
        prev = item.console_pos
    return ranges


def _get_code_str(llm_resp) -> str:
    """Extract code string from a str, CodeBlock-like object, list, or None."""
    if llm_resp is None:
        return ''
    if hasattr(llm_resp, 'code'):
        code = llm_resp.code
        if isinstance(code, list):
            return ''.join(str(c) for c in code)
        return code or ''
    if isinstance(llm_resp, list):
        return ''.join(str(c) for c in llm_resp)
    return str(llm_resp)


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

_LLM_HEADER_BG    = 'background: linear-gradient(90deg, #e8eaf6, #ede7f6)'
_CODE_WARMUP_BG   = '#fdf6e3'
_CODE_LLM_BG      = '#f3f4fd'


def _render_code_block(code: str, bg_color: str, label: str) -> None:
    """Render a labeled code block with syntax highlighting style."""
    with ui.column().classes('w-full gap-0'):
        with ui.row().classes('items-center gap-2 px-3 py-1 rounded-t').style(
            f'background-color: {bg_color}; border: 1px solid #e0e0e0; border-bottom: none'
        ):
            ui.icon('code').classes('text-sm text-gray-500')
            ui.label(label).classes('text-xs font-semibold text-gray-600 uppercase tracking-wide')
        ui.code(code or '(empty)', language='python').classes('w-full rounded-t-none text-xs').style(
            'border-radius: 0 0 6px 6px; margin-top: 0'
        )


def _render_console_output(output: str, has_error: bool = False) -> None:
    """Render console output on a light background with wrapping."""
    if not output.strip():
        ui.label('(no output)').classes('text-xs text-gray-400 italic px-2')
        return
    color = '#b71c1c' if has_error else '#37474f'
    bg = '#fff5f5' if has_error else '#f8f9fa'
    border = '#ffcdd2' if has_error else '#e0e0e0'
    ui.html(
        f'<pre style="'
        f'white-space: pre-wrap; word-break: break-word; overflow-wrap: break-word;'
        f'font-family: \'JetBrains Mono\', monospace;'
        f'font-size: 0.75rem; line-height: 1.5;'
        f'color: {color}; background: {bg};'
        f'border: 1px solid {border}; border-radius: 6px;'
        f'padding: 12px; margin: 0; width: 100%; box-sizing: border-box;'
        f'">{output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</pre>'
    ).classes('w-full')



def _render_warmup_section(job, blocks: list, ranges: list[tuple[int, int]]) -> None:
    """Render all warmup blocks with their console output."""
    for i, (block, (from_pos, to_pos)) in enumerate(zip(blocks, ranges)):
        code = _get_code_str(block)
        console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
        block_label = f'Warmup Code {i + 1} / {len(blocks)}' if len(blocks) > 1 else 'Warmup Code'

        with ui.column().classes('w-full gap-2'):
            _render_code_block(code, _CODE_WARMUP_BG, block_label.lower())

            with ui.expansion('Console Output', icon='terminal', value=True).props('dense').classes(
                'w-full rounded border border-gray-200'
            ):
                _render_console_output(console_out)


def _render_turn_section(job, turn_idx: int, chat_item, from_pos: int, to_pos: int) -> None:
    """Render a single LLM turn with code and console output."""
    code = _get_code_str(chat_item.llm_resp)
    console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
    ts = getattr(chat_item, 'timestamp', None)
    ts_str = ts.strftime('%H:%M:%S') if ts else ''

    exceptions = getattr(job.py_env, 'exceptions', None) or {}
    has_error = turn_idx in exceptions

    with ui.column().classes('w-full gap-2'):
        # Section header
        with ui.row().classes('w-full items-center gap-3 px-3 py-2 rounded-lg').style(
            _LLM_HEADER_BG + '; border: 1px solid #c5cae9'
        ):
            ui.icon('smart_toy').classes('text-indigo-600')
            ui.label(f'Turn {turn_idx + 1}').classes('text-sm font-bold text-indigo-800 flex-1')
            if has_error:
                ui.icon('error_outline').classes('text-red-500 text-sm')
            if ts_str:
                ui.label(ts_str).classes('text-xs text-indigo-500 font-mono')
            ui.label('LLM CODE').classes(
                'text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700'
            )

        _render_code_block(code, _CODE_LLM_BG, 'LLM submitted code')

        if has_error:
            with ui.row().classes('items-center gap-2 px-3 py-2 rounded').style(
                'background: #fff0f0; border: 1px solid #ffcdd2'
            ):
                ui.icon('error').classes('text-red-500 text-sm')
                ui.label(exceptions[turn_idx]).classes('text-xs text-red-700 font-mono break-all')

        with ui.expansion('Console Output', icon='terminal').props('dense').classes(
            'w-full rounded border border-gray-200'
        ):
            _render_console_output(console_out, has_error=has_error)


def create_job_detail_dialog(job) -> None:
    """Open a full-screen dialog showing the job execution breakdown."""
    try:
        uuid_str = str(db0.uuid(job))
    except Exception:  # pylint: disable=broad-except
        uuid_str = '—'

    agent_role = '—'
    try:
        if job.job_def and job.job_def.agent:
            agent_role = job.job_def.agent.role
    except Exception:  # pylint: disable=broad-except
        pass

    model = getattr(job, 'model', None) or '—'
    total_cost = getattr(job, 'total_cost', 0.0) or 0.0
    num_turns = getattr(job, 'num_turns', 0) or 0
    exception_count = getattr(job, 'exception_count', 0) or 0
    status_str = '—'
    try:
        status_str = str(job.status) if job.status else '—'
    except Exception:  # pylint: disable=broad-except
        pass

    system_prompt = ''
    initial_prompt = ''
    try:
        if job.job_def and job.job_def.agent:
            system_prompt = job.job_def.agent.system_prompt(
                job_params=job.job_def.job_params
            ) or ''
        initial_prompt = job.job_def.prompt() or ''
    except Exception:  # pylint: disable=broad-except
        pass

    warmup_blocks = _get_warmup_blocks(job)
    warmup_ranges = _get_warmup_console_ranges(job)
    turn_ranges = _get_turn_console_ranges(job)

    with ui.dialog().props('maximized') as dlg:
        with ui.card().classes('w-full h-full rounded-none overflow-auto').style('max-height: 100vh'):
            # ── Header ──────────────────────────────────────────────────────
            with ui.row().classes('w-full items-start justify-between mb-2 gap-4'):
                with ui.column().classes('gap-1 flex-1 min-w-0'):
                    with ui.row().classes('items-center gap-3 flex-wrap'):
                        ui.label('Job Detail').classes('text-2xl font-bold text-gray-900')
                        create_status_badge(status_str)

                    ui.label(uuid_str).classes('text-xs font-mono text-gray-400 break-all')

                    with ui.row().classes('items-center gap-4 flex-wrap mt-1'):
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('smart_toy').classes('text-sm text-gray-500')
                            ui.label(agent_role).classes('text-sm font-medium text-gray-700')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('memory').classes('text-sm text-gray-500')
                            ui.label(model).classes('text-xs font-mono text-gray-600')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('payments').classes('text-sm text-gray-500')
                            ui.label(f'${total_cost:.4f}').classes('text-xs font-mono text-gray-600')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('repeat').classes('text-sm text-gray-500')
                            ui.label(f'{num_turns} turn{"s" if num_turns != 1 else ""}').classes(
                                'text-xs text-gray-600'
                            )
                        if exception_count:
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('error_outline').classes('text-sm text-red-500')
                                ui.label(f'{exception_count} error{"s" if exception_count != 1 else ""}').classes(
                                    'text-xs text-red-600 font-medium'
                                )

                ui.button(icon='close', on_click=dlg.close).props('flat round dense')

            ui.separator().classes('mb-4')

            # ── System Prompt ───────────────────────────────────────────────
            if system_prompt:
                with ui.expansion('System Prompt', icon='description').props('dense').classes(
                    'w-full rounded border border-gray-200 mb-4'
                ):
                    ui.label(system_prompt).classes(
                        'text-xs text-gray-700 whitespace-pre-wrap w-full rounded p-3'
                    ).style('font-family: "JetBrains Mono", monospace; background: #fdf6e3')

            # ── Initial Prompt ──────────────────────────────────────────────
            if initial_prompt:
                with ui.column().classes('w-full gap-1 mb-4'):
                    with ui.row().classes('items-center gap-2 px-3 py-2 rounded-lg').style(
                        'background: #e8f5e9; border: 1px solid #a5d6a7'
                    ):
                        ui.icon('chat').classes('text-green-700')
                        ui.label('Initial Prompt').classes('text-sm font-bold text-green-800')

                    ui.label(initial_prompt).classes(
                        'text-sm text-gray-700 whitespace-pre-wrap w-full rounded p-3'
                    ).style('background: #f1f8e9; border: 1px solid #c5e1a5')

            # ── Warmup Blocks ───────────────────────────────────────────────
            if warmup_blocks:
                with ui.column().classes('w-full gap-3 mb-4'):
                    _render_warmup_section(job, warmup_blocks, warmup_ranges)

            # ── LLM Turns ───────────────────────────────────────────────────
            if job.chat_log:
                with ui.column().classes('w-full gap-3'):
                    for i, (chat_item, (from_pos, to_pos)) in enumerate(
                        zip(job.chat_log, turn_ranges)
                    ):
                        _render_turn_section(job, i, chat_item, from_pos, to_pos)

            if not warmup_blocks and not job.chat_log:
                with ui.column().classes('items-center justify-center gap-3 mt-8'):
                    ui.icon('hourglass_empty').classes('text-4xl text-gray-300')
                    ui.label('No execution history yet.').classes('text-gray-400 italic')

    dlg.open()
