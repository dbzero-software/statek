"""Job detail view for the Statek web UI."""

import io
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
    """Return list of (from_pos, to_pos) console ranges for each LLM turn in chat_log.

    Each chat_log item records console_pos as the console length *before* that turn's
    code ran (i.e. the start of its output). The end is the next turn's console_pos,
    or the full console length for the last turn.
    """
    if not job.chat_log:
        return []

    console_len = len(job.py_env.console) if job.py_env.console else 0

    ranges = []
    for i, item in enumerate(job.chat_log):
        from_pos = item.console_pos
        if i + 1 < len(job.chat_log):
            to_pos = job.chat_log[i + 1].console_pos
        else:
            to_pos = console_len
        ranges.append((from_pos, to_pos))
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


def _get_tool_data_for_block(code_block, py_env, key: int) -> list:
    """Return [(call_spec, result_str), ...] for a code block's tool calls.

    Args:
        code_block: str or CodeBlock-like; tool calls come from code_block.tool_calls
        py_env: PyEnv instance (or duck-typed equivalent) with get_tool_result()
        key: the console position key to look up results under

    Returns:
        List of (call_spec, result_str) pairs. Empty if no tool calls or no log entry.
    """
    if not hasattr(code_block, 'tool_calls') or not code_block.tool_calls:
        return []
    if py_env is None:
        return []
    call_specs = list(code_block.tool_calls)
    result = []
    for i, cs in enumerate(call_specs):
        try:
            tool_result = py_env.get_tool_result(key, i)
        except KeyError:
            return []
        except IndexError:
            tool_result = ''
        result.append((cs, tool_result))
    return result


# ---------------------------------------------------------------------------
# Raw repr helper (pure — no NiceGUI dependency)
# ---------------------------------------------------------------------------

def _build_raw_repr(job) -> str:
    """Return a pprint-formatted string of all Job attributes for raw inspection."""
    import pprint  # pylint: disable=import-outside-toplevel

    def _to_display(obj, depth=0):
        if depth > 4:
            try:
                return repr(obj)
            except Exception as exc:  # pylint: disable=broad-except
                return f'(Error rendering repr: {exc})'
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple)):
            converted = []
            for x in obj:
                try:
                    converted.append(_to_display(x, depth + 1))
                except Exception as exc:  # pylint: disable=broad-except
                    converted.append(f'(Error: {exc})')
            return converted if isinstance(obj, list) else tuple(converted)
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                try:
                    result[str(k)] = _to_display(v, depth + 1)
                except Exception as exc:  # pylint: disable=broad-except
                    result[str(k)] = f'(Error: {exc})'
            return result
        try:
            attrs = vars(obj)
        except TypeError:
            try:
                return repr(obj)
            except Exception as exc:  # pylint: disable=broad-except
                return f'(Error rendering repr: {exc})'
        converted_attrs = {}
        for k, v in attrs.items():
            if k.startswith('__'):
                continue
            try:
                converted_attrs[k] = _to_display(v, depth + 1)
            except Exception as exc:  # pylint: disable=broad-except
                converted_attrs[k] = f'(Error: {exc})'
        return {
            '__type__': type(obj).__name__,
            **converted_attrs,
        }

    try:
        data = _to_display(job)
        return pprint.pformat(data, width=120, depth=10, sort_dicts=False)
    except Exception as exc:  # pylint: disable=broad-except
        return f'(Error building raw repr: {exc})'


# ---------------------------------------------------------------------------
# Export helpers (pure — no NiceGUI dependency)
# ---------------------------------------------------------------------------

def _escape_md(text: str) -> str:
    """Escape special markdown characters in plain text (used inside tables)."""
    return text.replace('|', '\\|').replace('\n', ' ')


def _md_code_fence(code: str, lang: str = '') -> str:
    return f'```{lang}\n{code}\n```'


def _build_md_content(
    uuid_str: str,
    status_str: str,
    agent_role: str,
    model: str,
    total_cost: float,
    num_turns: int,
    exception_count: int,
    system_prompt: str,
    initial_prompt: str,
    job,
    warmup_blocks: list,
    warmup_ranges: list[tuple[int, int]],
    turn_ranges: list[tuple[int, int]],
) -> str:
    """Build a markdown document representing the full job execution log."""
    parts: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    parts.append('# Job Detail\n')
    parts.append('| Field | Value |')
    parts.append('|---|---|')
    parts.append(f'| **Status** | {_escape_md(status_str)} |')
    parts.append(f'| **UUID** | `{_escape_md(uuid_str)}` |')
    parts.append(f'| **Agent** | {_escape_md(agent_role)} |')
    parts.append(f'| **Model** | `{_escape_md(model)}` |')
    parts.append(f'| **Cost** | `${total_cost:.4f}` |')
    parts.append(f'| **Turns** | {num_turns} |')
    if exception_count:
        parts.append(f'| **Errors** | {exception_count} |')
    parts.append('')

    # ── System Prompt ────────────────────────────────────────────────────────
    if system_prompt:
        parts.append('---\n')
        parts.append('## System Prompt\n')
        parts.append(_md_code_fence(system_prompt))
        parts.append('')

    # ── Initial Prompt ───────────────────────────────────────────────────────
    if initial_prompt:
        parts.append('---\n')
        parts.append('## Initial Prompt\n')
        parts.append(initial_prompt)
        parts.append('')

    # ── Warmup Blocks ────────────────────────────────────────────────────────
    if warmup_blocks:
        parts.append('---\n')
        parts.append('## Warmup\n')
        for i, (block, (from_pos, to_pos)) in enumerate(zip(warmup_blocks, warmup_ranges)):
            label = f'Warmup Code {i + 1}/{len(warmup_blocks)}' if len(warmup_blocks) > 1 else 'Warmup Code'
            parts.append(f'### {label}\n')
            code = _get_code_str(block)
            parts.append(_md_code_fence(code or '(empty)', 'python'))
            tool_data = _get_tool_data_for_block(block, job.py_env, 0)
            if tool_data:
                parts.append('\n**Tool Calls**\n')
                for cs, result in tool_data:
                    parts.append(f'- **`{cs.format()}`**')
                    if result:
                        parts.append(_md_code_fence(str(result).strip()))
            console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
            if console_out.strip():
                parts.append('\n**Console Output**\n')
                parts.append(_md_code_fence(console_out.rstrip()))
            parts.append('')

    # ── LLM Turns ────────────────────────────────────────────────────────────
    if job.chat_log:
        parts.append('---\n')
        exceptions = getattr(job.py_env, 'exceptions', None) or {}
        for i, (chat_item, (from_pos, to_pos)) in enumerate(zip(job.chat_log, turn_ranges)):
            ts = getattr(chat_item, 'timestamp', None)
            ts_str = f' — {ts.strftime("%H:%M:%S")}' if ts else ''
            parts.append(f'## Turn {i + 1}{ts_str}\n')

            code = _get_code_str(chat_item.llm_resp)
            parts.append(_md_code_fence(code or '(empty)', 'python'))

            tool_data = _get_tool_data_for_block(chat_item.llm_resp, job.py_env, i + 1)
            if tool_data:
                parts.append(f'\n### Tool Call{"s" if len(tool_data) != 1 else ""}\n')
                for cs, result in tool_data:
                    parts.append(f'**`{cs.format()}`**\n')
                    if result:
                        parts.append(_md_code_fence(str(result).strip()))

            if i in exceptions:
                parts.append(f'\n> **Error:** `{exceptions[i]}`\n')

            console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
            if console_out.strip():
                parts.append('\n**Console Output**\n')
                parts.append(_md_code_fence(console_out.rstrip()))

            parts.append('\n---\n')

    return '\n'.join(parts)


def _build_pdf_bytes(md_content: str) -> bytes:
    """Convert markdown content to PDF bytes via weasyprint."""
    import markdown as md_lib
    import weasyprint

    body_html = md_lib.markdown(
        md_content,
        extensions=['fenced_code', 'tables', 'nl2br'],
    )
    css = """
        @page { margin: 20mm 18mm; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.55;
            color: #212121;
        }
        h1 { font-size: 18pt; color: #3949ab; border-bottom: 2px solid #c5cae9; padding-bottom: 6px; }
        h2 { font-size: 13pt; color: #5c6bc0; margin-top: 18px; border-bottom: 1px solid #e8eaf6; }
        h3 { font-size: 11pt; color: #37474f; margin-top: 12px; }
        table { border-collapse: collapse; margin: 8px 0 14px; width: auto; }
        th, td { border: 1px solid #e0e0e0; padding: 5px 10px; font-size: 9.5pt; }
        th { background: #f5f5f5; font-weight: 600; }
        pre {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 10px 12px;
            font-family: 'Courier New', monospace;
            font-size: 8.5pt;
            white-space: pre-wrap;
            word-break: break-word;
            margin: 6px 0;
        }
        code { font-family: 'Courier New', monospace; font-size: 8.5pt; background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }
        pre code { background: none; padding: 0; }
        blockquote { border-left: 3px solid #ef9a9a; background: #fff5f5; margin: 8px 0; padding: 6px 12px; color: #b71c1c; }
        hr { border: none; border-top: 1px solid #e0e0e0; margin: 12px 0; }
    """
    html = f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{body_html}</body></html>'
    buf = io.BytesIO()
    weasyprint.HTML(string=html).write_pdf(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------

_LLM_HEADER_BG    = 'background: linear-gradient(90deg, #e8eaf6, #ede7f6)'
_CODE_WARMUP_BG   = '#fdf6e3'
_CODE_LLM_BG      = '#f3f4fd'
_TOOL_HEADER_BG   = '#fff3e0'
_TOOL_BORDER      = '#ffcc80'
_TOOL_CALL_BG     = '#ffe0b2'
_TOOL_RESULT_BG   = '#fafafa'


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



def _render_tool_calls(tool_data: list) -> None:
    """Render tool calls and their results inline on the timeline."""
    if not tool_data:
        return
    with ui.column().classes('w-full gap-0'):
        with ui.row().classes('items-center gap-2 px-3 py-1.5 rounded-t').style(
            f'background: {_TOOL_HEADER_BG}; border: 1px solid {_TOOL_BORDER}; border-bottom: none'
        ):
            ui.icon('build').classes('text-sm text-orange-700')
            ui.label('Tool Calls').classes('text-xs font-semibold text-orange-800 uppercase tracking-wide')
            ui.label(f'{len(tool_data)}').classes(
                'text-xs font-bold px-1.5 py-0.5 rounded-full bg-orange-200 text-orange-800'
            )
        with ui.column().classes('w-full gap-2').style(
            f'border: 1px solid {_TOOL_BORDER}; border-top: none; border-radius: 0 0 6px 6px;'
            f'padding: 10px; background: {_TOOL_RESULT_BG}'
        ):
            for i, (cs, result) in enumerate(tool_data):
                if i > 0:
                    ui.separator().classes('my-1')
                with ui.column().classes('w-full gap-1'):
                    # Call signature
                    with ui.row().classes('items-center gap-2 px-2 py-1 rounded').style(
                        f'background: {_TOOL_CALL_BG}'
                    ):
                        ui.icon('call_made').classes('text-xs text-orange-700')
                        ui.label(cs.format()).classes('text-xs font-mono font-semibold text-orange-900 break-all')
                    # Result
                    result_str = str(result).strip() if result else ''
                    if result_str:
                        ui.html(
                            f'<pre style="'
                            f'white-space: pre-wrap; word-break: break-word; overflow-wrap: break-word;'
                            f'font-family: \'JetBrains Mono\', monospace;'
                            f'font-size: 0.72rem; line-height: 1.5;'
                            f'color: #4e342e; background: #fff8f0;'
                            f'border: 1px solid {_TOOL_BORDER}; border-radius: 4px;'
                            f'padding: 8px; margin: 0; width: 100%; box-sizing: border-box;'
                            f'">{result_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</pre>'
                        ).classes('w-full')
                    else:
                        ui.label('(no result)').classes('text-xs text-gray-400 italic px-2')


def _render_warmup_section(job, blocks: list, ranges: list[tuple[int, int]]) -> None:
    """Render all warmup blocks with their console output."""
    for i, (block, (from_pos, to_pos)) in enumerate(zip(blocks, ranges)):
        code = _get_code_str(block)
        console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
        block_label = f'Warmup Code {i + 1} / {len(blocks)}' if len(blocks) > 1 else 'Warmup Code'
        tool_data = _get_tool_data_for_block(block, job.py_env, 0)

        with ui.column().classes('w-full gap-2'):
            _render_code_block(code, _CODE_WARMUP_BG, block_label.lower())

            if tool_data:
                _render_tool_calls(tool_data)

            with ui.expansion('Console Output', icon='terminal', value=True).props('dense').classes(
                'w-full rounded border border-gray-200'
            ):
                _render_console_output(console_out)


def _render_turn_section(job, turn_idx: int, chat_item, from_pos: int, to_pos: int) -> None:
    """Render a single LLM turn with code, tool calls, and console output."""
    code = _get_code_str(chat_item.llm_resp)
    console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
    ts = getattr(chat_item, 'timestamp', None)
    ts_str = ts.strftime('%H:%M:%S') if ts else ''

    exceptions = getattr(job.py_env, 'exceptions', None) or {}
    has_error = turn_idx in exceptions
    # tool_log key for LLM turn i is i+1 (stored after append_chat_log, len=i+1)
    tool_data = _get_tool_data_for_block(chat_item.llm_resp, job.py_env, turn_idx + 1)

    with ui.column().classes('w-full gap-2'):
        # Section header
        with ui.row().classes('w-full items-center gap-3 px-3 py-2 rounded-lg').style(
            _LLM_HEADER_BG + '; border: 1px solid #c5cae9'
        ):
            ui.icon('smart_toy').classes('text-indigo-600')
            ui.label(f'Turn {turn_idx + 1}').classes('text-sm font-bold text-indigo-800 flex-1')
            if tool_data:
                ui.icon('build').classes('text-orange-500 text-sm')
                ui.label(f'{len(tool_data)} tool call{"s" if len(tool_data) != 1 else ""}').classes(
                    'text-xs font-semibold px-2 py-0.5 rounded-full bg-orange-100 text-orange-700'
                )
            if has_error:
                ui.icon('error_outline').classes('text-red-500 text-sm')
            if ts_str:
                ui.label(ts_str).classes('text-xs text-indigo-500 font-mono')
            ui.label('LLM CODE').classes(
                'text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700'
            )

        _render_code_block(code, _CODE_LLM_BG, 'LLM submitted code')

        if tool_data:
            _render_tool_calls(tool_data)

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


def _get_system_prompt(job) -> tuple[str, Optional[str]]:
    """Return (prompt_text, error_message) for a job's system prompt.

    Tries to return the fully formatted prompt. On formatting failure, falls back
    to the raw template and sets error_message to describe what went wrong.
    Returns ('', None) if no system prompt is available.
    """
    try:
        if not job.job_def or not job.job_def.agent:
            return '', None
        try:
            return job.job_def.agent.system_prompt(
                job_params=job.job_def.job_params
            ) or '', None
        except Exception as exc:  # pylint: disable=broad-except
            raw = job.job_def.agent._system_prompt or ''  # pylint: disable=protected-access
            return raw, f'{type(exc).__name__}: {exc}'
    except Exception:  # pylint: disable=broad-except
        return '', None


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

    system_prompt, system_prompt_error = _get_system_prompt(job)
    initial_prompt = ''
    try:
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

                with ui.row().classes('items-center gap-1 shrink-0'):
                    def _download_md(
                        _uuid=uuid_str, _status=status_str, _agent=agent_role,
                        _model=model, _cost=total_cost, _turns=num_turns,
                        _errors=exception_count, _sys=system_prompt, _init=initial_prompt,
                        _job=job, _wb=warmup_blocks, _wr=warmup_ranges, _tr=turn_ranges,
                    ):
                        md = _build_md_content(
                            uuid_str=_uuid, status_str=_status, agent_role=_agent,
                            model=_model, total_cost=_cost, num_turns=_turns,
                            exception_count=_errors, system_prompt=_sys,
                            initial_prompt=_init, job=_job,
                            warmup_blocks=_wb, warmup_ranges=_wr, turn_ranges=_tr,
                        )
                        ui.download(md.encode(), filename=f'job_{_uuid}.md', media_type='text/markdown')

                    def _download_pdf(
                        _uuid=uuid_str, _status=status_str, _agent=agent_role,
                        _model=model, _cost=total_cost, _turns=num_turns,
                        _errors=exception_count, _sys=system_prompt, _init=initial_prompt,
                        _job=job, _wb=warmup_blocks, _wr=warmup_ranges, _tr=turn_ranges,
                    ):
                        md = _build_md_content(
                            uuid_str=_uuid, status_str=_status, agent_role=_agent,
                            model=_model, total_cost=_cost, num_turns=_turns,
                            exception_count=_errors, system_prompt=_sys,
                            initial_prompt=_init, job=_job,
                            warmup_blocks=_wb, warmup_ranges=_wr, turn_ranges=_tr,
                        )
                        pdf = _build_pdf_bytes(md)
                        ui.download(pdf, filename=f'job_{_uuid}.pdf', media_type='application/pdf')

                    if system_prompt:
                        def _show_system_prompt(_sp=system_prompt, _err=system_prompt_error):
                            with ui.dialog() as sp_dlg:
                                with ui.card().classes('w-full max-w-5xl'):
                                    with ui.row().classes('w-full items-center justify-between mb-2'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('description').classes('text-gray-600')
                                            ui.label('System Prompt').classes(
                                                'text-lg font-bold text-gray-800'
                                            )
                                        ui.button(
                                            icon='close', on_click=sp_dlg.close
                                        ).props('flat round dense')
                                    ui.separator().classes('mb-2')
                                    ui.label(_sp).classes(
                                        'text-sm text-gray-700 whitespace-pre-wrap w-full rounded p-3'
                                    ).style(
                                        'font-family: "JetBrains Mono", monospace;'
                                        ' background: #fdf6e3; max-height: 70vh; overflow-y: auto'
                                    )
                                    if _err:
                                        with ui.row().classes('items-center gap-2 mt-3 px-3 py-2 rounded').style(
                                            'background: #fff0f0; border: 1px solid #ffcdd2'
                                        ):
                                            ui.icon('warning').classes('text-amber-600 text-sm')
                                            ui.label(
                                                f'Showing raw template — rendering failed: {_err}'
                                            ).classes('text-xs text-red-700 font-mono')
                            sp_dlg.open()

                        ui.button('System Prompt', icon='description', on_click=_show_system_prompt).props(
                            'flat dense no-caps'
                        ).classes('text-xs text-gray-600').tooltip('View system prompt')
                    ui.button('MD', icon='download', on_click=_download_md).props(
                        'flat dense no-caps'
                    ).classes('text-xs text-indigo-600').tooltip('Download as Markdown')
                    ui.button('PDF', icon='picture_as_pdf', on_click=_download_pdf).props(
                        'flat dense no-caps'
                    ).classes('text-xs text-red-600').tooltip('Download as PDF')
                    ui.button(icon='close', on_click=dlg.close).props('flat round dense')

            ui.separator().classes('mb-2')

            # ── Tab bar ─────────────────────────────────────────────────────
            with ui.tabs().classes('mb-3') as tabs:
                tab_log = ui.tab('Execution Log', icon='timeline')
                tab_raw = ui.tab('Raw', icon='data_object')

            with ui.tab_panels(tabs, value=tab_log).classes('w-full'):

                # ── Execution Log tab ────────────────────────────────────────
                with ui.tab_panel(tab_log).classes('px-0'):

                    # ── System Prompt ────────────────────────────────────────
                    if system_prompt:
                        with ui.expansion('System Prompt', icon='description').props('dense').classes(
                            'w-full rounded border border-gray-200 mb-4'
                        ):
                            ui.label(system_prompt).classes(
                                'text-xs text-gray-700 whitespace-pre-wrap w-full rounded p-3'
                            ).style('font-family: "JetBrains Mono", monospace; background: #fdf6e3')

                    # ── Initial Prompt ───────────────────────────────────────
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

                    # ── Warmup Blocks ────────────────────────────────────────
                    if warmup_blocks:
                        with ui.column().classes('w-full gap-3 mb-4'):
                            _render_warmup_section(job, warmup_blocks, warmup_ranges)

                    # ── LLM Turns ────────────────────────────────────────────
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

                # ── Raw tab ──────────────────────────────────────────────────
                with ui.tab_panel(tab_raw).classes('px-0'):
                    raw_repr = _build_raw_repr(job)
                    raw_html = raw_repr.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    import re  # pylint: disable=import-outside-toplevel
                    raw_html = re.sub(
                        r'\(Error[^)]*\)',
                        lambda m: f'<span style="color: #b71c1c; font-weight: 600">{m.group()}</span>',
                        raw_html,
                    )
                    ui.html(
                        f'<pre style="'
                        f'white-space: pre-wrap; word-break: break-word; overflow-wrap: break-word;'
                        f'font-family: \'JetBrains Mono\', monospace;'
                        f'font-size: 0.75rem; line-height: 1.6;'
                        f'color: #263238; background: #f5f5f5;'
                        f'border: 1px solid #e0e0e0; border-radius: 6px;'
                        f'padding: 16px; margin: 0; width: 100%; box-sizing: border-box;'
                        f'">{raw_html}</pre>'
                    ).classes('w-full')

    dlg.open()
