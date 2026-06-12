"""Agents list page for the Statek web UI."""

import traceback
from typing import Callable, Optional

import dbzero as db0

from statek.agents.agent import SupervisedAgent
from statek.prompt_config import format_system_prompt
from statek.task_difficulty import TaskDifficulty
from ..nicegui_compat import ui
from ..model_bindings import get_all_agents


def _format_warmup_code(warmup_code) -> Optional[str]:
    """Format warmup_code for display as a readable string.

    Returns None if warmup_code is empty/None.
    """
    if warmup_code is None:
        return None
    if isinstance(warmup_code, str):
        return warmup_code or None
    if hasattr(warmup_code, 'code'):
        return _format_code_block(warmup_code)
    # Sequence of str/CodeBlock
    if not warmup_code:
        return None
    parts = []
    for i, block in enumerate(warmup_code, 1):
        header = f"# --- Block {i} ---"
        if hasattr(block, 'code'):
            parts.append(f"{header}\n{_format_code_block(block)}")
        else:
            parts.append(f"{header}\n{block}")
    return "\n\n".join(parts)


def _format_code_block(cb) -> str:
    """Format a single CodeBlock for display."""
    parts = []
    if cb.code:
        parts.append(cb.code)
    if cb.tool_calls:
        for tc in cb.tool_calls:
            parts.append(f"{tc.format()}  #STATEK: as tool")
    return "\n".join(parts) if parts else ""


def _get_tool_info(fn) -> tuple[str | None, str | None, str | None]:
    """Return (brief_formatted, full_docs, error) for a tool callable.

    Both outputs use Python syntax formatting. brief_formatted uses brief=True,
    full_docs uses brief=False. On error, brief and full_docs are None and error
    contains the full traceback string.
    """
    if not callable(fn):
        return None, None, None
    try:
        from statek.docstring import parse_docstring, format_docstring  # pylint: disable=import-outside-toplevel
        parsed = parse_docstring(fn)
        brief = format_docstring(parsed, brief=True, py_syntax=True)
        full_docs = format_docstring(parsed, brief=False, py_syntax=True)
        return brief, full_docs, None
    except Exception:  # pylint: disable=broad-except
        return None, None, traceback.format_exc()


def _get_tool_signature(fn: Callable) -> str:
    """Return 'name(param1, param2)' label string for a tool function."""
    import inspect  # pylint: disable=import-outside-toplevel
    name = getattr(fn, '__name__', str(fn))
    try:
        sig = inspect.signature(fn)
        skip_kinds = {inspect.Parameter.VAR_KEYWORD}
        params = [
            p_name for p_name, p in sig.parameters.items()
            if p_name not in ('self', 'cls') and p.kind not in skip_kinds
        ]
        return f"{name}({', '.join(params)})"
    except (ValueError, TypeError):
        return name


def _get_agent_system_prompt(agent) -> str:
    """Return agent system prompt text safe to pass into NiceGUI components."""
    try:
        raw_prompt = agent._system_prompt  # pylint: disable=protected-access
        if not raw_prompt:
            return ''

        try:
            return agent.system_prompt(TaskDifficulty.medium) or ''
        except Exception:  # pylint: disable=broad-except
            pass

        if isinstance(raw_prompt, str):
            return raw_prompt

        try:
            return format_system_prompt(raw_prompt, TaskDifficulty.medium)
        except Exception:  # pylint: disable=broad-except
            return str(raw_prompt)
    except Exception:  # pylint: disable=broad-except
        return ''


def _render_tool_row(fn: Callable) -> None:
    """Render a single tool function with expandable Brief / Docs tabs."""
    brief, full_docs, error = _get_tool_info(fn)
    label = _get_tool_signature(fn)

    if error:
        with ui.expansion(label, icon='error').props('dense').classes('w-full font-mono text-xs'):
            ui.code(error, language='python').classes('w-full text-xs text-red-700')
    elif brief or full_docs:
        with ui.expansion(label, icon='chevron_right').props('dense').classes('w-full font-mono text-xs'):
            with ui.tabs().props('align=left dense').classes('w-full border-b border-gray-200') as tabs:
                tab_brief = ui.tab('Brief', icon='short_text')
                tab_docs = ui.tab('Docs', icon='description')
            with ui.tab_panels(tabs, value=tab_brief).classes('w-full'):
                with ui.tab_panel(tab_brief):
                    if brief:
                        ui.code(brief, language='python').classes('w-full text-xs')
                    else:
                        ui.label('(no description)').classes('text-xs text-gray-400 italic')
                with ui.tab_panel(tab_docs):
                    if full_docs:
                        ui.code(full_docs, language='python').classes('w-full text-xs')
                    else:
                        ui.label('(no documentation)').classes('text-xs text-gray-400 italic')
    else:
        with ui.row().classes('items-center gap-1'):
            ui.icon('chevron_right').classes('text-xs text-primary')
            ui.label(label).classes('text-xs font-mono text-gray-700')


def _render_agent_card(agent) -> None:
    """Render a single Agent as an info card."""
    try:
        _render_agent_card_content(agent)
    except Exception as exc:  # pylint: disable=broad-except
        try:
            role_label = agent.role
        except Exception:  # pylint: disable=broad-except
            role_label = '<unknown agent>'
        with ui.card().classes('w-full shadow-sm border border-red-300'):
            ui.label(role_label).classes('text-lg font-bold text-gray-900')
            ui.separator()
            ui.label(f'Error rendering agent: {exc}').classes('text-sm text-red-600 font-mono')


def _render_agent_card_content(agent) -> None:
    """Render a single Agent as an info card (implementation)."""
    tools = list(agent._tools)  # pylint: disable=protected-access
    tools_by_name = list(agent._tools_by_name or [])  # pylint: disable=protected-access
    total_tools = len(tools) + len(tools_by_name)

    metadata = agent._metadata or {}  # pylint: disable=protected-access
    system_prompt = _get_agent_system_prompt(agent)

    try:
        uuid_str = str(db0.uuid(agent))
    except Exception:  # pylint: disable=broad-except
        uuid_str = '—'

    with ui.card().classes('w-full shadow-sm hover:shadow-md transition-shadow').style('padding: 8px'):
        with ui.row().classes('w-full items-center justify-between mb-1'):
            with ui.row().classes('items-center gap-2'):
                ui.label(agent.role).classes('text-sm font-bold text-gray-900')
                ui.label(f'({uuid_str})').classes('text-xs text-gray-400 font-mono')

            ui.badge(f'{total_tools} tools', color='primary').classes('text-xs')

        ui.separator().classes('mb-1')

        # Tools
        with ui.expansion('Tools', icon='build').props('dense').classes('w-full'):
            if tools or tools_by_name:
                with ui.column().classes('gap-1 pl-2 w-full'):
                    for fn in tools:
                        _render_tool_row(fn)
                    for name in tools_by_name:
                        fn = (getattr(agent, 'context', None) or {}).get(name)
                        if fn is None:
                            # Dynamic tools are not serialized to DB — force full re-init
                            agent._X__context = None  # pylint: disable=protected-access
                            fn = (getattr(agent, 'context', None) or {}).get(name)
                        if fn is not None:
                            _render_tool_row(fn)
                        else:
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('chevron_right').classes('text-sm text-primary')
                                ui.label(name).classes('text-sm font-mono text-gray-700')
            else:
                ui.label('No tools assigned.').classes('text-sm text-gray-500 italic')

        # Metadata
        if metadata:
            with ui.expansion('Metadata', icon='info').props('dense').classes('w-full'):
                with ui.column().classes('gap-1 pl-2'):
                    for key, value in metadata.items():
                        with ui.row().classes('items-start gap-2'):
                            ui.label(f'{key}:').classes('text-xs font-semibold text-gray-600 w-32 shrink-0')
                            ui.label(str(value)).classes('text-xs text-gray-700 font-mono break-all')

        # Warmup Code
        if isinstance(agent, SupervisedAgent) and agent.warmup_def is not None:
            formatted = _format_warmup_code(agent.warmup_def.warmup_code)
            if formatted:
                with ui.expansion('Warmup Code', icon='play_arrow').props('dense').classes('w-full'):
                    ui.code(formatted, language='python').classes('w-full text-xs')

        # System prompt
        if system_prompt:
            with ui.expansion('System Prompt', icon='description').props('dense').classes('w-full'):
                ui.label(system_prompt).classes(
                    'text-xs text-gray-700 whitespace-pre-wrap w-full rounded p-2'
                ).style('font-family: "JetBrains Mono", monospace; background-color: #fdf6e3')


def create_agents_page() -> None:
    """Render the Agents list page."""
    agents = list(get_all_agents())

    with ui.row().classes('w-full items-center justify-between mb-4'):
        ui.label('Agents').classes('text-2xl font-bold text-gray-900')
        ui.badge(str(len(agents)), color='primary').classes('text-sm')

    if not agents:
        with ui.card().classes('w-full'):
            ui.label('No agents found in the database.').classes('text-gray-500 italic')
        return

    agents_sorted = sorted(agents, key=lambda a: a.role)
    with ui.column().classes('w-full gap-2'):
        for agent in agents_sorted:
            _render_agent_card(agent)
