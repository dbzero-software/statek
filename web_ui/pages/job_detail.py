"""Job detail view for the Statek web UI."""

import io
import inspect
import json
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import dbzero as db0

from statek.chat_history import ChatRole, ContentSource
from statek.chat_style import ChatStyle
from statek.llm_api import LLM_API, select_request_tools
from statek.locale import LANGUAGE_HINTS
from statek.model_pricing import get_model_pricing
from statek.prompt_config import format_system_prompt
from statek.task_difficulty import TaskDifficulty
from statek.utils import _is_hidden_warmup_block
from statek.executors.chat_log_item import (
    LLM_LogItem,
    ReminderLogItem,
    ToolError,
    WarmupLogItem,
)
from statek.model_name import ensure_model_name, format_model_for_provider, select_model_provider
from ..nicegui_compat import ui
from ..components.json_viewer import create_json_viewer
from ..components.status_badge import create_status_badge


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


def _get_warmup_log_items(job) -> dict:
    """Return dict mapping warmup_block_num -> WarmupLogItem from job.chat_log."""
    if not hasattr(job, 'chat_log') or not job.chat_log:
        return {}
    return {
        item.warmup_block_num: item
        for item in job.chat_log
        if isinstance(item, WarmupLogItem)
    }


def _get_warmup_console_ranges(job) -> list[tuple[int, int]]:
    """Return list of (from_pos, to_pos) console ranges for each warmup block."""
    blocks = _get_warmup_blocks(job)
    if not blocks:
        return []

    n = len(blocks)
    positions = job._warmup_end_positions()  # recorded after each block completes
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


def _get_llm_items(job) -> list:
    """Return only LLM_LogItem entries from job.chat_log (filter out WarmupLogItems)."""
    if not job.chat_log:
        return []
    return [item for item in job.chat_log if isinstance(item, LLM_LogItem)]


def _get_reminder_texts(job) -> set[str]:
    """Return reminder texts recorded in the job log."""
    if not getattr(job, 'chat_log', None):
        return set()
    texts = set()
    for item in job.chat_log:
        if not isinstance(item, ReminderLogItem):
            continue
        text = getattr(getattr(item, 'reminder', None), 'text', None)
        if text:
            texts.add(str(text))
    return texts


def _get_turn_console_ranges(job) -> list[tuple[int, int]]:
    """Return list of (from_pos, to_pos) console ranges for each LLM turn in chat_log.

    Each chat_log item records console_pos as the console length *before* that turn's
    code ran (i.e. the start of its output). The end is the next turn's console_pos,
    or the full console length for the last turn.
    """
    llm_items = _get_llm_items(job)
    if not llm_items:
        return []

    console_len = len(job.py_env.console) if job.py_env.console else 0

    ranges = []
    for i, item in enumerate(llm_items):
        from_pos = item.console_pos
        if i + 1 < len(llm_items):
            to_pos = llm_items[i + 1].console_pos
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


def _get_locale_str(job) -> str:
    """Return a compact locale label for the job, or an empty string."""
    try:
        if not job.job_def:
            return ''
        locale = job.job_def.locale
        if locale is None:
            return ''
        lang_code = getattr(locale, 'lang_code', None)
        country_code = getattr(locale, 'country_code', None)
        if lang_code and country_code:
            return f'{lang_code}-{country_code}'
        if lang_code:
            return str(lang_code)
        if country_code:
            return str(country_code)
    except Exception:  # pylint: disable=broad-except
        return ''
    return ''


def _get_job_model(job) -> str:
    """Return the concrete model for the job's current difficulty."""
    try:
        if not job.job_def:
            return '—'
        model = job.get_current_model()
        if not model:
            return '—'
        return ensure_model_name(model).model or '—'
    except Exception:  # pylint: disable=broad-except
        return '—'


def _get_job_temperature(job) -> str:
    """Return explicit job temperature override, or an empty string when defaulted."""
    try:
        if not job.job_def:
            return ''
        metadata = job.job_def.metadata or {}
        temperature = metadata.get('TEMPERATURE')
        return str(temperature) if temperature is not None else ''
    except Exception:  # pylint: disable=broad-except
        return ''


def _get_job_provider(job) -> str:
    """Return the effective job provider override, or an empty string when defaulted."""
    try:
        if not job.job_def:
            return ''
        metadata = job.job_def.metadata or {}
        model = metadata.get('MODEL')
        get_current_model = getattr(job, 'get_current_model', None)
        if callable(get_current_model):
            current_model = get_current_model()
            if isinstance(current_model, str) and current_model.strip():
                model = current_model
        provider = select_model_provider(
            model,
            default_provider=metadata.get('PROVIDER'),
        )
        return str(provider) if provider else ''
    except Exception:  # pylint: disable=broad-except
        return ''


def _job_uses_reasoning(job) -> bool:
    """Return whether the job explicitly enabled reasoning."""
    try:
        if not job.job_def:
            return False
        metadata = job.job_def.metadata or {}
        value = metadata.get('REASONING')
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}
    except Exception:  # pylint: disable=broad-except
        return False


def _calculate_usage_cost(usage, pricing) -> Optional[float]:
    """Calculate usage cost from token counters and a valid pricing object."""
    if pricing is None or not getattr(pricing, 'is_valid', False):
        return None
    input_tokens = getattr(usage, 'total_input_tokens', 0) or 0
    output_tokens = getattr(usage, 'total_output_tokens', 0) or 0
    if input_tokens + output_tokens <= 0:
        return None
    cached_tokens = getattr(usage, 'total_cached_tokens', 0) or 0
    non_cached = input_tokens - cached_tokens
    cached_price = (
        pricing.input_price_per_cached_M
        if pricing.input_price_per_cached_M is not None
        else pricing.input_price_per_M
    )
    return float(
        (Decimal(non_cached) * pricing.input_price_per_M
         + Decimal(cached_tokens) * cached_price
         + Decimal(output_tokens) * pricing.output_price_per_M)
        / Decimal("1000000")
    )


def _get_effective_job_cost(job) -> float:
    """Return cost for a job, re-pricing stale usage objects when possible."""
    usage = getattr(job, 'usage', None)
    if usage is None:
        return 0.0

    total_cost = getattr(usage, 'total_cost', None)
    if total_cost:
        return float(total_cost)

    try:
        if not job.job_def:
            return float(total_cost or 0.0)
        metadata = job.job_def.metadata or {}
        raw_model = job.get_current_model()
        provider = select_model_provider(
            raw_model,
            default_provider=metadata.get("PROVIDER"),
        )
        model_name = ensure_model_name(raw_model)
        model = format_model_for_provider(raw_model, provider) if provider else model_name.model
        provider_key = provider.upper() if provider else None
        model_family = model_name.model_family if provider_key == "OPENROUTER" else None
        pricing = get_model_pricing(provider or "UNKNOWN", model or "", model_family, no_create=True)
        calculated = _calculate_usage_cost(usage, pricing)
        return float(calculated or total_cost or 0.0)
    except Exception:  # pylint: disable=broad-except
        return float(total_cost or 0.0)


def _get_tool_data_for_block(code_block, chat_log_item) -> list:
    """Return [(call_spec, result_str, error_str|None), ...] for a code block's tool calls.

    Args:
        code_block: str or CodeBlock-like; tool calls come from code_block.tool_calls
        chat_log_item: ChatLogItem instance (or duck-typed equivalent) with get_tool_result()

    Returns:
        List of (call_spec, result_str, error_or_none) triples. Empty if no tool calls
        or no log entry.
    """
    if not hasattr(code_block, 'tool_calls') or not code_block.tool_calls:
        return []
    if chat_log_item is None:
        return []
    call_specs = list(code_block.tool_calls)
    result = []
    for i, cs in enumerate(call_specs):
        try:
            entry = chat_log_item.get_tool_result(i)
        except (KeyError, AttributeError):
            return []
        except IndexError:
            entry = ''
        if isinstance(entry, ToolError):
            result.append((cs, '', entry.err_message))
        else:
            result.append((cs, entry, None))
    return result


def _get_reported_tools(job) -> list:
    """Return the tool callables that would be sent in this job's LLM request."""
    try:
        if not job.job_def:
            return []
        metadata = dict(job.job_def.metadata or {})
        agent = getattr(job.job_def, 'agent', None)
        available_tools = getattr(agent, 'all_tools', None)
        chat_style = getattr(job.job_def, 'chat_style', None)
        return list(select_request_tools(
            metadata=metadata,
            available_tools=available_tools,
            chat_style=chat_style,
        ) or [])
    except Exception:  # pylint: disable=broad-except
        return []


def _format_reported_tool_label(fn) -> str:
    """Return a compact signature label for a reported tool."""
    name = getattr(fn, '__name__', str(fn))
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return name

    params = [
        param.name
        for param in sig.parameters.values()
        if param.name not in ('self', 'cls')
        and param.kind != inspect.Parameter.VAR_KEYWORD
    ]
    return f'{name}({", ".join(params)})'


def _strip_language_hint_suffix(content: str) -> str:
    """Remove display-only locale hints appended to user messages."""
    if not content:
        return content

    suffixes = tuple(f' ({hint})' for hint in LANGUAGE_HINTS.values())
    stripped_lines = []
    for line in content.splitlines():
        suffix = next((item for item in suffixes if line.endswith(item)), None)
        stripped_lines.append(line[:-len(suffix)] if suffix else line)
    return '\n'.join(stripped_lines)


@dataclass
class _HistoryMessage:
    """A normalized message displayed inside or outside an assistant section."""

    title: str
    content: str
    content_src: Optional[ContentSource] = None


@dataclass
class _HistorySection:
    """A normalized job-detail section derived from ``Job.get_chat_history()``."""

    kind: str
    title: str
    content: str = ''
    content_src: Optional[ContentSource] = None
    render_as_code: bool = False
    tool_data: list = field(default_factory=list)
    followups: list[_HistoryMessage] = field(default_factory=list)
    warmup_num: Optional[int] = None
    warmup_total: Optional[int] = None
    turn_num: Optional[int] = None
    hidden: bool = False


def _warmup_section_title(index: int, hidden: bool) -> str:
    """Return the display title for a warmup block."""
    title = f'Warmup Code {index}'
    return f'{title} (Hidden)' if hidden else title


def _build_warmup_display_sections(job, blocks: list) -> list[_HistorySection]:
    """Build display-only warmup sections from raw job warmup blocks."""
    ranges = _get_warmup_console_ranges(job)
    warmup_items = _get_warmup_log_items(job)
    total = len(blocks)
    sections = []
    for i, block in enumerate(blocks):
        hidden = _is_hidden_warmup_block(block)
        from_pos, to_pos = ranges[i] if i < len(ranges) else (0, 0)
        section = _HistorySection(
            kind='assistant',
            title=_warmup_section_title(i + 1, hidden),
            content=_get_code_str(block),
            content_src=ContentSource.SYSTEM,
            render_as_code=True,
            tool_data=_get_tool_data_for_block(block, warmup_items.get(i)),
            warmup_num=i + 1,
            warmup_total=total,
            hidden=hidden,
        )
        console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
        if console_out.strip():
            section.followups.append(_HistoryMessage(
                title='Console Output',
                content=console_out,
                content_src=ContentSource.CONSOLE,
            ))
        sections.append(section)
    return sections


@dataclass
class _LatencySummary:
    """Aggregated latency statistics for one latency series."""

    sample_count: int = 0
    completed_count: int = 0
    pending_count: int = 0
    min_seconds: Optional[float] = None
    avg_seconds: Optional[float] = None
    max_seconds: Optional[float] = None


def _get_history_items(job) -> list:
    """Return normalized chat history items for job details."""
    getter = getattr(job, 'get_chat_history', None)
    if not callable(getter):
        return []
    try:
        return [item for item in getter() if item is not None]
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return []


def _message_title(item) -> str:
    """Return a display label for a non-assistant chat history item."""
    if item.role == ChatRole.SYSTEM:
        return 'Reminder'
    if item.role == ChatRole.TOOL:
        return 'Tool Result'
    if item.content_src == ContentSource.CONSOLE:
        return 'Console Output'
    return 'User Message'


def _build_history_sections(job) -> list[_HistorySection]:
    """Group ``Job.get_chat_history()`` into job-detail display sections."""
    sections: list[_HistorySection] = []
    current: Optional[_HistorySection] = None
    warmup_blocks = _get_warmup_blocks(job)
    expected_warmups = len(warmup_blocks)
    has_hidden_warmups = any(_is_hidden_warmup_block(block) for block in warmup_blocks)
    warmup_sections = (
        _build_warmup_display_sections(job, warmup_blocks)
        if has_hidden_warmups else []
    )
    warmup_sections_inserted = False
    warmup_num = 0
    turn_num = 0
    chat_style = getattr(getattr(job, 'job_def', None), 'chat_style', None)
    is_direct = chat_style == ChatStyle.DIRECT  # pylint: disable=no-member
    reminder_texts = _get_reminder_texts(job)
    skipping_warmup_history = False

    def _flush_current() -> None:
        nonlocal current
        if current is not None:
            sections.append(current)
            current = None

    def _insert_warmup_sections() -> None:
        nonlocal warmup_sections_inserted
        if warmup_sections_inserted or not warmup_sections:
            return
        _flush_current()
        sections.extend(warmup_sections)
        warmup_sections_inserted = True

    for item in _get_history_items(job):
        if item.role == ChatRole.SYSTEM:
            content = item.content or ''
            if content in reminder_texts:
                _flush_current()
                sections.append(
                    _HistorySection(
                        kind='message',
                        title='Reminder',
                        content=content,
                        content_src=item.content_src,
                    )
                )
            continue

        if item.role == ChatRole.ASSISTANT:
            _flush_current()
            is_warmup = (
                item.content_src == ContentSource.SYSTEM
                or (item.content_src is None and warmup_num < expected_warmups)
            )
            if is_warmup:
                warmup_num += 1
                if has_hidden_warmups:
                    _insert_warmup_sections()
                    skipping_warmup_history = True
                    continue
                title = _warmup_section_title(warmup_num, False)
                content_src = ContentSource.SYSTEM
                render_as_code = True
            else:
                skipping_warmup_history = False
                if has_hidden_warmups:
                    _insert_warmup_sections()
                turn_num += 1
                title = f'Turn {turn_num}'
                content_src = item.content_src
                render_as_code = not is_direct
            tool_data = []
            content = item.content or ''
            if item.tool_calls:
                tool_calls_list = list(item.tool_calls)
                if is_warmup and not content and len(tool_calls_list) == 1:
                    cs0 = tool_calls_list[0]
                    if cs0.func_name == 'python_cli' and cs0.kwargs and 'code' in cs0.kwargs:
                        content = cs0.kwargs['code']
                        tool_calls_list = []
                tool_data = [(cs, '', None) for cs in tool_calls_list]
            current = _HistorySection(
                kind='assistant',
                title=title,
                content=content,
                content_src=content_src,
                render_as_code=render_as_code,
                tool_data=tool_data,
                warmup_num=warmup_num if is_warmup else None,
                warmup_total=expected_warmups if is_warmup else None,
                turn_num=turn_num - 1 if not is_warmup else None,
            )
            continue

        if (
            skipping_warmup_history
            and item.role in (ChatRole.TOOL, ChatRole.USER)
            and item.content_src == ContentSource.CONSOLE
        ):
            continue
        skipping_warmup_history = False

        target = current.followups if current is not None else None

        if item.role == ChatRole.TOOL and current is not None and current.tool_data:
            next_idx = next(
                (idx for idx, (_, result, _err) in enumerate(current.tool_data) if result == ''),
                None,
            )
            if next_idx is not None:
                cs, _, err = current.tool_data[next_idx]
                current.tool_data[next_idx] = (cs, item.content or '', err)
                continue

        content = item.content or ''
        if item.content_src == ContentSource.USER:
            content = _strip_language_hint_suffix(content)
        message = _HistoryMessage(
            title=_message_title(item),
            content=content,
            content_src=item.content_src,
        )
        if target is not None:
            target.append(message)
        else:
            sections.append(
                _HistorySection(
                    kind='message',
                    title=message.title,
                    content=message.content,
                    content_src=message.content_src,
                )
            )

    if has_hidden_warmups:
        _insert_warmup_sections()
    _flush_current()
    return sections


def _get_exception_messages(job) -> list[str]:
    """Return code-execution exception messages only (py_env.exceptions)."""
    exceptions = getattr(getattr(job, 'py_env', None), 'exceptions', None) or {}
    return [str(msg) for _, msg in sorted(exceptions.items()) if msg]


def _get_latency_samples(job, getter_name: str) -> list[tuple[datetime, Optional[float]]]:
    """Safely call a latency getter on ``job`` and normalize the result to a list."""
    getter = getattr(job, getter_name, None)
    if not callable(getter):
        return []
    try:
        return list(getter())
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return []


def _summarize_latencies(
    samples: list[tuple[datetime, Optional[float]]]
) -> _LatencySummary:
    """Return compact summary stats for a series of latency samples."""
    completed = [seconds for _, seconds in samples if seconds is not None]
    sample_count = len(samples)
    completed_count = len(completed)
    pending_count = sample_count - completed_count
    if not completed:
        return _LatencySummary(
            sample_count=sample_count,
            completed_count=completed_count,
            pending_count=pending_count,
        )
    return _LatencySummary(
        sample_count=sample_count,
        completed_count=completed_count,
        pending_count=pending_count,
        min_seconds=min(completed),
        avg_seconds=sum(completed) / completed_count,
        max_seconds=max(completed),
    )


def _format_latency_seconds(seconds: Optional[float]) -> str:
    """Render a latency measurement for display."""
    if seconds is None:
        return 'Pending'
    return f'{seconds:.2f}s'


def _format_latency_timestamp(timestamp: Optional[datetime]) -> str:
    """Render a latency-series timestamp in a compact UTC-agnostic form."""
    if timestamp is None:
        return '—'
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------------------
# Raw repr helper (pure — no NiceGUI dependency)
# ---------------------------------------------------------------------------

def _build_raw_data(job):
    """Return a JSON-like structure of all Job attributes for raw inspection."""
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
            if hasattr(obj, 'items'):
                result = {}
                for k, v in obj.items():
                    try:
                        result[str(k)] = _to_display(v, depth + 1)
                    except Exception as exc:  # pylint: disable=broad-except
                        result[str(k)] = f'(Error: {exc})'
                return result
            if hasattr(obj, '__iter__'):
                converted = []
                for x in obj:
                    try:
                        converted.append(_to_display(x, depth + 1))
                    except Exception as exc:  # pylint: disable=broad-except
                        converted.append(f'(Error: {exc})')
                return converted
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
        try:
            type_name = db0.get_type(obj).__name__
        except Exception:  # pylint: disable=broad-except
            type_name = type(obj).__name__
        return {
            '__type__': type_name,
            **converted_attrs,
        }

    return _to_display(job)


def _build_step_preview_data(job, turn_num: int):
    """Return JSON-viewer-friendly data for a historical LLM request."""
    request_data = dict(job.get_request_data(turn_num))
    chat_history = request_data.get('chat_history')
    if chat_history is not None:
        request_data['chat_history'] = list(chat_history)
    provider = _get_job_provider(job) or None
    payload = LLM_API.get(provider_name=provider).preview_request(**request_data)
    return _build_raw_data(payload)


def _expand_json_viewer(editor) -> None:
    """Expand all nodes in a NiceGUI JSON editor tree."""
    editor.run_editor_method(':expand', [], '() => true')


def _collapse_json_viewer(editor) -> None:
    """Collapse all nodes in a NiceGUI JSON editor tree."""
    editor.run_editor_method('collapse', [], True)


def _json_viewer_expand_js(viewer) -> str:
    """Return a click handler that expands all nodes on the client."""
    return (
        f'() => {{'
        f' const component = getElement({viewer.id});'
        f' const host = component?.$el;'
        f" const prefix = '[statek-json-viewer]';"
        f' if (!host) {{ console.warn(prefix, "expand_all: host not found", {viewer.id}); return; }}'
        f" const selector = '.jse-json-node:not(.jse-expanded) > .jse-header-outer > .jse-header > .jse-expand';"
        f' const clickRecursive = (button) => button.dispatchEvent(new MouseEvent("click", {{'
        f'   bubbles: true, cancelable: true, ctrlKey: true, metaKey: true'
        f' }}));'
        f' let pass = 0;'
        f' const step = () => {{'
        f'   const buttons = Array.from(host.querySelectorAll(selector));'
        f'   console.info(prefix, "expand_all pass", pass, "buttons", buttons.length, "viewer", {viewer.id});'
        f'   if (!buttons.length) return;'
        f'   if (pass > 100) {{ console.warn(prefix, "expand_all aborted after too many passes", {viewer.id}); return; }}'
        f'   pass += 1;'
        f'   buttons.forEach(clickRecursive);'
        f'   requestAnimationFrame(step);'
        f' }};'
        f' step();'
        f' }}'
    )


def _json_viewer_collapse_js(viewer) -> str:
    """Return a click handler that collapses all nodes on the client."""
    return (
        f'() => {{'
        f' const component = getElement({viewer.id});'
        f' const host = component?.$el;'
        f" const prefix = '[statek-json-viewer]';"
        f' if (!host) {{ console.warn(prefix, "collapse_all: host not found", {viewer.id}); return; }}'
        f" const selector = '.jse-json-node.jse-root.jse-expanded > .jse-header-outer > .jse-header > .jse-expand';"
        f' const button = host.querySelector(selector);'
        f' console.info(prefix, "collapse_all root button", !!button, "viewer", {viewer.id});'
        f' if (!button) return;'
        f' button.dispatchEvent(new MouseEvent("click", {{'
        f'   bubbles: true, cancelable: true, ctrlKey: true, metaKey: true'
        f' }}));'
        f' }}'
    )


def _build_raw_repr(job) -> str:
    """Return a pprint-formatted string of all Job attributes for raw inspection."""
    import pprint  # pylint: disable=import-outside-toplevel
    try:
        return pprint.pformat(_build_raw_data(job), width=120, depth=10, sort_dicts=False)
    except Exception as exc:  # pylint: disable=broad-except
        return f'(Error building raw repr: {exc})'


_RAW_LIST_EVEN_BG = '#f0f0f0'
_RAW_LIST_ODD_BG = '#e0e0e0'


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _build_raw_html(job) -> str:
    """Return an HTML string of all Job attributes for raw inspection.

    List items are rendered with alternating background colours for readability.
    """
    import re as _re  # pylint: disable=import-outside-toplevel

    def _render_value(val, indent=0):
        """Recursively render a _to_display value to HTML lines."""
        pad = '  ' * indent
        if val is None:
            return f'{pad}<span style="color:#78909c">None</span>'
        if isinstance(val, bool):
            return f'{pad}<span style="color:#78909c">{val}</span>'
        if isinstance(val, (int, float)):
            return f'{pad}<span style="color:#1565c0">{val}</span>'
        if isinstance(val, str):
            if '\n' in val:
                # Multiline: show actual newlines with quotes around the block
                text = _esc(val)
                text = _re.sub(
                    r'\(Error[^)]*\)',
                    lambda m: f'<span style="color:#b71c1c;font-weight:600">{m.group()}</span>',
                    text,
                )
                return f'{pad}<span style="color:#2e7d32">\'{text}\'</span>'
            text = _esc(repr(val))
            # Highlight errors
            text = _re.sub(
                r'\(Error[^)]*\)',
                lambda m: f'<span style="color:#b71c1c;font-weight:600">{m.group()}</span>',
                text,
            )
            return f'{pad}<span style="color:#2e7d32">{text}</span>'
        if isinstance(val, (list, tuple)):
            if not val:
                return f'{pad}{"[]" if isinstance(val, list) else "()"}'
            bracket_open = '[' if isinstance(val, list) else '('
            bracket_close = ']' if isinstance(val, list) else ')'
            parts = [f'{pad}{bracket_open}']
            for i, item in enumerate(val):
                bg = _RAW_LIST_EVEN_BG if i % 2 == 0 else _RAW_LIST_ODD_BG
                item_html = _render_value(item, indent + 1)
                comma = ',' if i < len(val) - 1 else ''
                idx_label = f'<span style="color:#78909c;font-size:0.9em">{i}:</span> '
                parts.append(
                    f'<span style="background:{bg};display:block;'
                    f'padding:1px 4px;border-radius:3px;margin:0">'
                    f'{idx_label}{item_html}{comma}</span>'
                )
            parts.append(f'{pad}{bracket_close}')
            return ''.join(parts)
        if isinstance(val, dict):
            type_name = val.get('__type__')
            entries = [(k, v) for k, v in val.items() if k != '__type__']
            if type_name:
                if not entries:
                    return f'{pad}<span style="color:#6a1b9a;font-weight:600">{_esc(type_name)}</span>()'
                parts = [f'{pad}<span style="color:#6a1b9a;font-weight:600">{_esc(type_name)}</span>:']
            else:
                if not entries:
                    return f'{pad}{{}}'
                parts = [f'{pad}{{']
            for k, v in entries:
                val_html = _render_value(v, indent + 1).lstrip()
                parts.append(f'{"  " * (indent + 1)}<span style="color:#37474f;font-weight:600">{_esc(k)}</span>: {val_html}')
            if not type_name:
                parts.append(f'{pad}}}')
            return '\n'.join(parts)
        return f'{pad}{_esc(repr(val))}'

    try:
        body = _render_value(_build_raw_data(job))
    except Exception as exc:  # pylint: disable=broad-except
        body = _esc(f'(Error building raw repr: {exc})')
    return body


def _open_raw_json_dialog(job) -> None:
    """Open a full-screen JSON viewer for the raw job structure."""
    raw_data = _build_raw_data(job)
    raw_text_js = json.dumps(_build_raw_repr(job))

    with ui.dialog().props('maximized') as dlg:
        with ui.card().classes('w-full h-full rounded-none overflow-hidden').style('max-height: 100vh'):
            action_row = None
            with ui.row().classes('w-full items-center justify-between px-4 py-3 border-b border-gray-200'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('data_object').classes('text-indigo-600')
                    ui.label('Raw JSON').classes('text-lg font-semibold text-gray-900')
                action_row = ui.row().classes('items-center gap-1')

            with ui.column().classes('w-full flex-1 p-4'):
                viewer = create_json_viewer(raw_data, height='calc(100vh - 110px)')
            expand_js = _json_viewer_expand_js(viewer)
            collapse_js = _json_viewer_collapse_js(viewer)

            with action_row:
                ui.button(
                    'Expand All',
                    icon='unfold_more',
                ).props('flat dense no-caps').classes('text-gray-700').on(
                    'click',
                    js_handler=expand_js,
                )
                ui.button(icon='content_copy').props(
                    'flat dense round'
                ).classes('text-emerald-700').on(
                    'click',
                    js_handler=f'() => {{ navigator.clipboard.writeText({raw_text_js}); }}',
                ).tooltip('Copy raw text to clipboard')
                ui.button(
                    'Collapse All',
                    icon='unfold_less',
                ).props('flat dense no-caps').classes('text-gray-700').on(
                    'click',
                    js_handler=collapse_js,
                )
                ui.button(icon='close', on_click=dlg.close).props('flat round dense')

    dlg.open()


def _open_step_preview_dialog(job, turn_num: int, title: str) -> None:
    """Open a JSON viewer for the reconstructed request of one LLM turn."""
    preview_data = _build_step_preview_data(job, turn_num)

    with ui.dialog().props('maximized') as dlg:
        with ui.card().classes('w-full h-full rounded-none overflow-hidden').style('max-height: 100vh'):
            action_row = None
            with ui.row().classes('w-full items-center justify-between px-4 py-3 border-b border-gray-200'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('preview').classes('text-indigo-600')
                    ui.label(f'{title} Request JSON').classes('text-lg font-semibold text-gray-900')
                action_row = ui.row().classes('items-center gap-1')

            with ui.column().classes('w-full flex-1 p-4'):
                viewer = create_json_viewer(preview_data, height='calc(100vh - 110px)')
            expand_js = _json_viewer_expand_js(viewer)
            collapse_js = _json_viewer_collapse_js(viewer)

            with action_row:
                ui.button(
                    'Expand All',
                    icon='unfold_more',
                ).props('flat dense no-caps').classes('text-gray-700').on(
                    'click',
                    js_handler=expand_js,
                )
                ui.button(
                    'Collapse All',
                    icon='unfold_less',
                ).props('flat dense no-caps').classes('text-gray-700').on(
                    'click',
                    js_handler=collapse_js,
                )
                ui.button(icon='close', on_click=dlg.close).props('flat round dense')

    dlg.open()


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
    chat_style: str,
    system_prompt: str,
    job,
) -> str:
    """Build a markdown document representing the full job execution log."""
    parts: list[str] = []
    sections = _build_history_sections(job)
    errors = _get_exception_messages(job)
    locale_str = _get_locale_str(job)
    temperature_str = _get_job_temperature(job)
    provider_str = _get_job_provider(job)
    uses_reasoning = _job_uses_reasoning(job)

    # ── Header ──────────────────────────────────────────────────────────────
    parts.append('# Job Detail\n')
    parts.append('| Field | Value |')
    parts.append('|---|---|')
    parts.append(f'| **Status** | {_escape_md(status_str)} |')
    parts.append(f'| **UUID** | `{_escape_md(uuid_str)}` |')
    parts.append(f'| **Agent** | {_escape_md(agent_role)} |')
    parts.append(f'| **Model** | `{_escape_md(model)}` |')
    if provider_str:
        parts.append(f'| **Provider** | `{_escape_md(provider_str)}` |')
    if locale_str:
        parts.append(f'| **Locale** | `{_escape_md(locale_str)}` |')
    if temperature_str:
        parts.append(f'| **Temperature** | `{_escape_md(temperature_str)}` |')
    if uses_reasoning:
        parts.append('| **Reasoning** | Enabled |')
    parts.append(f'| **Cost** | `${total_cost:.4f}` |')
    parts.append(f'| **Turns** | {num_turns} |')
    if chat_style:
        parts.append(f'| **Chat Style** | `{_escape_md(chat_style)}` |')
    if exception_count:
        parts.append(f'| **Errors** | {exception_count} |')
    parts.append('')

    # ── System Prompt ────────────────────────────────────────────────────────
    if system_prompt:
        parts.append('---\n')
        parts.append('## System Prompt\n')
        parts.append(_md_code_fence(system_prompt))
        parts.append('')

    # ── History ──────────────────────────────────────────────────────────────
    if sections:
        parts.append('---\n')
        parts.append('## History\n')
        for section in sections:
            parts.append(f'### {section.title}\n')
            if section.kind == 'assistant':
                if section.render_as_code:
                    parts.append(_md_code_fence(section.content or '(empty)', 'python'))
                elif section.content.strip():
                    parts.append(section.content)
                else:
                    parts.append('(empty)')
                if section.tool_data:
                    parts.append(f'\n**Tool Call{"s" if len(section.tool_data) != 1 else ""}**\n')
                    for entry in section.tool_data:
                        cs, result = entry[0], entry[1]
                        error = entry[2] if len(entry) > 2 else None
                        parts.append(f'- **`{cs.format()}`**')
                        if error:
                            parts.append(f'  **Error:** `{_escape_md(str(error).strip())}`')
                        if result:
                            parts.append(_md_code_fence(str(result).strip()))
                for message in section.followups:
                    if not message.content.strip():
                        continue
                    parts.append(f'\n**{message.title}**\n')
                    if message.content_src == ContentSource.CONSOLE:
                        parts.append(_md_code_fence(message.content.rstrip()))
                    else:
                        parts.append(message.content)
            elif section.content.strip():
                if section.content_src == ContentSource.CONSOLE:
                    parts.append(_md_code_fence(section.content.rstrip()))
                else:
                    parts.append(section.content)
            parts.append('')

    if errors:
        parts.append('---\n')
        parts.append('## Errors\n')
        for error in errors:
            parts.append(f'- `{_escape_md(error)}`')
        parts.append('')

    return '\n'.join(parts)


def _build_pdf_bytes(md_content: str) -> bytes:
    """Convert markdown content to PDF bytes via weasyprint."""
    import markdown2
    import weasyprint

    body_html = markdown2.markdown(
        md_content,
        extras=['fenced-code-blocks', 'tables', 'break-on-newline'],
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
            for i, entry in enumerate(tool_data):
                cs, result = entry[0], entry[1]
                error = entry[2] if len(entry) > 2 else None
                if i > 0:
                    ui.separator().classes('my-1')
                with ui.column().classes('w-full gap-1'):
                    # Call signature
                    is_python_cli = (
                        cs.func_name == 'python_cli'
                        and cs.kwargs
                        and isinstance(cs.kwargs.get('code'), str)
                    )
                    if is_python_cli:
                        other_kwargs = {k: v for k, v in cs.kwargs.items() if k != 'code'}
                        sig_parts = [repr(a) for a in (cs.args or [])]
                        sig_parts.extend(f'{k}={v!r}' for k, v in other_kwargs.items())
                        sig_text = f"{cs.func_name}({', '.join(sig_parts)}{', ' if sig_parts else ''}code=...)"
                    else:
                        sig_text = cs.format()
                    with ui.row().classes('items-center gap-2 px-2 py-1 rounded').style(
                        f'background: {_TOOL_CALL_BG}'
                    ):
                        ui.icon('call_made').classes('text-xs text-orange-700')
                        ui.label(sig_text).classes('text-xs font-mono font-semibold text-orange-900 break-all')
                    if is_python_cli:
                        ui.code(cs.kwargs['code'] or '(empty)', language='python').classes(
                            'w-full text-xs'
                        ).style('border-radius: 4px; margin: 0')
                    # Error
                    if error:
                        error_str = str(error).strip()
                        ui.html(
                            f'<pre style="'
                            f'white-space: pre-wrap; word-break: break-word; overflow-wrap: break-word;'
                            f'font-family: \'JetBrains Mono\', monospace;'
                            f'font-size: 0.72rem; line-height: 1.5;'
                            f'color: #b71c1c; background: #fbe9e7;'
                            f'border: 1px solid #ef9a9a; border-radius: 4px;'
                            f'padding: 8px; margin: 0; width: 100%; box-sizing: border-box;'
                            f'">{_esc(error_str)}</pre>'
                        ).classes('w-full')
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
                            f'">{_esc(result_str)}</pre>'
                        ).classes('w-full')
                    elif not error and cs.func_name != 'python_cli':
                        ui.label('(no result)').classes('text-xs text-gray-400 italic px-2')


def _render_history_message(message: _HistoryMessage) -> None:
    """Render a normalized non-assistant message."""
    if not message.content.strip():
        return
    if message.content_src == ContentSource.CONSOLE:
        with ui.expansion(message.title, icon='terminal').props('dense').classes(
            'w-full rounded border border-gray-200'
        ):
            _render_console_output(message.content)
        return

    if message.title == 'Reminder':
        icon = 'tips_and_updates'
        bg = '#fffde7'
        border = '#fdd835'
        text = '#795548'
    elif message.content_src == ContentSource.USER:
        icon = 'chat'
        bg = '#f1f8e9'
        border = '#c5e1a5'
        text = '#33691e'
    else:
        icon = 'description'
        bg = '#f5f5f5'
        border = '#e0e0e0'
        text = '#455a64'
    with ui.column().classes('w-full gap-0'):
        with ui.row().classes('items-center gap-2 px-3 py-1 rounded-t').style(
            f'background: {bg}; border: 1px solid {border}; border-bottom: none'
        ):
            ui.icon(icon).classes('text-sm')
            ui.label(message.title).classes(
                'text-xs font-semibold uppercase tracking-wide'
            ).style(f'color: {text}')
        ui.label(message.content).classes(
            'text-sm whitespace-pre-wrap w-full p-3'
        ).style(
            f'background: {bg}; border: 1px solid {border}; color: {text};'
            f'border-radius: 0 0 6px 6px; margin-top: 0'
        )


def _render_history_section(section: _HistorySection, job=None) -> None:
    """Render one normalized job-detail section."""
    if section.kind != 'assistant':
        _render_history_message(
            _HistoryMessage(
                title=section.title,
                content=section.content,
                content_src=section.content_src,
            )
        )
        return

    code_bg = _CODE_WARMUP_BG if section.content_src == ContentSource.SYSTEM else _CODE_LLM_BG
    if section.content_src == ContentSource.SYSTEM:
        if section.warmup_num is not None and (section.warmup_total or 0) > 1:
            code_label = f'warmup code {section.warmup_num} / {section.warmup_total}'
        else:
            code_label = 'warmup code'
        if section.hidden:
            code_label = f'{code_label} (hidden)'
    else:
        code_label = 'LLM submitted code' if section.render_as_code else 'LLM response'

    with ui.column().classes('w-full gap-2'):
        if section.content_src != ContentSource.SYSTEM:
            with ui.row().classes('w-full items-center gap-3 px-3 py-2 rounded-lg').style(
                _LLM_HEADER_BG + '; border: 1px solid #c5cae9'
            ):
                ui.icon('smart_toy').classes('text-indigo-600')
                ui.label(section.title).classes('text-sm font-bold text-indigo-800 flex-1')
                if section.tool_data:
                    ui.icon('build').classes('text-orange-500 text-sm')
                    ui.label(
                        f'{len(section.tool_data)} tool call{"s" if len(section.tool_data) != 1 else ""}'
                    ).classes('text-xs font-semibold px-2 py-0.5 rounded-full bg-orange-100 text-orange-700')
                if job is not None and section.turn_num is not None:
                    ui.button(
                        'Request JSON',
                        icon='data_object',
                        on_click=lambda _job=job, _turn=section.turn_num, _title=section.title: (
                            _open_step_preview_dialog(_job, _turn, _title)
                        ),
                    ).props('flat dense no-caps').classes(
                        'text-xs text-indigo-700'
                    ).tooltip('Inspect the reconstructed JSON request for this LLM turn')

        if section.render_as_code:
            _render_code_block(section.content, code_bg, code_label)
        else:
            _render_history_message(
                _HistoryMessage(
                    title=code_label,
                    content=section.content or '(empty)',
                    content_src=section.content_src,
                )
            )

        if section.tool_data:
            _render_tool_calls(section.tool_data)

        for message in section.followups:
            _render_history_message(message)


def _render_warmup_section(job, blocks: list, ranges: list[tuple[int, int]]) -> None:
    """Render all warmup blocks with their console output."""
    warmup_items = _get_warmup_log_items(job)
    for i, (block, (from_pos, to_pos)) in enumerate(zip(blocks, ranges)):
        code = _get_code_str(block)
        console_out = _get_console_slice(job.py_env.console, from_pos, to_pos)
        block_label = f'Warmup Code {i + 1} / {len(blocks)}' if len(blocks) > 1 else 'Warmup Code'
        if _is_hidden_warmup_block(block):
            block_label = f'{block_label} (Hidden)'
        tool_data = _get_tool_data_for_block(block, warmup_items.get(i))

        with ui.column().classes('w-full gap-2'):
            if code.strip():
                _render_code_block(code, _CODE_WARMUP_BG, block_label.lower())

            if tool_data:
                _render_tool_calls(tool_data)

            if console_out.strip():
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

    tool_data = _get_tool_data_for_block(chat_item.llm_resp, chat_item)

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
            if ts_str:
                ui.label(ts_str).classes('text-xs text-indigo-500 font-mono')
            ui.label('LLM CODE').classes(
                'text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700'
            )

        _render_code_block(code, _CODE_LLM_BG, 'LLM submitted code')

        if tool_data:
            _render_tool_calls(tool_data)

        with ui.expansion('Console Output', icon='terminal').props('dense').classes(
            'w-full rounded border border-gray-200'
        ):
            _render_console_output(console_out)


def _get_system_prompt(job, difficulty: Optional[TaskDifficulty] = None) -> str:
    """Return the formatted system prompt, or the raw template on render failure."""
    try:
        if not job.job_def or not job.job_def.agent:
            return ''
        try:
            if difficulty is not None:
                return job.system_prompt(difficulty) or ''
            return job.system_prompt() or ''
        except Exception:  # pylint: disable=broad-except
            raw = job.job_def.agent._system_prompt or ''  # pylint: disable=protected-access
            if raw and difficulty is not None:
                try:
                    return format_system_prompt(raw, difficulty)
                except Exception:  # pylint: disable=broad-except
                    pass
            return str(raw) if raw else ''
    except Exception:  # pylint: disable=broad-except
        return ''


def _get_current_difficulty(job) -> Optional[TaskDifficulty]:
    """Return the job's current difficulty, or None when unavailable."""
    try:
        return job.get_current_difficulty()
    except Exception:  # pylint: disable=broad-except
        return None


def _get_difficulty_button_specs() -> list[dict]:
    """Return display metadata for the system prompt difficulty buttons."""
    return [
        {
            'key': str(difficulty),
            'difficulty': difficulty,
            'label': label,
            'tooltip': f'{label_full} difficulty',
        }
        for difficulty, label, label_full in [
            (TaskDifficulty.low, 'L', 'Low'),  # pylint: disable=no-member
            (TaskDifficulty.medium, 'M', 'Medium'),  # pylint: disable=no-member
            (TaskDifficulty.high, 'H', 'High'),  # pylint: disable=no-member
        ]
    ]


def _render_latency_summary_card(title: str, summary: _LatencySummary, color: str) -> None:
    """Render a compact summary card for one latency series."""
    with ui.card().classes('gap-2 shadow-sm').style(
        f'border-top: 3px solid {color}; min-width: 220px'
    ):
        ui.label(title).classes('text-sm font-semibold text-gray-800')
        with ui.row().classes('items-center gap-3 flex-wrap'):
            ui.label(
                f'{summary.sample_count} '
                f'sample{"s" if summary.sample_count != 1 else ""}'
            ).classes(
                'text-xs text-gray-600'
            )
            ui.label(f'{summary.completed_count} resolved').classes('text-xs text-green-700')
            if summary.pending_count:
                ui.label(f'{summary.pending_count} pending').classes('text-xs text-amber-700')
        with ui.row().classes('items-center gap-3 flex-wrap'):
            ui.label(f'Min: {_format_latency_seconds(summary.min_seconds)}').classes(
                'text-xs font-mono text-gray-600'
            )
            ui.label(f'Avg: {_format_latency_seconds(summary.avg_seconds)}').classes(
                'text-xs font-mono text-gray-600'
            )
            ui.label(f'Max: {_format_latency_seconds(summary.max_seconds)}').classes(
                'text-xs font-mono text-gray-600'
            )


def _render_latency_table(
    title: str,
    icon: str,
    description: str,
    samples: list[tuple[datetime, Optional[float]]],
    empty_label: str,
) -> None:
    """Render one latency series."""
    summary = _summarize_latencies(samples)
    with ui.column().classes('w-full gap-3'):
        with ui.row().classes('items-start gap-2'):
            ui.icon(icon).classes('text-base text-gray-600 mt-0.5')
            with ui.column().classes('gap-1'):
                ui.label(title).classes('text-base font-semibold text-gray-900')
                ui.label(description).classes('text-sm text-gray-600 whitespace-pre-wrap')

        accent_color = '#90caf9' if icon == 'schedule' else '#a5d6a7'
        _render_latency_summary_card(title, summary, accent_color)

        if not samples:
            ui.label(empty_label).classes('text-sm italic text-gray-500')
            return

        with ui.card().classes('w-full shadow-sm p-0 overflow-hidden'):
            with ui.row().classes(
                'w-full items-center gap-4 px-4 py-2 bg-gray-50 '
                'border-b border-gray-200'
            ):
                ui.label('Request Timestamp').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-56 shrink-0'
                )
                ui.label('Latency').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-28 shrink-0'
                )
                ui.label('Status').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide flex-1'
                )

            for timestamp, seconds in samples:
                with ui.row().classes(
                    'w-full items-start gap-4 px-4 py-3 border-b border-gray-100'
                ):
                    ui.label(_format_latency_timestamp(timestamp)).classes(
                        'text-xs font-mono text-gray-600 w-56 shrink-0'
                    )
                    latency_classes = (
                        'text-xs font-mono w-28 shrink-0 text-amber-700'
                        if seconds is None else
                        'text-xs font-mono w-28 shrink-0 text-gray-800'
                    )
                    ui.label(_format_latency_seconds(seconds)).classes(latency_classes)
                    ui.label(
                        'Waiting for response' if seconds is None else 'Recorded'
                    ).classes(
                        'text-sm text-gray-600 flex-1'
                    )


def _open_latency_dialog(job) -> None:
    """Open the latency dialog."""
    overall_samples = _get_latency_samples(job, 'get_response_times')
    llm_samples = _get_latency_samples(job, 'get_llm_response_times')

    with ui.dialog() as dlg:
        with ui.card().classes('w-[1000px] max-w-[95vw] max-h-[90vh] overflow-auto'):
            with ui.row().classes('w-full items-start justify-between gap-4'):
                with ui.column().classes('gap-1 flex-1'):
                    ui.label('Latency Breakdown').classes(
                        'text-xl font-bold text-gray-900'
                    )
                    ui.label(
                        'Overall shows user-visible wait time. LLM shows model round-trip time.'
                    ).classes(
                        'text-sm text-gray-600'
                    )
                ui.button(icon='close', on_click=dlg.close).props('flat round dense')

            ui.separator()

            with ui.column().classes('w-full gap-6'):
                _render_latency_table(
                    title='Overall Latency',
                    icon='schedule',
                    description='From each user message to the first visible LLM response.',
                    samples=overall_samples,
                    empty_label='No overall latency samples.',
                )
                _render_latency_table(
                    title='LLM Response Latency',
                    icon='smart_toy',
                    description=(
                        'From the LLM request boundary to the response. '
                        'Without an explicit pending marker, values are approximate.'
                    ),
                    samples=llm_samples,
                    empty_label='No LLM latency samples.',
                )

    dlg.open()


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

    model = _get_job_model(job)
    locale_str = _get_locale_str(job)
    temperature_str = _get_job_temperature(job)
    provider_str = _get_job_provider(job)
    uses_reasoning = _job_uses_reasoning(job)
    chat_style_str = ''
    try:
        if job.job_def and job.job_def.chat_style is not None:
            chat_style_str = str(job.job_def.chat_style)
    except Exception:  # pylint: disable=broad-except
        pass
    total_cost = _get_effective_job_cost(job)
    num_turns = getattr(job, 'num_turns', 0) or 0
    exception_count = getattr(job, 'exception_count', 0) or 0
    try:
        tps = job.tokens_per_sec()
    except Exception:  # pylint: disable=broad-except
        tps = 0.0
    status_str = '—'
    try:
        status_str = str(job.status) if job.status else '—'
    except Exception:  # pylint: disable=broad-except
        pass

    current_difficulty = _get_current_difficulty(job)
    system_prompt = _get_system_prompt(job, current_difficulty)
    history_sections = _build_history_sections(job)
    exception_messages = _get_exception_messages(job)
    reported_tools = _get_reported_tools(job)
    difficulty_state = {'difficulty': current_difficulty}
    try:
        difficulty_button_specs = _get_difficulty_button_specs()
    except Exception:  # pylint: disable=broad-except
        difficulty_button_specs = []

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
                        if provider_str:
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('cloud').classes('text-sm text-gray-500')
                                ui.label(provider_str).classes('text-xs font-mono text-gray-600')
                        if locale_str:
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('language').classes('text-sm text-gray-500')
                                ui.label(locale_str).classes('text-xs font-mono text-gray-600')
                        if temperature_str:
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('device_thermostat').classes('text-sm text-gray-500')
                                ui.label(temperature_str).classes('text-xs font-mono text-gray-600')
                        if uses_reasoning:
                            with ui.row().classes('items-center'):
                                ui.icon('psychology_alt').classes(
                                    'text-lg text-orange-500 bg-orange-50 rounded-full p-1'
                                ).tooltip('Reasoning was enabled for this job.')
                        if chat_style_str:
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('forum').classes('text-sm text-gray-500')
                                ui.label(chat_style_str).classes('text-xs font-mono text-gray-600')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('payments').classes('text-sm text-gray-500')
                            ui.label(f'${total_cost:.4f}').classes('text-xs font-mono text-gray-600')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('repeat').classes('text-sm text-gray-500')
                            ui.label(f'{num_turns} turn{"s" if num_turns != 1 else ""}').classes(
                                'text-xs text-gray-600'
                            )
                        if tps:
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('bolt').classes('text-sm text-gray-500')
                                ui.label(f'{tps:.0f} tok/s').classes('text-xs font-mono text-gray-600')
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
                        _errors=exception_count, _chat_style=chat_style_str,
                        _sys=system_prompt, _job=job,
                    ):
                        md = _build_md_content(
                            uuid_str=_uuid, status_str=_status, agent_role=_agent,
                            model=_model, total_cost=_cost, num_turns=_turns,
                            exception_count=_errors, chat_style=_chat_style,
                            system_prompt=_sys, job=_job,
                        )
                        ui.download(md.encode(), filename=f'job_{_uuid}.md', media_type='text/markdown')

                    def _download_pdf(
                        _uuid=uuid_str, _status=status_str, _agent=agent_role,
                        _model=model, _cost=total_cost, _turns=num_turns,
                        _errors=exception_count, _chat_style=chat_style_str,
                        _sys=system_prompt, _job=job,
                    ):
                        md = _build_md_content(
                            uuid_str=_uuid, status_str=_status, agent_role=_agent,
                            model=_model, total_cost=_cost, num_turns=_turns,
                            exception_count=_errors, chat_style=_chat_style,
                            system_prompt=_sys, job=_job,
                        )
                        pdf = _build_pdf_bytes(md)
                        ui.download(pdf, filename=f'job_{_uuid}.pdf', media_type='application/pdf')

                    ui.button('MD', icon='download', on_click=_download_md).props(
                        'flat dense no-caps'
                    ).classes('text-xs text-indigo-600 mx-1').tooltip('Download as Markdown')
                    ui.button('PDF', icon='picture_as_pdf', on_click=_download_pdf).props(
                        'flat dense no-caps'
                    ).classes('text-xs text-red-600 mx-1').tooltip('Download as PDF')
                    ui.button(
                        'JSON',
                        icon='data_object',
                        on_click=lambda j=job: _open_raw_json_dialog(j),
                    ).props(
                        'flat dense no-caps'
                    ).classes('text-xs text-indigo-700 mx-1').tooltip(
                        'Inspect the raw job structure as expandable JSON'
                    )
                    raw_text_js = json.dumps(_build_raw_repr(job))
                    ui.button(icon='content_copy').props(
                        'flat dense round'
                    ).classes('text-emerald-700 mx-1').on(
                        'click',
                        js_handler=f'() => {{ navigator.clipboard.writeText({raw_text_js}); }}',
                    ).tooltip('Copy raw text to clipboard')
                    ui.button(
                        'Latency',
                        icon='speed',
                        on_click=lambda j=job: _open_latency_dialog(j),
                    ).props(
                        'flat dense no-caps'
                    ).classes('text-xs text-sky-700 mx-1').tooltip(
                        'Show overall and LLM latency details'
                    )
                    ui.button(icon='close', on_click=dlg.close).props('flat round dense').classes('ml-2')

            ui.separator().classes('mb-2')

            # ── Tab bar ─────────────────────────────────────────────────────
            with ui.tabs().classes('mb-3') as tabs:
                tab_log = ui.tab('Execution Log', icon='timeline')
                tab_tools = ui.tab(f'Tools ({len(reported_tools)})', icon='build')
                tab_raw = ui.tab('Raw', icon='data_object')

            with ui.tab_panels(tabs, value=tab_log).classes('w-full'):

                # ── Execution Log tab ────────────────────────────────────────
                with ui.tab_panel(tab_log).classes('px-0'):

                    # ── System Prompt ────────────────────────────────────────
                    if system_prompt:
                        with ui.expansion('System Prompt', icon='description').props('dense').classes(
                            'w-full rounded border border-gray-200 mb-4'
                        ):
                            @ui.refreshable
                            def system_prompt_view() -> None:
                                prompt = _get_system_prompt(
                                    job,
                                    difficulty_state['difficulty'],
                                )
                                with ui.element('div').classes('w-full relative'):
                                    if difficulty_button_specs:
                                        with ui.row().classes(
                                            'absolute top-2 right-2 z-10 gap-1 '
                                            'rounded border border-amber-300/50 '
                                            'bg-amber-50/55 p-0.5 shadow-sm backdrop-blur-sm'
                                        ):
                                            for spec in difficulty_button_specs:
                                                is_selected = (
                                                    difficulty_state['difficulty']
                                                    == spec['difficulty']
                                                )
                                                button_classes = (
                                                    'w-8 h-8 min-w-0 px-0 text-xs font-bold '
                                                    'border transition-colors '
                                                )
                                                button_classes += (
                                                    'bg-amber-200/75 text-amber-950 border-amber-400/70 '
                                                    'shadow-sm'
                                                    if is_selected else
                                                    'bg-white/35 text-gray-600 border-white/30 '
                                                    'hover:bg-white/70 hover:text-gray-800 '
                                                    'hover:border-amber-200/70'
                                                )

                                                def _select_difficulty(
                                                    _event=None,
                                                    _difficulty=spec['difficulty'],
                                                ):
                                                    difficulty_state['difficulty'] = _difficulty
                                                    system_prompt_view.refresh()

                                                ui.button(
                                                    spec['label'],
                                                    on_click=_select_difficulty,
                                                ).props(
                                                    'flat dense no-caps'
                                                ).classes(button_classes).tooltip(spec['tooltip'])

                                    prompt_classes = (
                                        'text-xs text-gray-700 whitespace-pre-wrap w-full '
                                        'rounded p-3 min-h-12'
                                    )
                                    if difficulty_button_specs:
                                        prompt_classes += ' pr-32'

                                    ui.label(prompt).classes(prompt_classes).style(
                                        'font-family: "JetBrains Mono", monospace; '
                                        'background: #fdf6e3'
                                    )

                            system_prompt_view()

                    if history_sections:
                        with ui.column().classes('w-full gap-3'):
                            for section in history_sections:
                                _render_history_section(section, job=job)

                    if exception_messages:
                        with ui.expansion('Errors', icon='error_outline').props('dense').classes(
                            'w-full rounded border border-red-200 mt-4'
                        ):
                            for error in exception_messages:
                                with ui.row().classes('items-center gap-2 px-3 py-2'):
                                    ui.icon('error').classes('text-red-500 text-sm')
                                    ui.label(error).classes('text-xs text-red-700 font-mono break-all')

                    if not history_sections:
                        with ui.column().classes('items-center justify-center gap-3 mt-8'):
                            ui.icon('hourglass_empty').classes('text-4xl text-gray-300')
                            ui.label('No chat history yet.').classes('text-gray-400 italic')

                # ── Tools tab ────────────────────────────────────────────────
                with ui.tab_panel(tab_tools).classes('px-0'):
                    if reported_tools:
                        with ui.column().classes('w-full gap-2'):
                            for fn in reported_tools:
                                with ui.card().classes('w-full shadow-sm border border-gray-200'):
                                    ui.label(_format_reported_tool_label(fn)).classes(
                                        'text-sm font-mono text-gray-800'
                                    )
                                    doc = (getattr(fn, '__doc__', '') or '').strip()
                                    if doc:
                                        ui.label(doc.splitlines()[0]).classes('text-xs text-gray-600')
                    else:
                        with ui.column().classes('items-center justify-center gap-3 mt-8'):
                            ui.icon('build_circle').classes('text-4xl text-gray-300')
                            ui.label('No tools were reported in the LLM request.').classes(
                                'text-gray-400 italic'
                            )

                # ── Raw tab ──────────────────────────────────────────────────
                with ui.tab_panel(tab_raw).classes('px-0'):
                    raw_html = _build_raw_html(job)
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
