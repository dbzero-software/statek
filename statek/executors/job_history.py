# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Job history rendering, snapshots, replay, summaries, and metrics."""

from __future__ import annotations

# dbzero enum members are generated at runtime.
# pylint: disable=no-member

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
import json
from typing import (
    TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union,
)

from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, NotificationLogItem,
    PostProcessedItem, ReminderLogItem, SubTaskLogItem, ToolError, UserLogItem, WarmupLogItem,
)
from statek.pyenv import ErrorKind, PyEnv
from statek.utils import (
    CallSpecWrapper, CodeBlock, _is_hidden_warmup_block, extract_dialog,
    format_default_llm_repr, parse_tool_log,
)

if TYPE_CHECKING:
    from statek.executors.job import DialogItem, Job, JobSummary
    from statek.task import SubTaskHandler


@dataclass
class HistoryStep:
    """One normalized source entry; final carries the snapshot's diagnostic index."""

    messages: tuple = ()
    log_item: Optional[Any] = None
    console_start: int = 0
    console_end: int = 0
    console: tuple = ()
    exceptions: tuple = ()
    turn_num: Optional[int] = None
    warmup_num: Optional[int] = None
    final: bool = False


class _HistoryStepIterator:
    """Retry a failed read against the same captured append boundaries."""

    def __init__(self, factory):
        self.factory = factory
        self.iterator = None
        self.delivered = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.iterator is None:
                self.iterator = self.factory()
                for _ in range(self.delivered):
                    next(self.iterator)
            step = next(self.iterator)
        except StopIteration:
            raise
        except Exception:
            self.iterator = None
            raise
        self.delivered += 1
        return step


class _HistoryLogView(Sequence):
    """Bound a persistent log without copying its elements."""

    def __init__(self, log, length):
        self.log = log
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.log[i] for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.log[index]


class _HistoryConsoleView(Sequence):
    """Cache only the most recently requested console slice."""

    def __init__(self, env, length):
        self.env = env
        self.length = length
        self.bounds = None
        self.values = ()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not isinstance(index, slice):
            return self[index:index + 1][0]
        start, stop, stride = index.indices(self.length)
        bounds = (start, stop)
        if self.bounds != bounds:
            self.values = tuple(self.env.iter_console(start, stop)) if stop > start else ()
            self.bounds = bounds
        return self.values[::stride]


def format_job_summary(job_summary: JobSummary, py_env: PyEnv,
                       chat_style: ChatStyle) -> str:
    """Format a compact job summary, showing only handles still in py_env."""
    # pylint: disable=no-member
    preamble = (
        "Below is a summary of past communication and actions in this session. "
        "It includes Input Summary, Output Summary, System Summary, and State "
        "(available Python CLI handles)."
    )
    if chat_style in (ChatStyle.MARKDOWN, ChatStyle.DIRECT):
        heading_prefix = "# "
    elif chat_style in (ChatStyle.CONSOLE, ChatStyle.MD_DIALOG):
        heading_prefix = ""
    else:
        raise ValueError(f"Unsupported chat style: {chat_style!r}")

    descriptions = sorted(
        job_summary.state.locals if job_summary.state else (),
        key=lambda handle: (handle.var_name, handle.desc or ""),
    )
    types_by_name = py_env.resolve_locals(handle.var_name for handle in descriptions)
    variable_lines = []
    for handle in descriptions:
        resolved_type = types_by_name.get(handle.var_name)
        if resolved_type is None:
            continue
        line = f"    {handle.var_name} ({resolved_type.__name__})"
        if handle.desc:
            line += f": {handle.desc}"
        variable_lines.append(line)

    sections = [
        f"{heading_prefix}Input Summary\n{job_summary.input_summary}",
        f"{heading_prefix}Output Summary\n{job_summary.output_summary}",
        f"{heading_prefix}System Summary"
        + (f"\n{job_summary.system_summary}" if job_summary.system_summary else ""),
        f"{heading_prefix}State\nVariables:"
        + ("\n" + "\n".join(variable_lines) if variable_lines else ""),
    ]
    result = "\n\n".join([preamble, *sections])
    if chat_style == ChatStyle.MD_DIALOG:
        return f"<job_summary>\n{result}\n</job_summary>"
    return result


def _warmup_code_text(warmup) -> str:
    """Return concatenated LLM-facing text of non-hidden warmup code blocks."""
    if warmup is None:
        return ""
    if _is_hidden_warmup_block(warmup):
        return ""
    if isinstance(warmup, str):
        return warmup
    if isinstance(warmup, CodeBlock):
        return warmup.code or ""
    return "".join(
        w if isinstance(w, str) else (w.code or "")
        for w in warmup
        if not _is_hidden_warmup_block(w)
    )


def _tool_log_text(item: Optional["LLM_LogItem"]) -> str:
    """Return all tool log text from an LLM_LogItem as a single string."""
    if item is None or item.tool_log is None:
        return ""
    tool_log = item.tool_log
    if isinstance(tool_log, str):
        return tool_log
    if isinstance(tool_log, ToolError):
        return tool_log.err_message
    return "\n".join(tl if isinstance(tl, str) else tl.err_message for tl in tool_log)


def _llm_resp_output_tokens(llm_resp) -> int:
    """Approximate output token count for an LLM response."""
    if isinstance(llm_resp, str):
        return len(llm_resp) // 4
    code = (llm_resp.code or "") if llm_resp else ""
    tool_calls = list(llm_resp.tool_calls) if llm_resp and llm_resp.tool_calls else []
    if not tool_calls:
        return len(code) // 4
    tool_json = json.dumps([
        {"name": call.func_name, "arguments": dict(call.kwargs or {})} for call in tool_calls
    ])
    return (len(code) + len(tool_json)) // 4


def _summary_ranges(job, log_length):
    """Normalize and validate the persisted, non-overlapping summary ranges."""
    from statek.executors.job import JobCompactionResult  # pylint: disable=import-outside-toplevel

    summaries = getattr(job, "_Job__job_summaries", None)
    if isinstance(summaries, JobCompactionResult):
        summaries = [summaries]
    summaries = sorted(summaries or [], key=lambda result: result.chat_log_pos)
    previous_end = 0
    for result in summaries:
        end = result.chat_log_pos + result.chat_log_len
        if (result.chat_log_pos < previous_end or result.chat_log_pos < 0
                or result.chat_log_len <= 0 or end > log_length):
            raise ValueError("Invalid or overlapping job summary chat-log range")
        previous_end = end
    return summaries


class _HistoryCompaction:
    """Locate summaries without mixing range bookkeeping into message rendering."""

    def __init__(self, summaries, chat_log):
        self.by_index = {result.chat_log_pos: result.job_summary for result in summaries}
        self.covered = {
            index for result in summaries
            for index in range(result.chat_log_pos, result.chat_log_pos + result.chat_log_len)
        }
        self.last_warmup = -1
        if summaries:
            for index, item in enumerate(chat_log):
                if isinstance(item, WarmupLogItem):
                    self.last_warmup = index
                    if index in self.covered:
                        raise ValueError("Job summary range cannot cover warmup chat-log items")
        self.deferred = [
            summary for start, summary in sorted(self.by_index.items())
            if start <= self.last_warmup
        ]


def _history_entries(chat_log, console_len, summary_indices):
    """Yield source entries and their next execution boundary with bounded lookahead."""
    boundary_index = -1
    next_pos = console_len
    for raw_index, item in enumerate(chat_log):
        if ((isinstance(item, str) or (raw_index == 0 and isinstance(item, UserLogItem)))
                and not (raw_index != 0 and raw_index in summary_indices)):
            continue
        if raw_index >= boundary_index:
            boundary_index = raw_index + 1
            while boundary_index < len(chat_log):
                following = chat_log[boundary_index]
                if isinstance(following, (WarmupLogItem, LLM_LogItem, PostProcessedItem,
                                          ReminderLogItem, SubTaskLogItem, NotificationLogItem)):
                    break
                boundary_index += 1
            next_pos = (chat_log[boundary_index].console_pos
                        if boundary_index < len(chat_log) else console_len)
        yield raw_index, item, next_pos


def _tool_result(result):
    return result.err_message if isinstance(result, ToolError) else result


class _HistoryRenderer:
    """Normalize one entry at a time, keeping per-render inputs out of Job."""

    def __init__(self, history, console, push_log):
        self.history = history
        self.job = history._job  # pylint: disable=protected-access
        self.console = console
        self.push_log = push_log or {}
        self.push_keys = sorted(key for key in self.push_log if key != 0)
        warmup = self.job.job_def.warmup_code
        self.warmup_blocks = (
            [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup or ())
        )

    def user_message(self, message):
        return ChatHistoryItem(
            role=ChatRole.USER,
            content=self.job._with_language_hint(message),  # pylint: disable=protected-access
            content_src=ContentSource.USER,
        )

    def initial_messages(self, initial):
        message = initial.message if isinstance(initial, UserLogItem) else initial
        parts = [message] if isinstance(message, str) and message else []
        pushed = self.push_log.get(0)
        parts.extend(value for value in (pushed if isinstance(pushed, list) else [pushed])
                     if value)
        if parts:
            yield ChatHistoryItem(
                role=ChatRole.USER,
                content="\n".join(self.user_message(part).content for part in parts),
                content_src=ContentSource.USER,
            )

    def pushes(self, start, end):
        for key in self.push_keys[bisect_right(self.push_keys, start):
                                  bisect_right(self.push_keys, end)]:
            values = self.push_log[key]
            for message in values if isinstance(values, list) else [values]:
                yield self.user_message(message)

    def summary_step(self, summary):
        return HistoryStep(messages=(ChatHistoryItem(
            role=ChatRole.USER,
            content=format_job_summary(
                summary, self.job.py_env, self.job.job_def.chat_style or ChatStyle.CONSOLE,
            ),
            content_src=ContentSource.SYSTEM,
        ),))

    def messages(self, item, index, end):
        if isinstance(item, UserLogItem):
            if item.message:
                yield self.user_message(item.message)
        elif isinstance(item, (PostProcessedItem, ReminderLogItem)):
            text = item.message if isinstance(item, PostProcessedItem) else item.reminder.text
            if text:
                yield ChatHistoryItem(
                    role=ChatRole.SYSTEM, content=text, content_src=ContentSource.SYSTEM,
                )
        elif isinstance(item, SubTaskLogItem):
            yield from self.subtask_messages(item, index)
        elif isinstance(item, NotificationLogItem):
            call = self.history._notification_tool_call(index)  # pylint: disable=protected-access
            result = item.tool_log if item.tool_log is not None else format_default_llm_repr([item])
            yield from self.synthetic_tool_messages(call, result)
        elif isinstance(item, (WarmupLogItem, LLM_LogItem)):
            yield from self.execution_messages(item, end)

    def subtask_messages(self, item, index):
        try:
            notification = item.handler.get_log_message()
        except Exception:  # pylint: disable=broad-except
            task_id = f" id={item.handler.id}" if item.handler.id is not None else ""
            notification = (
                f"[Error] sub-task{task_id} failed." if item.handler.error is not None
                else f"[Notification] sub-task{task_id} completed successfully."
            )
        yield ChatHistoryItem(
            role=ChatRole.SYSTEM, content=notification, content_src=ContentSource.SYSTEM,
        )
        if item.tool_log is not None:
            call = self.history._subtask_tool_call(index, item.handler)  # pylint: disable=protected-access
            yield from self.synthetic_tool_messages(call, item.get_tool_result(0))

    @staticmethod
    def synthetic_tool_messages(call, result):
        yield ChatHistoryItem(
            role=ChatRole.ASSISTANT, content=None, content_src=ContentSource.SYSTEM,
            tool_calls=[call],
        )
        yield ChatHistoryItem(
            role=ChatRole.TOOL, content=_tool_result(result), content_src=ContentSource.CONSOLE,
            tool_calls=[call],
        )

    def console_text(self, start, end):
        return "\n".join(self.console[start:end]) if end > start and self.console else ""

    def execution_messages(self, item, end):
        is_warmup = isinstance(item, WarmupLogItem)
        if is_warmup:
            number = item.warmup_block_num
            block = self.warmup_blocks[number] if number < len(self.warmup_blocks) else None
            if _is_hidden_warmup_block(block):
                return
        else:
            block = item.llm_resp
        source = ContentSource.SYSTEM if is_warmup else ContentSource.ASSISTANT
        code = block.code if isinstance(block, CodeBlock) else block
        calls = list(block.tool_calls) if isinstance(block, CodeBlock) and block.tool_calls else []
        direct_warmup = (
            is_warmup and self.job.job_def.chat_style == ChatStyle.DIRECT and not calls and code
        )
        if direct_warmup:
            call = CallSpecWrapper(
                id=f"STATEK-WARMUP-{item.warmup_block_num:03d}",
                func_name="python_cli", args=None, kwargs={"code": code},
            )
            yield from self.synthetic_tool_messages(call, self.console_text(item.console_pos, end))
        else:
            yield from self.assistant_messages(item, code, calls, source)
            if not calls and end > item.console_pos and self.console:
                yield ChatHistoryItem(
                    role=ChatRole.USER, content=self.console_text(item.console_pos, end),
                    content_src=ContentSource.CONSOLE,
                )
        yield from self.pushes(item.console_pos, end)

    @staticmethod
    def assistant_messages(item, code, calls, source):
        reasoning = (
            getattr(item, "llm_reasoning_payload", None) if isinstance(item, LLM_LogItem) else None
        )
        if calls or code:
            yield ChatHistoryItem(
                role=ChatRole.ASSISTANT, content=code or None, content_src=source,
                tool_calls=calls or None, provider_reasoning_payload=reasoning,
            )
        elif isinstance(item, LLM_LogItem) and item.llm_resp is not None and reasoning is not None:
            yield ChatHistoryItem(role=ChatRole.ASSISTANT, provider_reasoning_payload=reasoning)
        for index, call in enumerate(calls):
            try:
                result = item.get_tool_result(index)
            except (KeyError, IndexError):
                result = ""
            yield ChatHistoryItem(
                role=ChatRole.TOOL, content=_tool_result(result), content_src=ContentSource.CONSOLE,
                tool_calls=call,
            )


def _annotate_steps(steps, log, console, errors):
    """Attach source counters and sparse diagnostics to normalized steps."""
    error_positions = [pos for pos, _ in errors]
    first_pos = next((item.console_pos for item in log if hasattr(item, "console_pos")),
                     len(console))
    leading = errors[:bisect_left(error_positions, first_pos)]
    if leading:
        yield HistoryStep(exceptions=leading, console_end=first_pos)
    turn = 0
    for step in steps:
        if isinstance(step.log_item, LLM_LogItem) and any(
                message.role == ChatRole.ASSISTANT for message in step.messages):
            step.turn_num = turn
            turn += 1
        if isinstance(step.log_item, WarmupLogItem):
            step.warmup_num = step.log_item.warmup_block_num
        if step.log_item is not None:
            step.console = tuple(console[step.console_start:step.console_end])
        step.exceptions = errors[bisect_left(error_positions, step.console_start):
                                 bisect_left(error_positions, step.console_end)]
        yield step
    yield HistoryStep(exceptions=errors, final=True)


class JobHistory:
    """A transient view; all durable state remains owned by the job."""

    def __init__(self, job: "Job"):
        self._job = job


    @staticmethod
    def _subtask_lookup_code(handler: "SubTaskHandler") -> str:
        """Return the synthetic lookup code shown for a completed subtask."""
        if handler.id is None:
            return "print(find_sub_task_handler())"
        return f"print(find_sub_task_handler(id={handler.id!r}))"

    @staticmethod
    def _subtask_tool_call(idx: int, handler: "SubTaskHandler") -> CallSpecWrapper:
        """Return the synthetic python_cli call representing subtask result lookup."""
        return CallSpecWrapper(
            id=f"STATEK-SUBTASK-{idx:03d}",
            func_name="python_cli",
            args=None,
            kwargs={"code": JobHistory._subtask_lookup_code(handler)},
        )

    @staticmethod
    def _notification_tool_call(idx: int) -> CallSpecWrapper:
        """Return the synthetic python_cli call for one delivered text message."""
        return CallSpecWrapper(
            id=f"STATEK-NOTIFICATION-{idx:03d}",
            func_name="python_cli",
            args=None,
            kwargs={"code": "print(get_pending_notifications())"},
        )

    def iter_notification_log_items(self) -> Iterable[Tuple[str, NotificationLogItem]]:
        """Yield logged notifications paired with their synthetic history call IDs.

        IDs are scoped to this job's append-only chat log. Pending notifications
        are excluded; reading this iterator does not deliver or consume them.
        """
        for index, item in enumerate(self._job.chat_log):
            if isinstance(item, NotificationLogItem):
                yield self._notification_tool_call(index).id, item


    def get_compacted_chat_log(
        self,
    ) -> Iterable[Union[str, ChatLogItem, UserLogItem, JobSummary]]:
        """Yield committed chat history with summarized slices replaced."""
        chat_log = self._job.chat_log
        summaries = _summary_ranges(self._job, len(chat_log))
        if not summaries:
            yield from chat_log
            return
        position = 0
        for result in summaries:
            yield from chat_log[position:result.chat_log_pos]
            yield result.job_summary
            position = result.chat_log_pos + result.chat_log_len
        yield from chat_log[position:]

    def _iter_chat_history(
        self,
        chat_log: Optional[Sequence[Union[str, ChatLogItem, UserLogItem]]] = None,
        console: Optional[Sequence[str]] = None,
        push_log: Optional[Dict[int, Union[str, List[str]]]] = None,
        compacted: bool = False,
    ) -> Iterable[ChatHistoryItem]:
        """Yield LLM-facing messages for the supplied state or the live job."""
        summaries = (
            _summary_ranges(self._job, len(self._job.chat_log))
            if compacted and chat_log is None else []
        )
        chat_log = self._job.chat_log if chat_log is None else chat_log
        push_log = dict(self._job.py_env.iter_push_log()) if push_log is None else push_log
        console = list(self._job.py_env.iter_console()) if console is None else console
        for step in self._build_history_steps(chat_log, console, push_log, summaries):
            yield from step.messages

    def _build_history_steps(self, chat_log, console, push_log, summaries):
        """Combine source boundaries, summary placement, and per-entry rendering."""
        console = console if console is not None else []
        compaction = _HistoryCompaction(summaries, chat_log)
        renderer = _HistoryRenderer(self, console, push_log)
        initial = chat_log[0] if chat_log else None
        initial_messages = tuple(renderer.initial_messages(initial))
        if initial_messages:
            yield HistoryStep(messages=initial_messages)
        if (0 in compaction.by_index and compaction.last_warmup < 0
                and isinstance(initial, (str, UserLogItem))):
            yield renderer.summary_step(compaction.by_index[0])

        deferred = compaction.deferred
        entries = _history_entries(chat_log, len(console), compaction.by_index)
        for index, (raw_index, item, end) in enumerate(entries):
            if raw_index > compaction.last_warmup and deferred:
                yield from (renderer.summary_step(summary) for summary in deferred)
                deferred = []
            if raw_index in compaction.by_index and raw_index > compaction.last_warmup:
                yield renderer.summary_step(compaction.by_index[raw_index])
            if raw_index in compaction.covered:
                continue
            yield HistoryStep(
                messages=tuple(renderer.messages(item, index, end)), log_item=item,
                console_start=getattr(item, "console_pos", end), console_end=end,
            )
        yield from (renderer.summary_step(summary) for summary in deferred)

    def iter_history_steps(self, compacted: bool = True) -> Iterator["HistoryStep"]:
        """Capture append boundaries now and read each step's payload on demand.

        Existing entries are not deep-copied. Retrying a failed read retains the
        same boundaries; empty steps retain hidden warmup and pending-turn identity.
        """
        log = _HistoryLogView(self._job.chat_log, len(self._job.chat_log))
        console = _HistoryConsoleView(self._job.py_env, self._job.py_env.console_len)
        summaries = _summary_ranges(self._job, len(log)) if compacted else []
        pushes = dict(self._job.py_env.iter_push_log())
        errors = tuple(self._job.py_env.iter_exceptions())

        def generate():
            steps = self._build_history_steps(log, console, pushes, summaries)
            yield from _annotate_steps(steps, log, console, errors)

        return _HistoryStepIterator(generate)

    def get_chat_history(self, compacted: bool = True) -> Iterable[ChatHistoryItem]:
        """Yield LLM-facing chat history, using available summaries by default."""
        yield from self._iter_chat_history(compacted=compacted)

    def get_request_data(self, turn_num: int) -> Dict[str, Any]:
        """Reconstruct the request params used for a historical LLM turn."""
        from statek.executors.job import _get_static_task_difficulty  # pylint: disable=import-outside-toplevel

        if turn_num < 0:
            raise IndexError(f"turn_num out of range: {turn_num}")

        target_idx = None
        target_item = None
        console_limit = 0
        llm_turn_num = 0
        for idx, item in enumerate(self._job.chat_log):
            if not isinstance(item, LLM_LogItem) or item.llm_resp is None:
                continue
            if llm_turn_num == turn_num:
                target_idx = idx
                target_item = item
                console_limit = item.console_pos
                break
            llm_turn_num += 1

        if target_idx is None:
            raise IndexError(f"turn_num out of range: {turn_num}")

        console = list(self._job.py_env.iter_console())
        push_log = dict(self._job.py_env.iter_push_log())
        history_chat_log = self._job.chat_log[:target_idx]
        history_console = console[:console_limit]
        history_push_log = {
            key: value for key, value in push_log.items()
            if key <= console_limit
        }
        request_difficulty = getattr(target_item, "request_difficulty", None)
        if request_difficulty is None:
            request_difficulty = _get_static_task_difficulty(self._job.job_def.metadata)
        return self._job._build_request_data(  # pylint: disable=protected-access
            chat_log=history_chat_log,
            console=history_console,
            push_log=history_push_log,
            difficulty=request_difficulty,
        )

    @property
    def last_chat_log_item(self) -> Optional[Union[str, ChatLogItem, UserLogItem]]:
        """Return the current conversation/execution item, ignoring budget metadata."""
        return next((item for item in reversed(self._job.chat_log)
                     if not isinstance(item, DifficultyChangeLogItem)), None)

    @property
    def last_response(self) -> Union[str, CodeBlock, None]:
        """
        Retrieves the last response received from the LLM (i.e. the Python code to be executed,
        or a CodeBlock when the LLM requested tool calls) or None if chat_log is empty.
        """
        if not self._job.chat_log:
            return None
        last = self._job.last_chat_log_item
        if isinstance(last, LLM_LogItem):
            return last.llm_resp
        return None

    def _is_user_facing_llm_response(self, item: LLM_LogItem) -> bool:
        """Return True when the LLM log item corresponds to a visible reply.

        ``get_response_times()`` tracks user-visible latency, so internal
        execution turns in dialog styles should not count as the first reply.
        """
        response = item.llm_resp
        if response is None:
            return False

        chat_style = self._job.job_def.chat_style
        if chat_style == ChatStyle.DIRECT:  # pylint: disable=no-member
            return isinstance(response, str) and bool(response.strip())

        if chat_style == ChatStyle.MD_DIALOG:  # pylint: disable=no-member
            if isinstance(response, CodeBlock):
                return False
            if not isinstance(response, str):
                return False
            return any(line.startswith("# ") for line in response.splitlines())

        return True

    @staticmethod
    def _chat_response_from_tool_log(line: str) -> Optional[str]:
        """Extract a user-facing answer body from a console tool-log line."""
        call_params = parse_tool_log(line)
        if call_params is None or call_params.name != "answer":
            return None

        body = None
        if call_params.args:
            body = call_params.args[0]
        if call_params.kwargs:
            body = call_params.kwargs.get("body", body)
            body = call_params.kwargs.get("content", body)
        if not isinstance(body, str) or not body:
            return None
        return body

    def _iter_chat_response_tool_logs(self, from_pos: int, to_pos: int) -> Iterable[str]:
        """Yield answer bodies from console output in ``[from_pos, to_pos)``."""
        for output in self._job.py_env.iter_console(from_pos, to_pos):
            for line in output.splitlines():
                body = self._chat_response_from_tool_log(line)
                if body is not None:
                    yield body

    @staticmethod
    def _dialog_from_code_block(resp: CodeBlock) -> Optional[str]:
        """Extract dialog from a raw MD_DIALOG response stored as a CodeBlock."""
        return extract_dialog(resp.code or "")

    def get_chat_responses(self) -> Iterable[str]:
        """Yield user-facing chat responses from the chat log, oldest first."""
        console_len = self._job.py_env.console_len
        items = [
            item for item in self._job.chat_log
            if isinstance(item, (WarmupLogItem, LLM_LogItem))
        ]
        end_positions = [
            items[idx + 1].console_pos if idx + 1 < len(items) else console_len
            for idx in range(len(items))
        ]

        for idx, item in enumerate(items):
            if isinstance(item, LLM_LogItem) and self._is_user_facing_llm_response(item):
                resp = item.llm_resp
                if isinstance(resp, str):
                    yield resp
                elif isinstance(resp, CodeBlock) and resp.code:
                    yield resp.code
            yield from self._iter_chat_response_tool_logs(
                item.console_pos,
                end_positions[idx],
            )

    def get_dialog(self) -> Iterable[DialogItem]:
        """Yield user/assistant dialog messages in chronological order."""
        from statek.executors.job import DialogItem  # pylint: disable=import-outside-toplevel

        console_len = self._job.py_env.console_len
        turn_items = [
            item for item in self._job.chat_log
            if isinstance(item, (WarmupLogItem, LLM_LogItem))
        ]
        end_positions = {
            id(item): (
                turn_items[idx + 1].console_pos
                if idx + 1 < len(turn_items) else console_len
            )
            for idx, item in enumerate(turn_items)
        }

        for item in self._job.chat_log:
            if isinstance(item, str):
                if item:
                    yield DialogItem("user", item)
                continue

            if isinstance(item, UserLogItem):
                if item.message:
                    yield DialogItem("user", item.message)
                continue

            if not isinstance(item, (WarmupLogItem, LLM_LogItem)):
                continue

            if isinstance(item, LLM_LogItem) and self._is_user_facing_llm_response(item):
                resp = item.llm_resp
                if isinstance(resp, str):
                    yield DialogItem("assistant", resp)
            elif (
                isinstance(item, LLM_LogItem)
                and self._job.job_def.chat_style == ChatStyle.MD_DIALOG  # pylint: disable=no-member
                and isinstance(item.llm_resp, CodeBlock)
            ):
                dialog = self._dialog_from_code_block(item.llm_resp)
                if dialog:
                    yield DialogItem("assistant", dialog)

            for body in self._iter_chat_response_tool_logs(
                item.console_pos,
                end_positions[id(item)],
            ):
                yield DialogItem("assistant", body)

    def get_response_times(self) -> Iterable[tuple[datetime, Optional[float]]]:
        """Return user-message timestamps paired with time until their wait ended.

        Only timestamped user messages stored in ``chat_log`` are included:
        the initial dialog-style message and subsequent ``UserLogItem`` entries.
        Dialog timing reads each message's ``wait_finished_at``; an unset value
        means its wait is still pending. Non-dialog jobs retain their LLM-log timing.
        """
        response_times: List[tuple[datetime, Optional[float]]] = []
        pending_indices: List[int] = []
        if self._job.job_def.chat_style in (ChatStyle.DIRECT, ChatStyle.MD_DIALOG):
            for item in self._job.chat_log:
                if isinstance(item, UserLogItem) and item.message:
                    response_times.append((
                        item.timestamp,
                        (item.wait_finished_at - item.timestamp).total_seconds()
                        if item.wait_finished_at is not None else None,
                    ))
            return response_times

        for item in self._job.chat_log:
            if isinstance(item, str):
                if item and self._job.created_at is not None:
                    pending_indices.append(len(response_times))
                    response_times.append((self._job.created_at, None))
                continue

            if isinstance(item, UserLogItem):
                if item.message:
                    pending_indices.append(len(response_times))
                    response_times.append((item.timestamp, None))
                continue

            if isinstance(item, LLM_LogItem):
                if not pending_indices or not self._is_user_facing_llm_response(item):
                    continue
                for idx in pending_indices:
                    message_time, _ = response_times[idx]
                    response_times[idx] = (
                        message_time,
                        (item.timestamp - message_time).total_seconds(),
                    )
                pending_indices = []

        return response_times

    def get_llm_response_times(self) -> Iterable[tuple[datetime, Optional[float]]]:
        """Return per-request LLM timings from the chat log.

        Each tuple contains the request timestamp and time-to-response in
        seconds. Explicit pending ``LLM_LogItem`` markers (``llm_resp=None``)
        are treated as the cleanest request boundary and paired with the next
        concrete ``LLM_LogItem``. When no pending marker exists, the first
        concrete response is measured from ``created_at`` and later concrete
        ``LLM_LogItem`` entries use the previous response timestamp as an
        approximate boundary.

        Non-LLM entries are ignored. A trailing pending marker with no later
        response is returned with ``None`` as its duration.
        """
        response_times: List[tuple[datetime, Optional[float]]] = []
        pending_request_time: Optional[datetime] = None
        previous_response_time: Optional[datetime] = None

        for item in self._job.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue

            if item.llm_resp is None:
                pending_request_time = item.timestamp
                continue

            request_time = pending_request_time or previous_response_time
            if request_time is None and self._job.created_at is not None:
                if self._job.created_at <= item.timestamp:
                    request_time = self._job.created_at
            if request_time is not None:
                response_times.append((
                    request_time, (item.timestamp - request_time).total_seconds(),
                ))
            pending_request_time = None
            previous_response_time = item.timestamp

        if pending_request_time is not None:
            response_times.append((pending_request_time, None))

        return response_times

    def get_llm_initial_step_size(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) for the constant initial portion.

        Covers the system prompt and all warmup code blocks — the fixed overhead
        sent with every first LLM request. Token counts use the ``len // 4``
        approximation. Output tokens are always zero (no LLM output at this stage).
        """
        system_prompt = self._job.system_prompt()
        warmup = self._job.job_def.warmup_code if self._job.job_def else None
        warmup_text = _warmup_code_text(warmup)
        return (len(system_prompt) + len(warmup_text)) // 4, 0

    def get_llm_step_sizes(self) -> Iterable[tuple[int, int]]:
        """Return (input_tokens, output_tokens) per LLM step, ordered as get_llm_response_times.

        One entry is yielded per concrete ``LLM_LogItem`` (``llm_resp`` is not
        ``None``) plus one entry for a trailing pending marker if present.
        Token counts use the ``len // 4`` approximation.

        Input tokens for each step cover: console output produced since the
        previous step, tool log from the previous step, and any user messages
        accumulated since then. The first step additionally includes the initial
        step size (system prompt + warmup code).

        Output tokens cover the LLM response text plus the JSON encoding of any
        tool call requests.
        """
        console = list(self._job.py_env.iter_console())
        is_first = True
        prev_console_pos = 0
        prev_item = None
        accumulated_text = ""
        trailing_pending = None

        for item in self._job.chat_log:
            if isinstance(item, str):
                accumulated_text += item
                continue
            if isinstance(item, UserLogItem):
                accumulated_text += item.message or ""
                continue
            if not isinstance(item, LLM_LogItem):
                continue
            console_text = "\n".join(console[prev_console_pos:item.console_pos])
            tool_text = _tool_log_text(prev_item)
            input_tokens = (len(console_text) + len(tool_text) + len(accumulated_text)) // 4
            if is_first:
                input_tokens += self._job.get_llm_initial_step_size()[0]
            if item.llm_resp is None:
                trailing_pending = input_tokens
                continue
            is_first = False
            trailing_pending = None
            yield (input_tokens, _llm_resp_output_tokens(item.llm_resp))
            accumulated_text = ""
            prev_console_pos = item.console_pos
            prev_item = item

        if trailing_pending is not None:
            yield (trailing_pending, 0)

    def tokens_per_sec(self) -> float:
        """Return total tokens divided by total LLM processing time in seconds.

        Combines ``get_llm_response_times`` and ``get_llm_step_sizes`` to
        compute the overall throughput metric. Steps with unknown duration
        (``None``) are excluded from both totals. Returns ``0.0`` when no
        step has a known duration.
        """
        total_tokens = 0
        total_seconds = 0.0
        for (_, duration), (input_tok, output_tok) in zip(
            self._job.get_llm_response_times(), self._job.get_llm_step_sizes()
        ):
            if duration is None:
                continue
            total_tokens += input_tok + output_tok
            total_seconds += duration
        if total_seconds == 0.0:
            return 0.0
        return total_tokens / total_seconds

    @property
    def num_turns(self) -> int:
        """Returns the number of LLM turns (excludes warmup code blocks)."""
        return sum(1 for item in self._job.chat_log if isinstance(item, LLM_LogItem))

    def count_llm_actions(self) -> int:
        """Return the number of LLM actions recorded in chat_log.

        This differs from ``num_turns``: ``num_turns`` counts LLM log items,
        while this method counts finer-grained work inside those items. One
        non-empty executable code block, one requested tool call, and one
        user-facing response each count as one action. Framework-generated
        messages such as warmup, reminder, subtask, user, and post-processor
        items are ignored.

        Returns:
            The total number of recorded LLM actions.
        """
        action_count = 0
        for item in self._job.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue

            response = item.llm_resp
            if isinstance(response, CodeBlock):
                if response.code and response.code.strip():
                    action_count += 1
                if response.tool_calls:
                    action_count += len(response.tool_calls)
                continue

            if isinstance(response, str) and self._is_user_facing_llm_response(item):
                action_count += 1

        return action_count

    @property
    def _warmup_console_positions(self) -> set:
        """Returns console_pos values that correspond to warmup blocks."""
        return {item.console_pos for item in self._job.chat_log if isinstance(item, WarmupLogItem)}

    def _exception_messages_by_turn(self) -> Iterable[Tuple[ChatLogItem, List[str]]]:
        """Resolve console events once; latest equal execution boundary wins.

        Error console entries belong to half-open execution ranges.
        """
        items = [item for item in self._job.chat_log
                 if isinstance(item, (LLM_LogItem, WarmupLogItem))]
        errors = list(self._job.py_env.iter_exceptions())
        position = 0
        for index, item in enumerate(items):
            end = items[index + 1].console_pos if index + 1 < len(items) else None
            messages = []
            while position < len(errors) and (end is None or errors[position][0] < end):
                key, error = errors[position]
                position += 1
                if key >= item.console_pos and error.kind == ErrorKind.EXECUTION:
                    messages.append(error.message)
            tool_log = item.tool_log
            if isinstance(tool_log, ToolError):
                messages.append(tool_log.err_message)
            elif tool_log is not None and not isinstance(tool_log, str):
                messages.extend(
                    entry.err_message for entry in tool_log if isinstance(entry, ToolError)
                )
            yield item, messages

    @property
    def exception_count(self) -> int:
        """Returns the total number of exceptions from LLM turns (excludes warmup blocks).

        Counts both code-execution exceptions (py_env.exceptions) and
        tool-call exceptions (ToolError entries in tool_log).
        """
        return sum(len(messages) for item, messages in self._exception_messages_by_turn()
                   if isinstance(item, LLM_LogItem))

    @property
    def max_consecutive_exceptions(self) -> int:
        """Returns the maximum number of consecutive exceptions among LLM turns only.

        Considers both code-execution exceptions and tool-call exceptions.
        """
        if not self._job.chat_log:
            return 0
        max_streak = 0
        streak = 0
        for item, messages in self._exception_messages_by_turn():
            if not isinstance(item, LLM_LogItem):
                continue
            if messages:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @property
    def approx_token_usage(self) -> int:
        """Calculates approximate token usage based on total bytes sent and received."""
        return (self._job.usage.total_bytes_sent + self._job.usage.total_bytes_received) // 4
