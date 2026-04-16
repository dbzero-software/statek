from dataclasses import dataclass
from datetime import datetime
import re
import traceback as _traceback_module
from typing import Callable, List, Optional, Iterable, Dict, Any, Sequence, Union
import dbzero as db0
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.executors.chat_log_item import ChatLogItem, LLM_LogItem, ToolError, WarmupLogItem, UserLogItem
from statek.llm_api import LLM_Response, CallParams
from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.utils import (prompt_append_console, CodeBlock, CallSpec, CallSpecWrapper,
                          strip_markup, extract_dialog,
                          parse_warmup_block, build_warmup_code, _STATEK_TOOL_MARKER)
from statek.future import FutureResult
from statek.locale import get_language_rule, get_language_hint
from statek.settings import get_statek_settings, ChatStyle, statek_log

"""
READY: a fresh job instance ready for execution
WARMING_UP: executing startup code before first LLM interaction
STARTED: job execution in progress
SUSPENDED: job execution suspended - waiting on external events
DONE: execution has been completed (with either success or failure)
"""
@enum(values=["READY", "WARMING_UP", "STARTED", "SUSPENDED", "DONE"])
class JobStatus:
    pass


def parse_warmup_code(
    warmup_code: Optional[Union[str, Sequence[str]]],
) -> Optional[Union[str, "CodeBlock", List[Union[str, "CodeBlock"]]]]:
    """Parse warmup_code through parse_warmup_block and build_warmup_code.

    If warmup_code is a string containing comment lines with 10+ dashes
    (e.g. # ----------), it will be split into multiple blocks first.
    Each block is then parsed for tool-call annotations and converted into
    a plain string (no tool calls) or a CodeBlock (tool calls present).

    Args:
        warmup_code: Single string, sequence of strings, or None

    Returns:
        None if input is None or results in no blocks
        Single str or CodeBlock if only one block
        List of str and/or CodeBlock if multiple blocks
    """
    if warmup_code is None:
        return None

    if isinstance(warmup_code, str):
        # Split on comment lines containing 10 or more dashes (e.g. # ----------)
        raw_blocks = re.split(r'\n\s*#\s*-{10,}\s*\n', warmup_code)
    else:
        # Already a sequence — each item is a separate block
        raw_blocks = list(warmup_code)

    # Strip each block and filter empty ones
    raw_blocks = [block.strip() for block in raw_blocks if block.strip()]

    if not raw_blocks:
        return None

    parsed_blocks = [parse_warmup_block(block) for block in raw_blocks]
    return build_warmup_code(parsed_blocks)


@memo
class JobDefError:
    """Container for storing error information from a job failure."""
    error_message: str
    """Populated when the traceback collection is turned on"""
    traceback: Optional[List[str]] = None

    def __init__(self, error: Exception, collect_traceback: bool = True):
        """Create JobDefError initialized with information from exception."""
        self.error_message = str(error)
        if collect_traceback and error.__traceback__ is not None:
            self.traceback = _traceback_module.format_tb(error.__traceback__)
        else:
            self.traceback = None


@memo
@dataclass
class JobDef:
    """
    The `JobDescr` instances, as the name suggests - hold job descriptions / definitions.
    """
    # An agent assigned to this job
    agent: "Agent"
    # Job params to be fed into the agent's prompt template
    job_params: Optional[Dict[str, Any]] = None
    # Optional warmup code (single block or sequence of blocks) executed before the first prompt
    warmup_code: Optional[Union[str, CodeBlock, Sequence[Union[str, CodeBlock]]]] = None
    # Optional job-level chat style override (falls back to StatekSettings.chat_style)
    _chat_style: Optional[ChatStyle] = None
    # Optional locale for language-specific behaviour
    locale: Optional["StatekLocale"] = None
    # Frozen LLM model family for jobs created from this definition
    model_family: Optional[str] = None
    # Frozen LLM model for jobs created from this definition
    model: Optional[str] = None

    def __post_init__(self):
        if self.agent is not None:
            db0.tags(self).add(self.agent)
        metadata = (
            self.agent._metadata
            if self.agent is not None and self.agent._metadata
            else {}
        )
        metadata_model = metadata.get('MODEL')
        metadata_model_family = metadata.get('MODEL_FAMILY')
        if self.model is None and metadata_model is None:
            role = self.agent.role if self.agent is not None else '<unknown>'
            raise ValueError(
                f"JobDef for agent '{role}' requires metadata field 'MODEL'"
            )
        if self.model_family is None:
            self.model_family = (
                metadata_model_family
                or (
                    (self.model or metadata_model).split('/', 1)[0]
                    if (self.model or metadata_model) and '/' in (self.model or metadata_model)
                    else None
                )
            )
        if self.model is None:
            self.model = metadata_model
        if metadata_model_family is None and self.model_family is None and self.model and '/' in self.model:
            self.model_family = self.model.split('/', 1)[0]

    def set_error(self, error: Exception, collect_traceback: bool = True) -> None:
        """Create a JobDefError from the given exception and associate it with this JobDef."""
        jde = JobDefError(error, collect_traceback=collect_traceback)
        db0.tags(jde).add(db0.as_tag(self))

    def get_errors(self) -> Iterable["JobDefError"]:
        """Yield all JobDefError instances associated with this JobDef."""
        yield from db0.find(JobDefError, db0.as_tag(self))

    def has_errors(self) -> bool:
        """Return True if any errors are associated with this JobDef."""
        return any(True for _ in self.get_errors())

    def clear_errors(self):
        """Remove all JobDefError instances associated with this JobDef."""
        tag = db0.as_tag(self)
        for err in list(self.get_errors()):
            db0.tags(err).remove(tag)
        
    def update_warmup_code(
        self,
        warmup_code: Optional[Union[str, Sequence[str]]],
    ) -> None:
        """Update warmup_code only when the parsed new value differs from the current one.

        Parses the input through parse_warmup_code before comparing. A debug
        message is logged only when an actual change is applied.

        Args:
            warmup_code: New warmup code (raw string, sequence of strings, or None).
        """
        new_value = parse_warmup_code(warmup_code)
        if new_value != self.warmup_code:
            statek_log(f"updating warmup code: {new_value!r}", level='debug')
            self.warmup_code = new_value

    def set_chat_style(self, chat_style: Optional[ChatStyle]) -> None:
        """Update _chat_style only when the new value differs from the current one.

        Args:
            chat_style: New chat style value, or None to clear the override.
        """
        if self._chat_style != chat_style:
            self._chat_style = chat_style

    @property
    def chat_style(self) -> Optional[ChatStyle]:
        """Return the job-level chat style, or fall back to the system default from StatekSettings."""
        if self._chat_style is not None:
            return self._chat_style
        return get_statek_settings().chat_style

    @property
    def system_prompt(self) -> str:
        """Return the agent's system prompt formatted with job-specific parameters."""
        if self.agent is None:
            return ""
        return self.agent.system_prompt(job_params=self.job_params)



@memo
@dataclass
class ErrorHandler:
    """Holds an error-handling callable together with the context it should be called with.

    The ``__call__`` method forwards the underlying handler with ``context`` and an
    optional exception, suppressing any exception the handler itself might raise.
    """
    error_handler: Callable
    context: Any

    def __call__(self, error: Optional[Exception] = None):
        try:
            self.error_handler(self.context, error=error)
        except Exception:  # pylint: disable=broad-except
            pass


@memo
class Job:
    """
    A single "job" is a stateful class representing the current state of a single unit-of-work, being performed end-to-end by a single agent. By a "job" we might mean either a very simple operation such as answering a basic question ("Hey, what day of week is today") or a complex task involving retrieving information from external systems, communicating with external actors, waiting for approvals etc. - before the final response is generated.
    """

    def __init__(
        self,
        job_def: JobDef,
        model_family: Optional[str] = None,
        model: Optional[str] = None,
        job_status: JobStatus = JobStatus.READY,
        session_id: str = None,
        py_env: PyEnv = None,
        chat_log: List[Union[str, ChatLogItem, UserLogItem]] = None,
        awaited_result: Optional[FutureResult] = None,
        next_instr_num: Optional[int] = None,
        warmup_block_num: Optional[int] = None,
        error: Optional[JobDefError] = None,
        created_at: Optional[datetime] = None,
    ):
        del model_family, model  # backward-compatible init args; source of truth is JobDef
        self.job_def = job_def
        if self.job_def.agent is not None:
            db0.tags(self).add(self.job_def.agent)
        # Private job status attribute
        self.__job_status = None
        self.set_status(job_status)
        # Associated LLM API's session ID (where available)
        self.session_id = session_id
        # LLM program's execution environment
        self.py_env = py_env if py_env is not None else PyEnv()
        # Current chat state
        self.chat_log = chat_log if chat_log is not None else []
        # Suspended job's awaited result
        self.awaited_result = awaited_result
        # Continuation instruction number
        self.next_instr_num = next_instr_num
        # Continuation warmup block number (for multi-block warmup_code)
        self.warmup_block_num = warmup_block_num
        # Error information if the job failed
        self.error: Optional[JobDefError] = error
        # Timestamp when the job object was created
        self.created_at = created_at or datetime.now()
        # Registered error handlers (ErrorHandler instances)
        self.error_handlers: List[ErrorHandler] = []
        # Total context bytes used by this job so far
        self.context_bytes = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        # Total cost as reported by the LLM API provider
        self.total_cost = 0.0
        # Persistent execution context (created on demand by perm_ctx_set)
        self.perm_ctx: Optional[dict] = None
        # Number of completed DONE transitions (None until first completion)
        self.num_completions: Optional[int] = None

        # Log system prompt on job creation if logging is enabled
        if self.logs_path and self.job_def.agent is not None:
            self._log(self.job_def.system_prompt)

    @property
    def model_family(self) -> Optional[str]:
        """Return the frozen model family stored on the job definition."""
        return self.job_def.model_family if self.job_def is not None else None

    @property
    def model(self) -> Optional[str]:
        """Return the frozen model stored on the job definition."""
        return self.job_def.model if self.job_def is not None else None

    def add_error_handler(self, error_handler: Callable, context: Any) -> None:
        """Register an error handler with this job.

        Args:
            error_handler: The error-handling callable (must satisfy the
                error-handler protocol, i.e. decorated with
                ``@statek.error_handler``).
            context: The context value passed as the first argument when the
                handler is invoked.
        """
        self.error_handlers.append(ErrorHandler(error_handler=error_handler, context=context))

    def add_error_handlers_from(self, parent_job: 'Job') -> None:
        """Copy all registered error handlers from *parent_job* into this job."""
        self.error_handlers.extend(parent_job.error_handlers)

    def notify_handlers(self, error: Optional[Exception] = None) -> None:
        """Invoke all registered error handlers and clear the handler list.

        Args:
            error: Optional exception that caused the job to terminate.
        """
        handlers, self.error_handlers = self.error_handlers, []
        for handler in handlers:
            handler(error=error)

    @property
    def logs_path(self) -> Optional[str]:
        """Get the logs path from StatekSettings."""
        return get_statek_settings().logs_path

    def _log(self, content: str):
        """
        Write content to the log file and to the console logger.
        
        Args:
            content: The content to log
        """
        if not self.logs_path:
            return

        from statek.settings import get_statek_logger  # pylint: disable=import-outside-toplevel
        import os

        # Write to log file
        agent_name = self.job_def.agent.role if self.job_def.agent else "unknown"
        job_uuid = db0.uuid(db0.materialized(self))
        log_filename = f"{agent_name}_{job_uuid}.log"
        log_filepath = os.path.join(self.logs_path, log_filename)
        os.makedirs(self.logs_path, exist_ok=True)
        with open(log_filepath, 'a', encoding='utf-8') as f:
            f.write(f"{content}\n\n")

        # Write to console logger
        logger = get_statek_logger()
        logger.info("%s", content)

    def _log_console_batch(self, from_pos: int, to_pos: int):
        """Log a slice of the console as a single batch with XML tags if configured.

        Called at block/turn boundaries so the log matches the format sent to the LLM.

        Args:
            from_pos: First console element to include
            to_pos: One-past-the-last console element to include
        """
        if to_pos <= from_pos:
            return
        settings = get_statek_settings()
        text = prompt_append_console(
            self.py_env.console, self.job_def.chat_style,
            from_pos=from_pos, limit=to_pos - from_pos,
            xml_tags=settings.get_xml_box_tags()
        )
        if text:
            self._log(text)

    def _log_tool_call_result(self, call_spec, result: str):
        """Log a tool call and its result.

        The call line is annotated with the STATEK tool marker.
        The result is formatted like console output (``> ``-prefixed in CONSOLE
        style, plain in MARKDOWN style) and wrapped in XML console tags when
        configured.

        Args:
            call_spec: The CallSpec describing the tool call
            result: The tool execution result string
        """
        settings = get_statek_settings()
        chat_style = self.job_def.chat_style
        raw_call = f"{call_spec.format()}  {_STATEK_TOOL_MARKER}"
        if chat_style in (ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
            call_line = f"```python\n{raw_call}\n```"
        else:
            call_line = raw_call
        if result:
            result_lines = result.split('\n')
            if chat_style == ChatStyle.MD_DIALOG:  # pylint: disable=no-member
                result_text = '<CONSOLE>\n' + '\n'.join(result_lines) + '\n</CONSOLE>'
            elif chat_style in (ChatStyle.MARKDOWN, ChatStyle.DIRECT):  # pylint: disable=no-member
                result_text = '\n'.join(result_lines)
            else:
                result_text = '\n'.join(f"> {line}" for line in result_lines)
            xml_tags = settings.get_xml_box_tags()
            tag = xml_tags and xml_tags.get("console") if chat_style != ChatStyle.DIRECT else None  # pylint: disable=no-member
            if tag:
                result_text = f"<{tag}>\n{result_text}\n</{tag}>"
            self._log(f"{call_line}\n{result_text}")
        else:
            self._log(call_line)

    def console_append(self, output: str, error_message: str = None):
        """
        Append output to the console and optionally log it.

        Args:
            output: The output string to append
            error_message: optional error message (if execution resulted in an exception)
        """
        self.py_env.console_append(output)
        if error_message is not None:
            console_pos = self.chat_log[-1].console_pos if self.chat_log else 0
            if self.py_env.exceptions is None:
                self.py_env.exceptions = {}
            self.py_env.exceptions[console_pos] = error_message
        # Console is logged in batches at block/turn boundaries (see _log_console_batch)

    @property
    def status(self) -> JobStatus:
        """
        Returns the current job status.

        Returns:
            JobStatus: The current status of the job
        """
        return self.__job_status

    def set_status(self, new_status: JobStatus):
        """
        Sets or updates job status. If state is updated - existing tag is removed
        and new status tag applied.

        Args:
            new_status: The status/tag to be assigned or updated
        """
        if self.__job_status is not None:
            db0.tags(self).remove(self.__job_status)
        db0.tags(self).add(new_status)
        self.__job_status = new_status

    def _language_hint_suffix(self) -> str:
        """Return the language hint suffix for push_log messages, or empty string.

        Returns a string like ``' (PAMIĘTAJ: ...)'`` when the job's locale
        specifies a non-EN language and AUTO_LANG_HINT is not disabled.
        """
        locale = self.job_def.locale
        if locale is None:
            return ""
        metadata = self.job_def.agent._metadata or {}  # pylint: disable=protected-access
        if metadata.get("AUTO_LANG_HINT", "").upper() == "FALSE":
            return ""
        hint = get_language_hint(locale.lang_code)
        if hint:
            return f" ({hint})"
        return ""

    def _with_language_hint(self, message: str) -> str:
        """Append the locale language hint to a real user message when enabled."""
        if not message:
            return message
        return f"{message}{self._language_hint_suffix()}"

    def _collect_push_log(self, from_pos: int) -> str:
        """Return push_log messages with key >= from_pos as a newline-joined string.

        Args:
            from_pos: Include only entries whose console-position key is >= this value.

        Returns:
            Newline-joined string of all matching messages, or empty string if none.
        """
        if not self.py_env.push_log:
            return ""
        hint_suffix = self._language_hint_suffix()
        parts = []
        for key in sorted(self.py_env.push_log.keys()):
            if key >= from_pos:
                value = self.py_env.push_log[key]
                if not isinstance(value, str):
                    parts.extend(f"{v}{hint_suffix}" for v in value)
                else:
                    parts.append(f"{value}{hint_suffix}")
        return "\n".join(parts)

    def get_next_prompt(self) -> str:
        """
        Generate the next prompt to be included in the LLM chat.

        If this is the first prompt (chat_log is empty), use job_def.prompt
        and append the entire py_env console starting from position 0.
        Otherwise, format the console starting from the last chat element's
        console position to provide the console result for LLM analysis.

        If push_log is available, push_log entries whose console-position key falls
        within the relevant range are appended after the console output.

        Returns:
            The formatted prompt string ready to be sent to the LLM
        """
        chat_style = self.job_def.chat_style
        xml_tags = get_statek_settings().get_xml_box_tags()
        # Filter for LLM items only (WarmupLogItems don't count as LLM turns)
        llm_items = [item for item in self.chat_log if isinstance(item, LLM_LogItem)]
        if not llm_items:
            warmup = self.job_def.warmup_code
            if warmup:
                # Warmup exists: template and warmup turns go in chat_history as
                # alternating assistant/user pairs.  The prompt (current user message)
                # is the console output OF the last warmup block, spanning from the
                # end of the previous block to the end of the last block.
                warmup_blocks = [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)
                n = len(warmup_blocks)
                positions = self._warmup_end_positions()
                last_end = (
                    positions[n - 1]
                    if n - 1 < len(positions)
                    else len(self.py_env.console) if self.py_env.console else 0
                )
                prev_end = (
                    positions[n - 2]
                    if n >= 2 and n - 2 < len(positions)
                    else 0
                )
                limit = last_end - prev_end
                console_part = prompt_append_console(
                    self.py_env.console, chat_style,
                    from_pos=prev_end, limit=limit if limit > 0 else 0,
                    xml_tags=xml_tags
                ) if limit > 0 else ""
                push_part = self._collect_push_log(from_pos=prev_end)
                parts = [p for p in [console_part, push_part] if p]
                return "\n".join(parts)
            else:
                # No warmup: system_prompt + all console is the first user prompt
                sys_prompt = self.job_def.system_prompt or None
                console_part = prompt_append_console(
                    self.py_env.console, chat_style, from_pos=0, xml_tags=xml_tags
                )
                push_part = self._collect_push_log(from_pos=0)
                parts = [p for p in [sys_prompt, console_part, push_part] if p]
                return "\n".join(parts)
        else:
            # Not first prompt: format console from last LLM element's console position
            from_pos = llm_items[-1].console_pos
            console_part = prompt_append_console(
                self.py_env.console,
                chat_style,
                from_pos=from_pos,
                xml_tags=xml_tags
            )
            push_part = self._collect_push_log(from_pos=from_pos)
            parts = [p for p in [console_part, push_part] if p]
            return "\n".join(parts)

    def get_chat_history(self) -> Iterable[ChatHistoryItem]:
        """Yield the full LLM-facing chat history as ``ChatHistoryItem`` objects.

        Lazy generator producing a normalised, provider-agnostic timeline:

          1. Optional USER item with the initial user message — sourced from
             ``chat_log[0]`` (str, set by ``push_user_message`` in
             MD_DIALOG/DIRECT) and/or ``py_env.push_log[0]``.  When neither
             exists this slot is omitted.
          2. Warmup blocks as alternating ASSISTANT (code or tool-call request)
             + USER/TOOL (console output / tool result) pairs, in
             ``warmup_block_num`` order.
          3. The same alternating pattern for LLM turns: ASSISTANT (LLM
             response) + USER/TOOL (console / tool result), woven together
             with any USER messages from ``UserLogItem`` entries and from
             ``py_env.push_log`` keyed by console position.

        ``CodeBlock`` responses with ``tool_calls`` produce an ASSISTANT item
        with ``tool_calls`` set, followed by one TOOL item per call carrying
        its captured result.  Plain code (no tool calls) becomes an
        ASSISTANT item with ``content`` set.  Pending LLM turns (e.g.
        ``llm_resp=None`` after a DONE→STARTED transition) emit no assistant
        item but still surface any preceding push_log user messages.

        Yields:
            ChatHistoryItem instances ready for ``format_chat_history_item``.
        """
        push_log = self.py_env.push_log or {}
        console = self.py_env.console or []
        console_len = len(console)
        is_direct = self.job_def.chat_style == ChatStyle.DIRECT  # pylint: disable=no-member
        warmup = self.job_def.warmup_code
        warmup_blocks: List[Union[str, CodeBlock]] = (
            [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)
        ) if warmup else []

        # 1) Initial user message — chat_log[0] (str) and/or push_log[0].
        initial_parts: List[str] = []
        if (
            self.chat_log
            and isinstance(self.chat_log[0], str)
            and self.chat_log[0]
        ):
            initial_parts.append(self._with_language_hint(self.chat_log[0]))
        if 0 in push_log:
            v = push_log[0]
            if isinstance(v, list):
                initial_parts.extend(
                    self._with_language_hint(s) for s in v if s
                )
            elif v:
                initial_parts.append(self._with_language_hint(v))
        if initial_parts:
            yield ChatHistoryItem(
                role=ChatRole.USER,
                content="\n".join(initial_parts),
                content_src=ContentSource.USER,
            )

        # 2) Walk chat_log items in chronological order.
        items = [it for it in self.chat_log if not isinstance(it, str)]

        # Pre-compute the console end position for each ChatLogItem (the
        # console_pos of the next ChatLogItem, or console_len for the last).
        end_positions: Dict[int, int] = {}
        for i, item in enumerate(items):
            if isinstance(item, UserLogItem):
                continue
            next_pos = console_len
            for j in range(i + 1, len(items)):
                if isinstance(items[j], (WarmupLogItem, LLM_LogItem)):
                    next_pos = items[j].console_pos
                    break
            end_positions[i] = next_pos

        # Push log keys (excluding 0, already consumed as the initial user msg).
        sorted_push_keys = sorted(k for k in push_log.keys() if k != 0)

        def _yield_pushes(lo: int, hi: int) -> Iterable[ChatHistoryItem]:
            """Yield USER items for push_log entries in (lo, hi]."""
            for k in sorted_push_keys:
                if lo < k <= hi:
                    v = push_log[k]
                    msgs = v if isinstance(v, list) else [v]
                    for msg in msgs:
                        yield ChatHistoryItem(
                            role=ChatRole.USER,
                            content=self._with_language_hint(msg),
                            content_src=ContentSource.USER,
                        )

        for idx, item in enumerate(items):
            if isinstance(item, UserLogItem):
                if item.message:
                    yield ChatHistoryItem(
                        role=ChatRole.USER,
                        content=self._with_language_hint(item.message),
                        content_src=ContentSource.USER,
                    )
                continue

            # Determine the source content of the assistant turn.
            if isinstance(item, WarmupLogItem):
                block_num = item.warmup_block_num
                block = (
                    warmup_blocks[block_num]
                    if block_num < len(warmup_blocks) else None
                )
                asst_src = ContentSource.SYSTEM
            elif isinstance(item, LLM_LogItem):
                block = item.llm_resp
                asst_src = ContentSource.ASSISTANT
            else:
                continue

            # Yield the assistant turn (and any tool results).
            if block is not None:
                tool_calls = (
                    list(block.tool_calls)
                    if isinstance(block, CodeBlock) and block.tool_calls
                    else None
                )
                block_code = block.code if isinstance(block, CodeBlock) else block

                # DIRECT style: wrap plain warmup code as python_cli tool calls
                # so the LLM sees them as tool use, not confusing text/console.
                if is_direct and isinstance(item, WarmupLogItem) and not tool_calls and block_code:
                    warmup_call = CallSpecWrapper(
                        id=f"STATEK-WARMUP-{item.warmup_block_num:03d}",
                        func_name="python_cli",
                        args=None,
                        kwargs={"code": block_code},
                    )
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=None,
                        content_src=asst_src,
                        tool_calls=[warmup_call],
                    )
                    # Console output as TOOL result instead of USER message
                    from_pos = item.console_pos
                    to_pos = end_positions[idx]
                    console_text = "\n".join(console[from_pos:to_pos]) if to_pos > from_pos and console else ""
                    yield ChatHistoryItem(
                        role=ChatRole.TOOL,
                        content=console_text,
                        content_src=ContentSource.CONSOLE,
                        tool_calls=[warmup_call],
                    )
                    yield from _yield_pushes(from_pos, to_pos)
                    continue

                if tool_calls:
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=block_code if block_code else None,
                        content_src=asst_src,
                        tool_calls=tool_calls,
                    )
                    for j, cs in enumerate(tool_calls):
                        try:
                            result = item.get_tool_result(j)
                        except (KeyError, IndexError):
                            result = ""
                        if isinstance(result, ToolError):
                            result = result.err_message
                        yield ChatHistoryItem(
                            role=ChatRole.TOOL,
                            content=result,
                            content_src=ContentSource.CONSOLE,
                            tool_calls=cs,
                        )
                elif block_code:
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=block_code,
                        content_src=asst_src,
                    )
                # block_code empty + no tool_calls — nothing to yield.

            # Console output produced by this block (USER + CONSOLE).
            from_pos = item.console_pos
            to_pos = end_positions[idx]
            if to_pos > from_pos and console:
                console_text = "\n".join(console[from_pos:to_pos])
                yield ChatHistoryItem(
                    role=ChatRole.USER,
                    content=console_text,
                    content_src=ContentSource.CONSOLE,
                )

            # Any push_log messages emitted while this block was running.
            yield from _yield_pushes(from_pos, to_pos)

    def get_next_request(self) -> Dict[str, Any]:
        """Build the parameter dict consumed by ``LLM_API.process_request``.

        ``chat_history`` is the lazy generator returned by
        :meth:`get_chat_history` and contains only conversational turns.
        The agent system prompt is passed separately via ``system_prompt``.

        Returns:
            Dict[str, Any]: With keys ``chat_history`` (Iterable[ChatHistoryItem]),
            ``system_prompt`` (str), ``metadata`` (dict), ``available_tools`` (list),
            optionally ``chat_style`` and ``session_id``.
        """
        metadata = (
            dict(self.job_def.agent._metadata)  # pylint: disable=protected-access
            if self.job_def.agent._metadata else {}
        )
        if self.job_def.model is not None:
            metadata["MODEL"] = self.job_def.model
        system_prompt = self.job_def.system_prompt

        # Append language rule when locale specifies a non-EN language
        # and AUTO_LANG_RULE is not explicitly disabled in metadata.
        if (
            self.job_def.locale is not None
            and metadata.get("AUTO_LANG_RULE", "").upper() != "FALSE"
        ):
            lang_rule = get_language_rule(self.job_def.locale.lang_code)
            if lang_rule:
                system_prompt = f"{system_prompt}\n\n{lang_rule}"

        request_params: Dict[str, Any] = {
            "chat_history": self.get_chat_history(),
            "system_prompt": system_prompt,
            "metadata": metadata,
            "available_tools": self.job_def.agent.all_tools,
        }

        chat_style = self.job_def.chat_style
        if chat_style is not None:
            request_params["chat_style"] = chat_style

        if self.session_id is not None:
            request_params["session_id"] = self.session_id

        return request_params

    def append_chat_log(self, request: Dict, llm_resp: LLM_Response):
        """
        Register the LLM response in the Job's chat_log container.

        This method appends a new ChatLogItem to the chat_log, recording the LLM's
        response and the current console position. When the LLM_Response contains
        tool call requests, the response is stored as a CodeBlock; otherwise it is
        stored as a plain string.

        Args:
            request: The original request parameters (compatible with LLM_API.process_request)
            llm_resp: The LLM's response as an LLM_Response object

        The console_pos is set to len(console), marking the position past the end
        of the current console output.
        """
        chat_style = self.job_def.chat_style
        is_md_style = chat_style in (  # pylint: disable=no-member
            ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG)
        is_direct = chat_style == ChatStyle.DIRECT  # pylint: disable=no-member

        if llm_resp.call_requests:
            # When tool calls are present we build a CodeBlock and need to
            # decide what (if anything) goes into its `code` field.
            #   - DIRECT: dialog text (not Python) — stored for chat history,
            #     never executed (exec_all_steps skips code in DIRECT mode).
            #   - MARKDOWN/MD_DIALOG: code MUST be inside ```python fences,
            #     so extract only those; never feed surrounding prose to
            #     ast.parse.
            #   - other styles: pass the response through as-is.
            if is_direct:
                response_code = extract_dialog(llm_resp.text or "")
            elif is_md_style:
                response_code = strip_markup(llm_resp.text, strict=True)
            else:
                response_code = llm_resp.text
            tool_calls = [
                CallSpec(id=cp.id, func_name=cp.name, args=cp.args or [], kwargs=cp.kwargs or {})
                for cp in llm_resp.call_requests
            ]
            stored_resp = CodeBlock(code=response_code or None, tool_calls=tool_calls)
        else:
            # No tool calls — store the response so it appears in chat
            # history. DIRECT keeps only the dialog text; markdown styles
            # keep the cleaned-up form; other styles store as-is.
            if is_direct:
                stored_resp = extract_dialog(llm_resp.text or "")
            else:
                stored_resp = strip_markup(llm_resp.text, strict=is_md_style)

        chat_item = LLM_LogItem(
            console_pos=len(self.py_env.console) if self.py_env.console else 0,
            llm_resp=stored_resp
        )
        self.chat_log.append(chat_item)

        # Log the LLM response in the same format sent back as history
        resp_code = stored_resp.code if isinstance(stored_resp, CodeBlock) else stored_resp
        if resp_code is None:
            resp_code = ""
        log_resp = f"```python\n{resp_code}\n```" if is_md_style else resp_code
        self._log(log_resp)

    @property
    def last_response(self) -> Union[str, CodeBlock, None]:
        """
        Retrieves the last response received from the LLM (i.e. the Python code to be executed,
        or a CodeBlock when the LLM requested tool calls) or None if chat_log is empty.
        """
        if not self.chat_log:
            return None
        last = self.chat_log[-1]
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

        chat_style = self.job_def.chat_style
        if chat_style == ChatStyle.DIRECT:  # pylint: disable=no-member
            return isinstance(response, str) and bool(response.strip())

        if chat_style == ChatStyle.MD_DIALOG:  # pylint: disable=no-member
            if isinstance(response, CodeBlock):
                return False
            if not isinstance(response, str):
                return False
            return any(line.startswith("# ") for line in response.splitlines())

        return True

    def get_response_times(self) -> Iterable[tuple[datetime, Optional[float]]]:
        """Return user-message timestamps paired with time-to-first user reply.

        Only timestamped user messages stored in ``chat_log`` are included:
        the initial dialog-style message (stored as a plain ``str`` and
        measured from ``created_at``) and any subsequent ``UserLogItem``
        entries. Warmup items are ignored. Internal dialog-mode execution
        turns such as tool-call requests or script-only responses are skipped.
        If no later user-facing ``LLM_LogItem`` responds to a message, its
        duration is ``None``.
        """
        response_times: List[tuple[datetime, Optional[float]]] = []
        pending_indices: List[int] = []

        for item in self.chat_log:
            if isinstance(item, str):
                if item and self.created_at is not None:
                    pending_indices.append(len(response_times))
                    response_times.append((self.created_at, None))
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

        for item in self.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue

            if item.llm_resp is None:
                pending_request_time = item.timestamp
                continue

            if pending_request_time is not None:
                response_times.append((
                    pending_request_time,
                    (item.timestamp - pending_request_time).total_seconds(),
                ))
                pending_request_time = None
            elif previous_response_time is not None:
                response_times.append((
                    previous_response_time,
                    (item.timestamp - previous_response_time).total_seconds(),
                ))
            elif self.created_at is not None and self.created_at <= item.timestamp:
                response_times.append((
                    self.created_at,
                    (item.timestamp - self.created_at).total_seconds(),
                ))

            previous_response_time = item.timestamp

        if pending_request_time is not None:
            response_times.append((pending_request_time, None))

        return response_times

    def _get_warmup_block_count(self) -> int:
        """
        Returns the total number of warmup blocks.

        Returns:
            0 if no warmup_code, 1 if single string, len(warmup_code) if sequence
        """
        warmup = self.job_def.warmup_code
        if warmup is None:
            return 0
        if isinstance(warmup, (str, CodeBlock)):
            return 1
        return len(warmup)

    def has_more_warmup_blocks(self) -> bool:
        """
        Check if there are more warmup blocks to execute.

        Returns:
            True if there are more warmup blocks pending, False otherwise
        """
        block_count = self._get_warmup_block_count()
        if block_count == 0:
            return False
        current_block = self.warmup_block_num if self.warmup_block_num is not None else 0
        return current_block < block_count - 1

    def advance_warmup_block(self):
        """
        Advance to the next warmup block.
        Sets warmup_block_num to the next block index.
        """
        if self.warmup_block_num is None:
            self.warmup_block_num = 1
        else:
            self.warmup_block_num += 1

    def _warmup_end_positions(self) -> List[int]:
        """Return end positions for each warmup block, ordered by warmup_block_num.

        The end position is derived from the next element's console_pos in chat_log.
        For the last warmup item (no next element), the current console length is used.
        """
        console_len = len(self.py_env.console) if self.py_env.console else 0
        # Collect (warmup_block_num, end_pos) pairs
        pairs = []
        for i, item in enumerate(self.chat_log):
            if isinstance(item, WarmupLogItem):
                end_pos = (
                    self.chat_log[i + 1].console_pos
                    if i + 1 < len(self.chat_log)
                    else console_len
                )
                pairs.append((item.warmup_block_num, end_pos))
        pairs.sort(key=lambda x: x[0])
        return [end_pos for _, end_pos in pairs]

    def get_next_code_block(self) -> Union[str, CodeBlock, None]:
        """
        Retrieves the Python code block pending execution (or execution continuation).

        Returns:
            - None if status is DONE
            - Current warmup block (str or CodeBlock) if status is READY or WARMING_UP
            - last_response (str or CodeBlock) in all other cases
        """
        if self.status == JobStatus.DONE:
            return None
        if self.status == JobStatus.READY or self.status == JobStatus.WARMING_UP:
            warmup = self.job_def.warmup_code
            if warmup is None:
                return None
            if isinstance(warmup, (str, CodeBlock)):
                return warmup
            # warmup is a sequence of blocks
            block_num = self.warmup_block_num if self.warmup_block_num is not None else 0
            if block_num < len(warmup):
                return warmup[block_num]
            # All blocks completed
            return None
        return self.last_response

    @property
    def num_turns(self) -> int:
        """Returns the number of LLM turns (excludes warmup code blocks)."""
        return sum(1 for item in self.chat_log if isinstance(item, LLM_LogItem))

    @property
    def _warmup_console_positions(self) -> set:
        """Returns console_pos values that correspond to warmup blocks."""
        return {item.console_pos for item in self.chat_log if isinstance(item, WarmupLogItem)}

    @staticmethod
    def _tool_error_count(item) -> int:
        """Return the number of ToolError entries in item.tool_log."""
        if item.tool_log is None:
            return 0
        if isinstance(item.tool_log, ToolError):
            return 1
        if isinstance(item.tool_log, str):
            return 0
        return sum(1 for entry in item.tool_log if isinstance(entry, ToolError))

    def _has_exception(self, item) -> bool:
        """Return True if *item* has a code exception or a tool exception."""
        if self.py_env.exceptions and item.console_pos in self.py_env.exceptions:
            return True
        return self._tool_error_count(item) > 0

    def _exception_count_for_item(self, item) -> int:
        """Return the number of exceptions for a single chat log item."""
        count = 0
        if self.py_env.exceptions and item.console_pos in self.py_env.exceptions:
            count += 1
        count += self._tool_error_count(item)
        return count

    @property
    def exception_count(self) -> int:
        """Returns the total number of exceptions from LLM turns (excludes warmup blocks).

        Counts both code-execution exceptions (py_env.exceptions) and
        tool-call exceptions (ToolError entries in tool_log).
        """
        if not self.chat_log:
            return 0
        return sum(self._exception_count_for_item(item) for item in self.chat_log
                   if isinstance(item, LLM_LogItem))

    @property
    def max_consecutive_exceptions(self) -> int:
        """Returns the maximum number of consecutive exceptions among LLM turns only.

        Considers both code-execution exceptions and tool-call exceptions.
        """
        if not self.chat_log:
            return 0
        max_streak = 0
        streak = 0
        for item in self.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue
            if self._has_exception(item):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @property
    def approx_token_usage(self) -> int:
        """Calculates approximate token usage based on total bytes sent and received."""
        return (self.total_bytes_sent + self.total_bytes_received) // 4

    def push_user_message(self, message: str) -> bool:
        """Append a user message and re-activate the job if DONE.

        Intended for processing push notifications (e.g. a user writing a
        message into a completed job).

        In MD_DIALOG chat style the message is stored directly in
        ``chat_log`` — as a plain ``str`` when ``chat_log`` is empty (first
        message) or as a ``UserLogItem`` for subsequent messages.

        For all other chat styles the message is appended to
        ``py_env.push_log`` keyed by the current console length.

        When the job transitions DONE → STARTED a new ``LLM_LogItem`` with
        ``llm_resp=None`` is appended to ``chat_log`` to indicate that an
        LLM response is now awaited.

        Args:
            message: The message to push.

        Returns:
            True if the job was transitioned DONE → STARTED, False otherwise.
        """
        if self.job_def.chat_style in (ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
            if not self.chat_log:
                self.chat_log.append(message)
            else:
                self.chat_log.append(UserLogItem(message=message))
        else:
            if self.py_env.push_log is None:
                self.py_env.push_log = {}
            key = len(self.py_env.console) if self.py_env.console else 0
            existing = self.py_env.push_log.get(key)
            if existing is None:
                self.py_env.push_log[key] = message
            elif isinstance(existing, list):
                existing.append(message)
            else:
                self.py_env.push_log[key] = [existing, message]

        if self.status == JobStatus.DONE:  # pylint: disable=no-member
            self.num_completions = 1 if self.num_completions is None else self.num_completions + 1
            self.set_status(JobStatus.STARTED)  # pylint: disable=no-member
            self.chat_log.append(LLM_LogItem(
                console_pos=len(self.py_env.console) if self.py_env.console else 0,
                llm_resp=None
            ))
            return True
        return False  # pylint: disable=no-member
