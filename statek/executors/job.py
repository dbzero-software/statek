from dataclasses import dataclass
import re
import traceback as _traceback_module
from typing import Callable, List, Optional, Iterable, Dict, Any, Sequence, Union
import dbzero as db0
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.executors.chat_log_item import ChatLogItem, LLM_LogItem, WarmupLogItem
from statek.llm_api import ChatStepData, LLM_Response, CallParams
from statek.utils import (prompt_append_console, CodeBlock, CallSpec, strip_markup,
                          parse_warmup_block, build_warmup_code, _STATEK_TOOL_MARKER)
from statek.future import FutureResult
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

    def __post_init__(self):
        if self.agent is not None:
            db0.tags(self).add(self.agent)

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

    def prompt(self) -> str:
        """
        Generate the user prompt from the agent's prompt template.

        Reads the 'prompt_template' key from agent._metadata and formats it
        with job_params using format_map.

        Returns:
            Formatted prompt string, or None if no template is set
        """
        if self.agent is None:
            return ""
        template = (
            self.agent._metadata.get('prompt_template')  # pylint: disable=protected-access
            if self.agent._metadata else None
        )
        if template is None:
            return None
        if self.job_params:
            return template.format_map(self.job_params)
        return template


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
        model_family: str,
        model: str,
        job_status: JobStatus = JobStatus.READY,
        session_id: str = None,
        py_env: PyEnv = None,
        chat_log: List[ChatLogItem] = None,
        awaited_result: Optional[FutureResult] = None,
        next_instr_num: Optional[int] = None,
        warmup_block_num: Optional[int] = None,
        error: Optional[JobDefError] = None
    ):
        self.job_def = job_def
        if self.job_def.agent is not None:
            db0.tags(self).add(self.job_def.agent)
        # The LLM model family assigned to this job (e.g. Gemini)
        self.model_family = model_family
        # The LLM model assigned to this job (includes version)
        self.model = model
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
        # Registered error handlers (ErrorHandler instances)
        self.error_handlers: List[ErrorHandler] = []
        # Total context bytes used by this job so far
        self.context_bytes = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        # Total cost as reported by the LLM API provider
        self.total_cost = 0.0

        # Log system prompt and prompt template on job creation if logging is enabled
        if self.logs_path and self.job_def.agent is not None:
            self._log(self.job_def.agent.system_prompt(job_params=self.job_def.job_params))
            self._log(self.job_def.prompt())

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
            self.py_env.console, settings.chat_style,
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
        raw_call = f"{call_spec.format()}  {_STATEK_TOOL_MARKER}"
        if settings.chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
            call_line = f"```python\n{raw_call}\n```"
        else:
            call_line = raw_call
        if result:
            result_lines = result.split('\n')
            if settings.chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
                result_text = '\n'.join(result_lines)
            else:
                result_text = '\n'.join(f"> {line}" for line in result_lines)
            xml_tags = settings.get_xml_box_tags()
            tag = xml_tags and xml_tags.get("console")
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

    def _collect_push_log(self, from_pos: int) -> str:
        """Return push_log messages with key >= from_pos as a newline-joined string.

        Args:
            from_pos: Include only entries whose console-position key is >= this value.

        Returns:
            Newline-joined string of all matching messages, or empty string if none.
        """
        if not self.py_env.push_log:
            return ""
        parts = []
        for key in sorted(self.py_env.push_log.keys()):
            if key >= from_pos:
                value = self.py_env.push_log[key]
                if not isinstance(value, str):
                    parts.extend(value)
                else:
                    parts.append(value)
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
        settings = get_statek_settings()
        chat_style = settings.chat_style
        xml_tags = settings.get_xml_box_tags()
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
                # No warmup: template + all console is the first user prompt
                template = self.job_def.prompt()
                console_part = prompt_append_console(
                    self.py_env.console, chat_style, from_pos=0, xml_tags=xml_tags
                )
                push_part = self._collect_push_log(from_pos=0)
                parts = [p for p in [template, console_part, push_part] if p]
                return "\n".join(parts)
        else:
            # Not first prompt: format console from last chat element's console position
            last_chat_item = self.chat_log[-1]
            from_pos = last_chat_item.console_pos
            console_part = prompt_append_console(
                self.py_env.console,
                chat_style,
                from_pos=from_pos,
                xml_tags=xml_tags
            )
            push_part = self._collect_push_log(from_pos=from_pos)
            parts = [p for p in [console_part, push_part] if p]
            return "\n".join(parts)

    def get_chat_history(self) -> Iterable[str]:
        """
        Generate chat history compatible with LLM_API.process_request method.

        This method yields chat history elements in the format expected by the LLM API,
        where messages alternate between user and assistant. The first element includes
        the initial prompt plus console output from position 0. Subsequent elements
        alternate between LLM responses and console fragments from processing those responses.

        Yields:
            str: Chat history elements alternating between user messages (prompt + console)
                 and assistant messages (LLM responses)

        Example:
            For a job with chat_log containing 2 items (CONSOLE style):
            - First yield: "initial_prompt\n> console_output_0\n> console_output_1"
            - Second yield: "llm_response_1"
            - Third yield: "> console_output_2\n> console_output_3"
            - Fourth yield: "llm_response_2"

            In MARKDOWN style, assistant messages (LLM code and warmup code) are wrapped
            in ```python fences; warmup blocks appear as assistant turns before the first
            LLM response.
        """
        settings = get_statek_settings()
        chat_style = settings.chat_style
        xml_tags = settings.get_xml_box_tags()

        def _wrap_code(resp: Union[str, CodeBlock]) -> str:
            code = resp.code if isinstance(resp, CodeBlock) else resp
            if not code:
                return ""
            if chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
                return f"```python\n{code}\n```"
            return code

        warmup = self.job_def.warmup_code
        warmup_blocks = ([warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)) if warmup else []

        # Filter chat_log to only LLM_LogItem entries for history building
        llm_items = [item for item in self.chat_log if isinstance(item, LLM_LogItem)]
        positions = self._warmup_end_positions()

        if not llm_items:
            if not warmup_blocks:
                return  # No history

            # First LLM call with warmup: yield template then warmup assistant/user pairs.
            # The last warmup block (assistant) is the final item; remaining console goes in prompt.
            yield self.job_def.prompt()  # user: template
            prev_pos = 0
            for i, block in enumerate(warmup_blocks):
                block_code = block.code if isinstance(block, CodeBlock) else block
                yield _wrap_code(block_code)  # assistant: warmup code
                is_last = (i == len(warmup_blocks) - 1)
                if not is_last:
                    # Yield console between this block and the next as a user message
                    block_end = (
                        positions[i]
                        if i < len(positions)
                        else len(self.py_env.console) if self.py_env.console else 0
                    )
                    limit = block_end - prev_pos
                    yield prompt_append_console(
                        self.py_env.console, chat_style,
                        from_pos=prev_pos, limit=limit if limit > 0 else 0,
                        xml_tags=xml_tags
                    ) if limit > 0 else ""  # user: console for this block
                    prev_pos = block_end
            return

        # llm_items is non-empty
        first_chat_item = llm_items[0]

        if not warmup_blocks:
            # No warmup: first user message = template + console up to first LLM turn
            template = self.job_def.prompt()
            console_part = prompt_append_console(
                self.py_env.console, chat_style,
                from_pos=0, limit=first_chat_item.console_pos,
                xml_tags=xml_tags
            ) if first_chat_item.console_pos > 0 else ""
            yield f"{template}\n{console_part}" if console_part else template
        else:
            # With warmup: template, then warmup assistant/user pairs.
            # The last warmup user message is extended to cover all console up to the first LLM turn.
            yield self.job_def.prompt()  # user: template
            prev_pos = 0
            for i, block in enumerate(warmup_blocks):
                block_code = block.code if isinstance(block, CodeBlock) else block
                yield _wrap_code(block_code)  # assistant: warmup code
                is_last = (i == len(warmup_blocks) - 1)
                if is_last:
                    block_end = first_chat_item.console_pos  # extend to cover remaining console
                elif i < len(positions):
                    block_end = positions[i]
                else:
                    block_end = len(self.py_env.console) if self.py_env.console else 0
                limit = block_end - prev_pos
                yield prompt_append_console(
                    self.py_env.console, chat_style,
                    from_pos=prev_pos, limit=limit if limit > 0 else 0,
                    xml_tags=xml_tags
                ) if limit > 0 else ""  # user: console for this block
                prev_pos = block_end

        # Yield first LLM response then remaining turns
        yield _wrap_code(first_chat_item.llm_resp)

        for i in range(1, len(llm_items)):
            prev_chat_item = llm_items[i - 1]
            current_chat_item = llm_items[i]

            console_fragment = prompt_append_console(
                self.py_env.console, chat_style,
                from_pos=prev_chat_item.console_pos,
                limit=current_chat_item.console_pos - prev_chat_item.console_pos,
                xml_tags=xml_tags
            )
            yield console_fragment
            yield _wrap_code(current_chat_item.llm_resp)

    def get_next_request(self) -> Dict[str, Any]:
        """
        Generate a complete set of parameters compatible with LLM_API.process_request method.

        This method creates a dictionary containing all necessary parameters for making
        an LLM API request, including the full chat history (with the latest user message
        as its final element), system prompt, and session ID.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - chat_history (Iterable[ChatStepData]): Generator of ChatStepData objects
                  where each step pairs the LLM code response with the resulting console output
                - system_prompt (str): The agent's system prompt
                - metadata (Dict[str, str]): The agent's metadata (may be None)
                - session_id (str, optional): The session ID if available

        Example:
            {
                "chat_history": <generator>,  # ends with the latest user prompt
                "system_prompt": "You are a helpful assistant",
                "metadata": {"MODEL": "gpt-4o"},
                "session_id": "abc123"  # Only if session_id is not None
            }
        """
        def _full_history():
            history_strings = list(self.get_chat_history())
            current_prompt = self.get_next_prompt()

            warmup = self.job_def.warmup_code
            warmup_blocks = (
                [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)
            ) if warmup else []
            num_warmup = len(warmup_blocks)

            def _make_tool_calls(code_block, chat_log_item):
                """Build Dict[CallParams, str] from a CodeBlock and its chat_log_item's tool_log."""
                if not isinstance(code_block, CodeBlock) or not code_block.tool_calls:
                    return None
                if chat_log_item is None:
                    return None
                call_specs = list(code_block.tool_calls)
                tool_calls_dict = {}
                for i, cs in enumerate(call_specs):
                    try:
                        result = chat_log_item.get_tool_result(i)
                    except KeyError:
                        return None
                    except IndexError:
                        result = ""
                    cp = CallParams(
                        call_id=cs.id, name=cs.func_name,
                        args=list(cs.args) if cs.args else [],
                        kwargs=dict(cs.kwargs) if cs.kwargs else {}
                    )
                    tool_calls_dict[cp] = result
                return tool_calls_dict if tool_calls_dict else None

            # Collect WarmupLogItems indexed by warmup_block_num
            warmup_log_items = {
                item.warmup_block_num: item
                for item in self.chat_log if isinstance(item, WarmupLogItem)
            }
            # Collect LLM_LogItems in order
            llm_items = [item for item in self.chat_log if isinstance(item, LLM_LogItem)]

            def _get_step_tool_calls(k):
                """Return tool_calls for the k-th ChatStepData (k=0 = first user-only step)."""
                if k == 0:
                    return None
                if k <= num_warmup:
                    block_idx = k - 1
                    warmup_item = warmup_log_items.get(block_idx)
                    return _make_tool_calls(warmup_blocks[block_idx], warmup_item)
                turn_idx = k - num_warmup - 1
                if turn_idx < len(llm_items):
                    return _make_tool_calls(llm_items[turn_idx].llm_resp, llm_items[turn_idx])
                return None

            if not history_strings:
                # No prior history: current prompt is the only message
                yield ChatStepData(code="", console_output=current_prompt)
                return

            # First string is always a user message with no preceding assistant
            yield ChatStepData(code="", console_output=history_strings[0])

            # Pair up remaining (assistant, user) strings, skipping the last asst
            i = 1
            k = 1
            while i < len(history_strings) - 1:
                yield ChatStepData(
                    code=history_strings[i], console_output=history_strings[i + 1],
                    tool_calls=_get_step_tool_calls(k)
                )
                i += 2
                k += 1

            # Last string is always an assistant message; pair with current prompt
            yield ChatStepData(
                code=history_strings[-1], console_output=current_prompt,
                tool_calls=_get_step_tool_calls(k)
            )

        request_params = {
            "chat_history": _full_history(),
            "system_prompt": self.job_def.agent.system_prompt(job_params=self.job_def.job_params),
            "metadata": self.job_def.agent._metadata,  # pylint: disable=protected-access
            "available_tools": self.job_def.agent.all_tools,
        }

        # Only include session_id if it's not None
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
        chat_style = get_statek_settings().chat_style
        response_code = strip_markup(llm_resp.text, strict=(chat_style == ChatStyle.MARKDOWN))  # pylint: disable=no-member

        if llm_resp.call_requests:
            tool_calls = [
                CallSpec(id=cp.id, func_name=cp.name, args=cp.args or [], kwargs=cp.kwargs or {})
                for cp in llm_resp.call_requests
            ]
            stored_resp = CodeBlock(code=response_code or None, tool_calls=tool_calls)
        else:
            stored_resp = response_code

        chat_item = LLM_LogItem(
            console_pos=len(self.py_env.console) if self.py_env.console else 0,
            llm_resp=stored_resp
        )
        self.chat_log.append(chat_item)

        # Log the LLM response in the same format sent back as history
        resp_code = stored_resp.code if isinstance(stored_resp, CodeBlock) else stored_resp
        if resp_code is None:
            resp_code = ""
        log_resp = f"```python\n{resp_code}\n```" if chat_style == ChatStyle.MARKDOWN else resp_code  # pylint: disable=no-member
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

    @property
    def exception_count(self) -> int:
        """Returns the number of exceptions from LLM turns (excludes warmup blocks)."""
        if not self.py_env.exceptions:
            return 0
        warmup = self._warmup_console_positions
        return sum(1 for pos in self.py_env.exceptions if pos not in warmup)

    @property
    def max_consecutive_exceptions(self) -> int:
        """Returns the maximum number of consecutive exceptions among LLM turns only."""
        if not self.py_env.exceptions or not self.chat_log:
            return 0
        exception_positions = set(self.py_env.exceptions.keys())
        max_streak = 0
        streak = 0
        for item in self.chat_log:
            if isinstance(item, WarmupLogItem):
                continue
            if item.console_pos in exception_positions:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @property
    def approx_token_usage(self) -> int:
        """Calculates approximate token usage based on total bytes sent and received."""
        return (self.total_bytes_sent + self.total_bytes_received) // 4

    def push_to_console(self, message: str) -> bool:
        """Append a pushed message to the job's push_log and re-activate if DONE.

        Intended for processing push notifications (e.g. a user writing a message
        into a completed job). Appends the message to py_env.push_log keyed by
        sequential index. Transitions DONE -> STARTED (updating tags) when the job
        is in DONE state. No other status transitions are performed.

        Args:
            message: The message to append to the push_log.

        Returns:
            True if the job was transitioned DONE -> STARTED, False otherwise.
        """
        if self.py_env.push_log is None:
            self.py_env.push_log = {}
        # Key by current console length so get_next_prompt can filter by position range.
        key = len(self.py_env.console) if self.py_env.console else 0
        existing = self.py_env.push_log.get(key)
        if existing is None:
            self.py_env.push_log[key] = message
        elif isinstance(existing, list):
            existing.append(message)
        else:
            self.py_env.push_log[key] = [existing, message]

        if self.status == JobStatus.DONE:  # pylint: disable=no-member
            self.set_status(JobStatus.STARTED)  # pylint: disable=no-member
            self.append_chat_log({}, LLM_Response(text="")) 
            return True
        return False  # pylint: disable=no-member
