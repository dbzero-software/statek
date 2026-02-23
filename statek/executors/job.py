from dataclasses import dataclass
import re
from typing import List, Optional, Iterable, Dict, Any, Sequence, Union
import dbzero as db0
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.executors.chat_log_item import ChatLogItem
from statek.llm_api import ChatStepData, LLM_Response
from statek.utils import prompt_append_console, CodeBlock, CallSpec, strip_markup, parse_warmup_block, build_warmup_code
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
        warmup_block_num: Optional[int] = None
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
        # Console position recorded after each warmup block completes
        self.warmup_console_positions: List[int] = []
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

    def _debug_log(self, content: str):
        """
        Write content to the debug log file.

        Writes to a separate file named <agent_name>_<uuid>-debug.log
        for detailed LLM request/response logging.

        Args:
            content: The content to log
        """
        if not self.logs_path:
            return

        import os  # pylint: disable=import-outside-toplevel

        agent_name = self.job_def.agent.role if self.job_def.agent else "unknown"
        job_uuid = db0.uuid(db0.materialized(self))
        log_filename = f"{agent_name}_{job_uuid}-debug.log"
        log_filepath = os.path.join(self.logs_path, log_filename)
        os.makedirs(self.logs_path, exist_ok=True)
        with open(log_filepath, 'a', encoding='utf-8') as f:
            f.write(f"{content}\n\n")

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

    def console_append(self, output: str, error_message: str = None):
        """
        Append output to the console and optionally log it.

        Args:
            output: The output string to append
            error_message: optional error message (if execution resulted in an exception)
        """
        self.py_env.console_append(output)
        if error_message is not None:
            chat_log_item_id = len(self.chat_log) - 1 if self.chat_log else 0
            if self.py_env.exceptions is None:
                self.py_env.exceptions = {}
            self.py_env.exceptions[chat_log_item_id] = error_message
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

    def get_next_prompt(self) -> str:
        """
        Generate the next prompt to be included in the LLM chat.

        If this is the first prompt (chat_log is empty), use job_def.prompt
        and append the entire py_env console starting from position 0.
        Otherwise, format the console starting from the last chat element's
        console position to provide the console result for LLM analysis.

        Returns:
            The formatted prompt string ready to be sent to the LLM
        """
        settings = get_statek_settings()
        chat_style = settings.chat_style
        xml_tags = settings.get_xml_box_tags()
        if not self.chat_log:
            warmup = self.job_def.warmup_code
            if warmup:
                # Warmup exists: template and warmup turns go in chat_history as
                # alternating assistant/user pairs.  The prompt (current user message)
                # is the console output OF the last warmup block, spanning from the
                # end of the previous block to the end of the last block.
                warmup_blocks = [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)
                n = len(warmup_blocks)
                last_end = (
                    self.warmup_console_positions[n - 1]
                    if n - 1 < len(self.warmup_console_positions)
                    else len(self.py_env.console) if self.py_env.console else 0
                )
                prev_end = (
                    self.warmup_console_positions[n - 2]
                    if n >= 2 and n - 2 < len(self.warmup_console_positions)
                    else 0
                )
                limit = last_end - prev_end
                return prompt_append_console(
                    self.py_env.console, chat_style,
                    from_pos=prev_end, limit=limit if limit > 0 else 0,
                    xml_tags=xml_tags
                ) if limit > 0 else ""
            else:
                # No warmup: template + all console is the first user prompt
                template = self.job_def.prompt()
                console_part = prompt_append_console(
                    self.py_env.console, chat_style, from_pos=0, xml_tags=xml_tags
                )
                parts = [p for p in [template, console_part] if p]
                return "\n".join(parts)
        else:
            # Not first prompt: format console from last chat element's console position
            last_chat_item = self.chat_log[-1]
            return prompt_append_console(
                self.py_env.console,
                chat_style,
                from_pos=last_chat_item.console_pos,
                xml_tags=xml_tags
            )

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
            if code is None:
                code = ""
            if chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
                return f"```python\n{code}\n```"
            return code

        warmup = self.job_def.warmup_code
        warmup_blocks = ([warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)) if warmup else []

        if not self.chat_log:
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
                        self.warmup_console_positions[i]
                        if i < len(self.warmup_console_positions)
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

        # chat_log is non-empty
        first_chat_item = self.chat_log[0]

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
                elif i < len(self.warmup_console_positions):
                    block_end = self.warmup_console_positions[i]
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

        for i in range(1, len(self.chat_log)):
            prev_chat_item = self.chat_log[i - 1]
            current_chat_item = self.chat_log[i]

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

            if not history_strings:
                # No prior history: current prompt is the only message
                yield ChatStepData(code="", console_output=current_prompt)
                return

            # First string is always a user message with no preceding assistant
            yield ChatStepData(code="", console_output=history_strings[0])

            # Pair up remaining (assistant, user) strings, skipping the last asst
            i = 1
            while i < len(history_strings) - 1:
                yield ChatStepData(code=history_strings[i], console_output=history_strings[i + 1])
                i += 2

            # Last string is always an assistant message; pair with current prompt
            yield ChatStepData(code=history_strings[-1], console_output=current_prompt)

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

        chat_item = ChatLogItem(
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
        return self.chat_log[-1].llm_resp

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

    def record_warmup_console_pos(self):
        """Record the current console length after a warmup block completes."""
        self.warmup_console_positions.append(len(self.py_env.console) if self.py_env.console else 0)

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
        """Returns the number of turns so far (i.e. the number of chat log items)."""
        return len(self.chat_log)

    @property
    def exception_count(self) -> int:
        """Returns the total number of exceptions so far."""
        if not self.py_env.exceptions:
            return 0
        return len(self.py_env.exceptions)

    @property
    def max_consecutive_exceptions(self) -> int:
        """Returns the maximum number of consecutive exceptions in Job history."""
        if not self.py_env.exceptions or not self.chat_log:
            return 0
        exception_ids = set(self.py_env.exceptions.keys())
        max_streak = 0
        streak = 0
        for i in range(len(self.chat_log)):
            if i in exception_ids:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @property
    def approx_token_usage(self) -> int:
        """Calculates approximate token usage based on total bytes sent and received."""
        return (self.total_bytes_sent + self.total_bytes_received) // 4
