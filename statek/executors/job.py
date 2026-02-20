from dataclasses import dataclass
import re
from typing import List, Optional, Iterable, Dict, Any, Sequence, Union
import dbzero as db0
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.executors.chat_log_item import ChatLogItem
from statek.utils import prompt_append_console
from statek.future import FutureResult
from statek.settings import get_statek_settings

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


def parse_warmup_code(warmup_code: Optional[Union[str, Sequence[str]]]) -> Optional[Union[str, List[str]]]:
    """Parse warmup_code, splitting on separator lines if present.

    If warmup_code is a string containing comment lines with 10+ dashes
    (e.g. # ----------), it will be split into multiple blocks.

    Args:
        warmup_code: Single string, sequence of strings, or None

    Returns:
        None if input is None
        Single string if no separators found
        List of strings if separators found or input was already a sequence
    """
    if warmup_code is None:
        return None

    if not isinstance(warmup_code, str):
        # Already a sequence, return as list
        return list(warmup_code)

    # Split on comment lines containing 10 or more dashes (e.g. # ----------)
    blocks = re.split(r'\n\s*#\s*-{10,}\s*\n', warmup_code)

    # Strip each block and filter empty ones
    blocks = [block.strip() for block in blocks if block.strip()]

    if len(blocks) == 0:
        return None
    elif len(blocks) == 1:
        return blocks[0]
    else:
        return blocks


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
    warmup_code: Optional[Union[str, Sequence[str]]] = None

    def prompt(self) -> str:
        """
        Generate the prompt by calling agent's prompt method with job_params.
        
        Returns:
            Formatted prompt string
        """
        if self.agent is None:
            return ""
        
        return self.agent.prompt(job_params=self.job_params)


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
        # Total context bytes used by this job so far
        self.context_bytes = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        # Total cost as reported by the LLM API provider
        self.total_cost = 0.0

        # Log system prompt and prompt template on job creation if logging is enabled
        if self.logs_path and self.job_def.agent is not None:
            self._log(self.job_def.agent.system_prompt)
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
        self._log(f"> {output.rstrip()}")

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
        chat_style = get_statek_settings().chat_style
        if not self.chat_log:
            # First prompt: use job_def.prompt and append entire console from position 0
            prompt = prompt_append_console(
                self.py_env.console,
                chat_style,
                self.job_def.prompt(),
                from_pos=0
            )
            return prompt
        else:
            # Not first prompt: format console from last chat element's console position
            last_chat_item = self.chat_log[-1]
            return prompt_append_console(
                self.py_env.console,
                chat_style,
                from_pos=last_chat_item.console_pos
            )
                # Log console output if logging is enabled

        return prompt

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
            For a job with chat_log containing 2 items:
            - First yield: "initial_prompt\n> console_output_0\n> console_output_1"
            - Second yield: "llm_response_1"
            - Third yield: "> console_output_2\n> console_output_3"
            - Fourth yield: "llm_response_2"
        """
        if not self.chat_log:
            # No history if chat_log is empty
            return

        chat_style = get_statek_settings().chat_style

        # First element: initial prompt + console from position 0 to first chat item's console_pos
        first_chat_item = self.chat_log[0]
        first_user_message = prompt_append_console(
            self.py_env.console,
            chat_style,
            self.job_def.prompt(),
            from_pos=0,
            limit=first_chat_item.console_pos
        )
        yield first_user_message

        # Yield first LLM response
        yield first_chat_item.llm_resp

        # Process remaining chat log items
        for i in range(1, len(self.chat_log)):
            prev_chat_item = self.chat_log[i - 1]
            current_chat_item = self.chat_log[i]

            # User message: console fragment from prev_chat_item.console_pos to current_chat_item.console_pos
            console_fragment = prompt_append_console(
                self.py_env.console,
                chat_style,
                from_pos=prev_chat_item.console_pos,
                limit=current_chat_item.console_pos - prev_chat_item.console_pos
            )
            yield console_fragment

            # Assistant message: current LLM response
            yield current_chat_item.llm_resp

    def get_next_request(self) -> Dict[str, Any]:
        """
        Generate a complete set of parameters compatible with LLM_API.process_request method.

        This method creates a dictionary containing all necessary parameters for making
        an LLM API request, including the prompt, chat history, system prompt, and session ID.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - prompt (str): The next prompt to send (from get_next_prompt)
                - chat_history (Iterable[str]): Generator of alternating user/assistant messages (from get_chat_history)
                - system_prompt (str): The agent's system prompt
                - session_id (str, optional): The session ID if available

        Example:
            {
                "prompt": "Process the data",
                "chat_history": <generator>,
                "system_prompt": "You are a helpful assistant",
                "session_id": "abc123"  # Only if session_id is not None
            }
        """
        request_params = {
            "prompt": self.get_next_prompt(),
            "chat_history": self.get_chat_history(),
            "system_prompt": self.job_def.agent.system_prompt
        }

        # Only include session_id if it's not None
        if self.session_id is not None:
            request_params["session_id"] = self.session_id

        return request_params

    def append_chat_log(self, request: Dict, llm_resp: str):
        """
        Register the LLM response in the Job's chat_log container.

        This method appends a new ChatLogItem to the chat_log, recording the LLM's
        response and the current console position.

        Args:
            request: The original request parameters (compatible with LLM_API.process_request)
            llm_resp: The LLM's response (Python code to be executed)

        The console_pos is set to len(console), marking the position past the end
        of the current console output.
        """
        chat_item = ChatLogItem(
            console_pos=len(self.py_env.console) if self.py_env.console else 0,
            llm_resp=llm_resp
        )
        self.chat_log.append(chat_item)
        
        # Log the LLM response
        self._log(llm_resp)

    @property
    def last_response(self) -> str | None:
        """
        Retrieves the last response received from the LLM (i.e. the Python code to be executed)
        or None if the last chat entry remains unanswered.
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
        if isinstance(warmup, str):
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

    def get_next_code_block(self) -> str | None:
        """
        Retrieves the Python code block pending execution (or execution continuation).

        Returns:
            - None if status is DONE
            - Current warmup block if status is READY or WARMING_UP (based on warmup_block_num)
            - last_response in all other cases
        """
        if self.status == JobStatus.DONE:
            return None
        if self.status == JobStatus.READY or self.status == JobStatus.WARMING_UP:
            warmup = self.job_def.warmup_code
            if warmup is None:
                return None
            if isinstance(warmup, str):
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
