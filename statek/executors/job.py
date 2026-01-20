from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any
import dbzero as db0
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.agent import Agent
from statek.executors.chat_log_item import ChatLogItem
from statek.utils import prompt_append_console

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


@memo
@dataclass
class JobDef:
    """
    The `JobDescr` instances, as the name suggests - hold job descriptions / definitions.
    """
    # An agent assigned to this job
    agent: Agent
    # f-string with job / task description, might include the {goal}
    description: str
    goal: Optional[str]
    # Optional warmup code (executed) before the first prompt
    warmup_code: Optional[str]

    def prompt(self) -> str:
        if self.goal:
            return self.description.format(goal=self.goal)
        return self.description


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
        chat_log: List[ChatLogItem] = None
    ):
        self.job_def = job_def
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
        if not self.chat_log:
            # First prompt: use job_def.prompt and append entire console from position 0
            return prompt_append_console(
                self.py_env.console,
                self.job_def.prompt(),
                from_pos=0
            )
        else:
            # Not first prompt: format console from last chat element's console position
            last_chat_item = self.chat_log[-1]
            return prompt_append_console(
                self.py_env.console,
                from_pos=last_chat_item.console_pos
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
            For a job with chat_log containing 2 items:
            - First yield: "initial_prompt\n> console_output_0\n> console_output_1"
            - Second yield: "llm_response_1"
            - Third yield: "> console_output_2\n> console_output_3"
            - Fourth yield: "llm_response_2"
        """
        if not self.chat_log:
            # No history if chat_log is empty
            return

        # First element: initial prompt + console from position 0 to first chat item's console_pos
        first_chat_item = self.chat_log[0]
        first_user_message = prompt_append_console(
            self.py_env.console,
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

    @property
    def last_response(self) -> str | None:
        """
        Retrieves the last response received from the LLM (i.e. the Python code to be executed)
        or None if the last chat entry remains unanswered.
        """
        if not self.chat_log:
            return None
        return self.chat_log[-1].llm_resp

    def get_next_code_block(self) -> str | None:
        """
        Retrieves the Python code block pending execution (or execution continuation).

        Returns:
            - None if status is DONE
            - job_def.warmup_code if status is READY
            - last_response in all other cases
        """
        if self.status == JobStatus.DONE:
            return None
        if self.status == JobStatus.READY:
            return self.job_def.warmup_code
        return self.last_response
