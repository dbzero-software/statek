from dataclasses import dataclass, field
from typing import List, Optional
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.agent import Agent
from statek.executors.chat_log_item import ChatLogItem

"""
READY: a fresh job instance ready for execution
STARTED: job execution in progress
SUSPENDED: job execution suspended - waiting on external events
DONE: execution has been completed (with either success or failure)
"""
@enum(values=["READY", "STARTED", "SUSPENDED", "DONE"])
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
    # Optional startup code (executed) before the first prompt
    startup_code: Optional[str]


@memo
@dataclass
class Job:
    """
    A single “job” is a stateful class representing the current state of a single unit-of-work, being performed end-to-end by a single agent. By a “job” we might mean either a very simple operation such as answering a basic question (“Hey, what day of week is today”) or a complex task involving retrieving information from external systems, communicating with external actors, waiting for approvals etc. - before the final response is generated.
    """
    job_def: JobDef
    #The LLM model family assigned to this job (e.g. Gemini)
    model_family: str
    # The LLM model assigned to this job (includes version)
    model: str
    job_status: JobStatus = JobStatus.READY
    # Associated LLM API's session ID (where available)
    session_id: str = None
    # LLM program's execution environment
    py_env: PyEnv = field(default_factory=PyEnv)
    # Current chat state
    chat_log: List[ChatLogItem] = field(default_factory=list)

    @property
    def last_response(self) -> str | None:
        """
        Retrieves the last response received from the LLM (i.e. the Python code to be executed)
        or None if the last chat entry remains unanswered.
        """
        if not self.chat_log:
            return None
        return self.chat_log[-1].llm_resp

    @property
    def get_next_code_block(self) -> str | None:
        """
        Retrieves the Python code block pending execution (or execution continuation).

        Returns:
            - None if job_status is DONE
            - job_def.startup_code if job_status is READY
            - last_response in all other cases
        """
        if self.job_status == JobStatus.DONE:
            return None
        if self.job_status == JobStatus.READY:
            return self.job_def.startup_code
        return self.last_response
