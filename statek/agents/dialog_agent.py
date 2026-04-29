"""DialogAgent — base class for 1-to-1 dialog agents."""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional
import dbzero as db0
from statek.agents.agent import SupervisedAgent
from statek.chat_style import ChatStyle

if TYPE_CHECKING:
    from statek.executors.job import Job


def _validate_send_message(send_message: Callable) -> None:
    """Validate that *send_message* meets the required interface.

    Requirements:
      - Must be callable.
      - Must NOT be a temporal function.
      - Must accept a ``body`` parameter.

    Raises:
        TypeError: If *send_message* is not callable.
        ValueError: If it is temporal or missing the ``body`` parameter.
    """
    if not callable(send_message):
        raise TypeError("send_message must be callable")

    if getattr(send_message, "__is_temporal__", False):
        raise ValueError(
            "send_message must not be a temporal function"
        )

    sig = inspect.signature(send_message)
    if "body" not in sig.parameters:
        raise ValueError(
            "send_message must accept a 'body' parameter"
        )


@db0.memo
@dataclass
class Reminder(ABC):
    """Message injected when a dialog job reaches a reminder condition."""

    text: str

    @abstractmethod
    def fire_ready(self, job: "Job") -> bool:
        """Return whether this reminder should fire for the current job."""


@db0.memo
@dataclass
class RecursiveReminder(Reminder):
    """Reminder that should be re-applied whenever the condition is reached."""

    min_dialog_len: Optional[int] = None

    def fire_ready(self, job: "Job") -> bool:
        """Return whether the dialog has reached the configured length."""
        if self.min_dialog_len is None:
            return True
        for idx, _ in enumerate(job.get_dialog(), start=1):
            if idx >= self.min_dialog_len:
                return True
        return False


@db0.memo
@dataclass
class DialogAgent(SupervisedAgent):
    """Base class for 1-to-1 dialog agents.

    Dialog agents communicate with a single user via the ``send_message``
    callable.  Free text from the LLM is forwarded to the user through
    ``send_message``, and incoming user messages are fed back into the
    job via the push-notification mechanism.

    Args:
        send_message: Function used to deliver agent responses.  Must
            accept ``body`` (str) and optional ``media`` parameters.
            Temporal functions are **not** allowed.
        tools: Additional agent tools.
    """

    send_message: Callable = None
    additional_tools: Iterable[Callable] = None
    add_answer_tool: bool = True
    __reminder: Optional[Reminder] = None

    def __init__(
        self,
        send_message: Callable,
        tools: Iterable[Callable] = None,
        role: str = "dialog_agent",
        add_answer_tool: bool = True,
        _metadata: Optional[Dict[str, str]] = None,
    ):
        _validate_send_message(send_message)

        self.send_message = send_message
        self.additional_tools = tools if tools is not None else []
        self.add_answer_tool = add_answer_tool
        self.__reminder = None

        basic_tools = list(self.additional_tools)

        super().__init__(
            role=role,
            _system_prompt=None,
            _tools=basic_tools,
            _metadata=_metadata,
        )

        # Register dynamic tools by name; their implementations are populated
        # in init_context() so they survive db0 persistence/reload.
        self.append_tool('_send_message')
        if add_answer_tool:
            self.append_tool('answer')

    @property
    def reminder(self) -> Optional[Reminder]:
        """Return the configured looping reminder, if any."""
        return getattr(self, "_DialogAgent__reminder", None)

    def set_reminder(  # pylint: disable=redefined-builtin
        self, text: str, type: str = "RECURSIVE", **kwargs
    ) -> Reminder:
        """Configure a reminder for dialog job looping.

        Args:
            text: Reminder text to feed back to the dialog.
            type: Reminder kind. Supported value: ``RECURSIVE``.
            **kwargs: Reminder implementation-specific properties.

        Returns:
            The stored reminder instance.
        """
        reminder_type = type.upper()
        if reminder_type == "RECURSIVE":
            self.__reminder = RecursiveReminder(text=text, **kwargs)
        else:
            raise ValueError(f"Unsupported reminder type: {type}")
        return self.__reminder

    def init_context(self):
        if self._X__context is None:
            super().init_context()
            self._create_send_message_tool()
            if self.add_answer_tool:
                self._create_answer_tool()

    def _create_send_message_tool(self):
        """Populate ``_send_message`` impl in agent context."""
        original = self.send_message

        def _send_message(body: str, media=None):
            return original(body=body, media=media)

        _send_message.__name__ = "_send_message"
        _send_message.__doc__ = getattr(original, "__doc__", None)
        self._X__context['_send_message'] = _send_message

    def _create_answer_tool(self):
        """Populate ``answer`` impl in agent context.

        Mirrors ``send_message``'s interface (``body``, optional ``media``)
        but is exposed to the LLM (and python_cli) as ``answer``. After
        forwarding the message it terminates the job via ``exit("Success")``.
        """
        original = self.send_message

        def answer(body: str, media=None):
            """Deliver the final message to the user and end the dialog.

            Args:
                body (required): The final message text to send to the user.
                media: Optional media attachment forwarded to send_message.
            """
            from statek.llm_api import CallParams  # pylint: disable=import-outside-toplevel
            from statek.utils import print_tool_log  # pylint: disable=import-outside-toplevel
            print_tool_log(CallParams(
                call_id="",
                name="answer",
                args=[],
                kwargs={"body": body, "media": media},
            ))
            original(body=body, media=media)
            exit("Success")  # pylint: disable=consider-using-sys-exit

        answer.__name__ = "answer"
        self._X__context['answer'] = answer

    def create_job_def(  # pylint: disable=no-member,arguments-renamed,too-many-arguments,too-many-positional-arguments
        self, tools=None, warmup_code=None, shared_vars=None, chat_style=None,
        locale=None, **kwargs
    ):
        """Create a JobDef with a dialog chat style.

        The chat style is resolved in priority order:
        1. Explicit ``chat_style`` argument (caller override)
        2. ``CHAT_STYLE`` key in the agent's prompt metadata
        3. Default: ``MD_DIALOG``

        Args:
            tools: additional tools for this job
            warmup_code: optional initialization code
            shared_vars: optional shared variables (forwarded to the parent)
            chat_style: chat style override; if omitted, falls back to prompt metadata
            locale: optional locale for job execution
            **kwargs: job-specific parameters forwarded to the parent
        """
        if chat_style is None and self._metadata and 'CHAT_STYLE' in self._metadata:
            chat_style = ChatStyle[self._metadata['CHAT_STYLE']]  # pylint: disable=no-member
        job_def = super().create_job_def(
            tools=tools, warmup_code=warmup_code, shared_vars=shared_vars,
            locale=locale, **kwargs
        )
        job_def.set_chat_style(chat_style or ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        return job_def
