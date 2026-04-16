"""Researcher agent implementation for looking up information and answering questions."""

from typing import Callable, Iterable
from dataclasses import dataclass
import sys
import dbzero as db0
from statek.agents.agent import SupervisedAgent
from statek.system import docstr



@db0.memo
@dataclass
class Researcher(SupervisedAgent):
    """
    Researcher agent for looking up information and answering questions.

    The researcher is a supervised agent implementation with the responsibility of
    looking-up information, possibly asking additional questions for clarity and
    then responding to the user with an answer.

    Args:
        send_message: User communication function which is used to create `ask` and
                     `answer` tools dynamically. Can be regular, async, or temporal.
                     Can be None if tools already contain 'ask' and 'answer'.
        tools: Additional task-specific tools available to the researcher.
              If tools named 'ask' or 'answer' are included, they take
              precedence over the dynamically generated ones.
    """

    send_message: Callable = None
    additional_tools: Iterable[Callable] = None

    def __init__(self, send_message: Callable = None, tools: Iterable[Callable] = None,
                 role: str = "researcher"):
        """
        Initialize the Researcher agent.

        Args:
            send_message: User communication function (can be regular, async, or temporal).
                         Can be None if tools already contain 'ask' and 'answer'.
            tools: Additional task-specific tools available to the researcher.
                  If tools named 'ask' or 'answer' are included, they take
                  precedence over the dynamically generated ones.
            role: Custom role name (default: "researcher")
        """
        # Store send_message and tools
        self.send_message = send_message
        self.additional_tools = tools if tools is not None else []

        # Initialize with basic tools (docstr + additional tools)
        basic_tools = [docstr] + list(self.additional_tools)

        # Call parent constructor
        super().__init__(
            role=role,
            _system_prompt=None, # Prompt is readed in StatekSetings
            _tools=basic_tools,
        )

        # Register dynamically created tools by name; their implementations are
        # populated in init_context() where send_message closure is available.
        self.append_tool('ask')
        self.append_tool('answer')

    def init_context(self):
        if self._X__context is None:
            super().init_context()
            # Use custom ask/answer from _tools if provided
            custom_ask = self._find_tool('ask')
            custom_answer = self._find_tool('answer')
            has_custom_ask = custom_ask is not None
            has_custom_answer = custom_answer is not None
            if has_custom_ask:
                self._X__context['ask'] = custom_ask
            if has_custom_answer:
                self._X__context['answer'] = custom_answer
            # Validate: send_message is required for any dynamic tool creation
            if self.send_message is None and (not has_custom_ask or not has_custom_answer):
                raise ValueError(
                    "send_message is required when custom 'ask' and/or 'answer' "
                    "tools are not both provided in tools"
                )
            if not has_custom_ask:
                self._create_ask_tool()
            if not has_custom_answer:
                self._create_answer_tool()

    def _create_ask_tool(self):
        """Create the ask tool dynamically."""
        docstring = """Ask the user a clarifying question.

        Use this ONLY if the user's intent is ambiguous or if key details are
        missing to conduct a search. Do not ask for confirmation to proceed;
        only ask for necessary clarification.

        Args:
            question (required): The clarifying question to ask the user

        Returns:
            The user's response
        """

        # Create ask as a wrapper around send_message
        # Note: create_tool is designed for zero-argument functions,
        # so we manually create this tool with a parameter
        def ask_impl(question: str):
            return self.send_message(question)

        ask_impl.__name__ = 'ask'
        ask_impl.__doc__ = docstring

        # Store in context
        self._X__context['ask'] = ask_impl

    def _answer_impl(self, content: str):
        """Implementation of answer functionality.

        Args:
            content: The answer content to send to the user
        """
        # Call send_message (ignore return value)
        self.send_message(content)
        # Exit immediately
        sys.exit("Success")

    def _create_answer_tool(self):
        """Create the answer tool dynamically."""
        docstring = """Deliver the final answer to the user.

        Use this to provide the final response after gathering and synthesizing
        all necessary information.

        Args:
            content (required): The answer content to send to the user

        Returns:
            None (exits immediately after sending)
        """

        # Create answer as a wrapper that takes content parameter
        def answer_impl(content: str):
            return self._answer_impl(content)

        answer_impl.__name__ = 'answer'
        answer_impl.__doc__ = docstring

        # Store in context; registration was done once in __init__ via append_tool('answer')
        self._X__context['answer'] = answer_impl
