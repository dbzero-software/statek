"""Researcher agent implementation for looking up information and answering questions."""

from typing import Callable, Dict, Iterable, Optional
from dataclasses import dataclass
import sys
import dbzero as db0
from statek.agents.agent import SupervisedAgent
from statek.system import docs


# Researcher's system prompt
RESEARCHER_SYSTEM_PROMPT = """You are an expert Research Agent. Your goal is to \
provide accurate, comprehensive, and well-cited answers to user queries.

### Your Toolkit
You have access to specific tools. You must use them to gather information.
- Use `ask(question)`: ONLY if the user's intent is ambiguous or if key details \
are missing to conduct a search. Do not ask for confirmation to proceed; only ask \
for necessary clarification.
- Use `answer(content)`: To deliver the final response to the user.

### Your Process
1. **Analyze:** deeply understand the user's request.
2. **Clarify:** If the request is vague, use the `ask` tool immediately.
3. **Research:** Use your available tools to gather facts.
4. **Synthesize:** Formulate a coherent answer based *only* on the gathered data.

### Constraints
- Do not fabricate information.
- If you cannot find the answer after using tools, state that clearly in the `answer`.

**User Question:**
"{user_question}"

{tools}
"""


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
        tools: Additional task-specific tools available to the researcher.
    """

    send_message: Callable = None
    additional_tools: Iterable[Callable] = None

    def __init__(self, send_message: Callable, tools: Iterable[Callable] = None):
        """
        Initialize the Researcher agent.

        Args:
            send_message: User communication function (can be regular, async, or temporal)
            tools: Additional task-specific tools available to the researcher
        """
        # Store send_message and tools
        self.send_message = send_message
        self.additional_tools = tools if tools is not None else []

        # Initialize with basic tools (docs + additional tools)
        basic_tools = [docs] + list(self.additional_tools)

        # Call parent constructor
        super().__init__(
            role="researcher",
            _system_prompt=RESEARCHER_SYSTEM_PROMPT,
            _tools=basic_tools,
            _X__context=None
        )

    @property
    def context(self) -> Optional[Dict]:
        """
        Get researcher's private context with dynamically created tools.

        When accessed, verifies if the 'ask' and 'answer' tools are available.
        If not, creates them dynamically within the private context using create_tool.

        Returns:
            Dictionary containing private tools and context
        """
        # Initialize context if it doesn't exist
        if self._X__context is None:
            self._X__context = {}

        # Check if ask and answer tools are already in context
        if 'ask' not in self._X__context:
            self._create_ask_tool()

        if 'answer' not in self._X__context:
            self._create_answer_tool()

        return self._X__context

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

        # Store in context
        self._X__context['answer'] = answer_impl
