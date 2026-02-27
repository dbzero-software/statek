"""Coordinator agent implementation for dispatching and managing specialized agents."""

from typing import Dict
from dataclasses import dataclass
import dbzero as db0
from statek.agents.agent import Agent, SupervisedAgent
from statek.system import create_tool, docs



@db0.memo
@dataclass
class Coordinator(SupervisedAgent):
    """
    Coordinator agent for analyzing user requests and delegating to specialized agents.
    
    The coordinator is on the front line of user/customer contact. It receives incoming
    messages (new requests), identifies the user, analyzes the message to understand
    the intent, then identifies the Agent most capable of serving the request and
    specifies its Job. If unable to match a capable agent, it informs the client.
    
    Another important task is identifying the client's communication language and when
    specifying the job - translating to the Agent's native language (e.g. PL -> EN).
    """

    task_agents: Dict[str, Agent] = None

    def __init__(self, task_agents: Dict[str, Agent]):
        """
        Initialize the Coordinator agent.

        Args:
            task_agents: Dictionary of available task-specialized agents (by role)
        """
        # Import here to avoid circular dependency
        from statek.task import delegate_task # pylint: disable=import-outside-toplevel

        # Store task agents
        self.task_agents = task_agents if task_agents is not None else {}

        # Initialize with basic tools that will be expanded in context property
        basic_tools = [docs, delegate_task]

        # Call parent constructor
        super().__init__(
            role="coordinator",
            _system_prompt=None, # Prompt is readed in StatekSetings
            _tools=basic_tools,
        )

    def init_context(self):
        if self._X__context is None:
            super().init_context()
            docstring = """Find available specialized agents.

            Returns:
                Dictionary mapping agent role names to Agent objects
            """
            create_tool(
                tool_name='find_agents',
                callable=self._find_agents_impl,
                docstring=docstring,
                context=self._X__context
            )

    def _find_agents_impl(self) -> dict:
        """
        Implementation of find_agents functionality.

        Returns:
            Dictionary mapping agent role names to Agent objects
        """
        return self.task_agents
