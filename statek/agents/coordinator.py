"""Coordinator agent implementation for dispatching and managing specialized agents."""

from typing import Dict, Optional
from dataclasses import dataclass
import dbzero as db0
from statek.agents.agent import Agent, SupervisedAgent
from statek.system import create_tool, docs, delegate_task


# Coordinator's system prompt
COORDINATOR_SYSTEM_PROMPT = """### Role
You are an Intelligent Dispatcher & Quality Controller.
Your goal is to analyze user requests, match them to specialized agents, and verify the outcome of their work before finalizing the interaction.
Dont try to solve user requests yourself; always delegate to the most suitable specialized agent.

### Instructions
1. **Analyze Intent:** Determine the core goal of the user's message and the language spoken.
2. **Agent Matching:** Identify the most capable agent for the task.
3. **Decision Logic (Initial):** 
   - **If a match is found:** Formulate a delegation plan.
   - **If no match is found:** Apologize in the user's native language and suggest the closest available capability.
4. **Verification Step (Post-Execution):** Once an agent has completed a task, review the **Exit Status**:
   - **If SUCCESS:** Just exit
   - **If FAILED:** Re-analyze the conversation history. You must then either: 
     a) Delegate to a *different* agent if a better match is identified based on the failure. 
     b) Provide a polite, helpful response to the user explaining the limitation and offering manual assistance.

### Communication Rules
- **User Response:** Always reply to the user in their native language. 
- **Delegation Task:** Define the task/instructions for the agent strictly in English.

### Available Tools
{tools}
"""


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
        # Store task agents
        self.task_agents = task_agents if task_agents is not None else {}

        # Initialize with basic tools that will be expanded in context property
        basic_tools = [docs, delegate_task]

        # Call parent constructor
        super().__init__(
            role="coordinator",
            _system_prompt=COORDINATOR_SYSTEM_PROMPT,
            _tools=basic_tools,
            _X__context=None
        )

    @property
    def context(self) -> Optional[Dict]:
        """
        Get coordinator's private context with dynamically created tools.

        When accessed, verifies if the 'find_agents' tool is available.
        If not, creates it dynamically within the private context using create_tool.

        Returns:
            Dictionary containing private tools and context
        """
        # Initialize context if it doesn't exist
        if self._X__context is None:
            self._X__context = {}

        # Check if find_agents tool is already in context
        if 'find_agents' not in self._X__context:
            # Create the find_agents tool dynamically using create_tool
            docstring = """Find available specialized agents.
            
            Returns:
                String describing available agents and their roles
            """
            create_tool(
                tool_name='find_agents',
                callable=self._find_agents_impl,
                docstring=docstring,
                context=self._X__context
            )

        return self._X__context

    def _find_agents_impl(self) -> str:
        """
        Implementation of find_agents functionality.

        Returns:
            String describing available agents and their roles
        """
        if not self.task_agents:
            return "No specialized agents are currently available."

        result = "Available specialized agents:\n"
        for role, agent in self.task_agents.items():
            # Extract key information from agent's system prompt
            if hasattr(agent, '_system_prompt'):
                prompt_preview = agent._system_prompt[:200]  # pylint: disable=protected-access
            else:
                prompt_preview = "No description available"
            result += f"\n- Role: {role}\n  Description: {prompt_preview}...\n"

        return result
