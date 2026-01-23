from dataclasses import dataclass
from typing import List, Callable, Dict, Optional
import dbzero as db0
from statek.utils import format_callable_decl

@db0.memo
@dataclass
class Agent:
    """
        This is the fundamental class to hold the workflow specification and available tools.
    """
    role: str  # An arbitrary role name
    _system_prompt: str  # f-string with the {tools} placeholder
    _tools: List[Callable]
    _X__context: Optional[Dict] = None  # Agent's specific context (e.g. with private tools)

    @property
    def system_prompt(self) -> str:
        """
        Format system_prompt with tool descriptions.

        Places all available tool descriptions in the placeholder using
        newlines and > character to start each line.
        """
        tools_str = "\n".join(">" + format_callable_decl(tool) for tool in self._tools)
        return self._system_prompt.format(tools=tools_str)

    @property
    def context(self) -> Optional[Dict]:
        """
        Get agent's private context.
        """
        return self._X__context
