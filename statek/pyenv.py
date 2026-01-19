from dataclasses import dataclass, field
from typing import List, Dict
import dbzero as db0

@db0.memo
@dataclass
class PyEnv:
    """
        This is a class representing the state of execution of the LLM-submitted Python code.
        It holds the job-specific global / local variables (agent’s state)
        as well as the instruction ID to continue from (in case of suspended or failed executions).
    """
    # The LLM program's global state (its persistent part)
    global_state: Dict = field(default_factory=dict)
    # The LLM program's local state
    local_state: Dict = field(default_factory=dict)
    # Console outputs of the LLM's program
    console: List[str] = None
    # The next instruction ID for continuation
    next_instr_id: int = None
    # Exit status (or None if exit not called yet)
    exit_status: str = None

    def console_append(self, out: str):
        """
        Append element into console buffer - creates new one if it does not exist.

        @param out: the output to be written to the console
        """
        if self.console is None:
            self.console = []
        self.console.append(out)
