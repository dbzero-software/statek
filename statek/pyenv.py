from dataclasses import dataclass, field
from typing import List, Dict, Union
import dbzero as db0

from statek.future import FutureResult

@db0.memo
@dataclass
class PyEnv:
    """
        This is a class representing the state of execution of the LLM-submitted Python code.
        It holds the job-specific global / local variables (agent's state)
        as well as the instruction ID to continue from (in case of suspended or failed executions).
    """
    # The LLM program's global state (its persistent part)
    global_state: Dict = field(default_factory=dict)
    # The LLM program's local state
    local_state: Dict = field(default_factory=dict)
    # Console outputs of the LLM's program
    console: List[str] = None
    # Optional error message by chat_log item ID
    exceptions: Dict[int, str] = None
    # Messages pushed into the console of an active job
    push_log: Dict[int, Union[str, List[str]]] = None
    # The next instruction ID for continuation
    next_instr_id: int = None
    # The warmup block number for continuation (for multi-block warmup_code)
    warmup_block_num: int = None
    # Exit status (or None if exit not called yet)
    exit_status: str = None
    # Result of the last instruction awaiting completion
    future_result: FutureResult = None

    @property
    def perm_ctx(self):
        """Return the persistent context dict if it exists in local_state."""
        if self.local_state is None:
            return None
        return self.local_state.get("_PERM_CTX")

    def console_append(self, out: str):
        """
        Append element into console buffer - creates new one if it does not exist.

        @param out: the output to be written to the console
        """
        if self.console is None:
            self.console = []
        self.console.append(out)

    def update_locals(self, **kwargs):
        """Merge values into local_state, creating it if needed."""
        if self.local_state is None:
            self.local_state = {}
        self.local_state.update(kwargs)
