from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
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
    # Optional error message by chat_log item ID
    exceptions: Dict[int, str] = None
    # Optional log of tool results by console log-ID
    tool_log: Dict[int, Union[str, List[str]]] = None
    # Messages pushed into the console of an active job
    push_log: Dict[int, Union[str, List[str]]] = None
    # The next instruction ID for continuation
    next_instr_id: int = None
    # The warmup block number for continuation (for multi-block warmup_code)
    warmup_block_num: int = None
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

    def push_tool_result(self, tool_result: str,
                         console_pos_adjuster: int = None) -> Tuple[int, int]:
        """Append a tool result to the tool_log.

        The result is stored under a key equal to the current console length
        (optionally adjusted by *console_pos_adjuster*). Multiple results
        pushed at the same key are collected into a list.

        Args:
            tool_result: the tool-generated result string
            console_pos_adjuster: optional offset added to the console length
                to compute the storage key (useful when console results are
                cached and not yet flushed)

        Returns:
            A (key, index) tuple — the tool_log key and the position of
            *tool_result* within that key's entry.
        """
        if self.tool_log is None:
            self.tool_log = {}
        key = len(self.console) if self.console else 0
        if console_pos_adjuster is not None:
            key += console_pos_adjuster
        if key not in self.tool_log:
            self.tool_log[key] = tool_result
            return key, 0
        existing = self.tool_log[key]
        if isinstance(existing, str):
            self.tool_log[key] = [existing, tool_result]
            return key, 1
        existing.append(tool_result)
        return key, len(existing) - 1

    def get_tool_result(self, console_pos: int, tool_call_id: int) -> str:
        """Retrieve a tool result from the tool_log.

        Args:
            console_pos: the console position key to look up
            tool_call_id: the index of the tool call within that position's
                entry (see also CodeBlock.get_tool_call_id)

        Returns:
            The tool result string.

        Raises:
            KeyError: if tool_log is None or *console_pos* is not present
            IndexError: if *tool_call_id* is out of range
        """
        if self.tool_log is None:
            raise KeyError(console_pos)
        entry = self.tool_log[console_pos]  # raises KeyError if missing
        if isinstance(entry, str):
            if tool_call_id != 0:
                raise IndexError(
                    f"tool_call_id {tool_call_id} out of range for single result "
                    f"at console_pos {console_pos}")
            return entry
        return entry[tool_call_id]  # raises IndexError if out of range
