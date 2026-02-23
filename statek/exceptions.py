from dataclasses import dataclass
from typing import Optional


class LLM_HarnessError(Exception):
    """Raised when an LLM harness limit has been exceeded."""


class InvalidFormat(Exception):
    """Raised when an input cannot be parsed due to an unexpected or unsupported format."""


@dataclass
class FutureError(Exception):
    """
    Raised when an operation cannot be completed because it depends on a future event.
    
    The FutureError class is a custom exception raised by a temporal function - when trying
    to retrieve a response which is not available yet. Originally the FutureError may not have
    the instr_num field set which is decorated later by the framework (see exec_step).
    
    NOTE: FutureError is reserved only for the temporal function's result retrieval - it cannot
    be called from tools or system functions etc.
    
    Attributes:
        future_result: The awaited result
        instr_num: The instruction number (to continue from)
    """
    future_result: 'FutureResult'  # Forward reference
    instr_num: Optional[int] = None
