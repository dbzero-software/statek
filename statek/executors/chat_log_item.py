from datetime import datetime
from dataclasses import dataclass, field
from typing import Union
import dbzero as db0

from statek.utils import CodeBlock


@db0.memo
@dataclass
class ChatLogItem:
    # The console's position before execution of the llm_resp code
    console_pos: int
    # Response received from the LLM — plain code string or CodeBlock with tool calls
    llm_resp: Union[str, CodeBlock]
    # Date and time of receiving the response (generating this log item)
    timestamp: datetime = field(default_factory=datetime.now)