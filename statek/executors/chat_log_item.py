from datetime import datetime
from dataclasses import dataclass, field
import dbzero as db0


@db0.memo
@dataclass
class ChatLogItem:
    # The console's position before execution of the llm_resp code
    console_pos: int
    # Response received from the LLM
    llm_resp: str
    # Date and time of receiving the response (generating this log item)
    timestamp: datetime = field(default_factory = datetime.now())