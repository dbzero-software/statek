# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union
import dbzero as db0

from statek.utils import CodeBlock

if TYPE_CHECKING:
    from statek.agents.dialog_agent import Reminder
    from statek.task import SubTaskHandler


@db0.memo(no_default_tags=True)
@dataclass
class ToolError:
    """Represents a tool-call exception stored positionally in tool_log."""
    err_message: str


@db0.memo(no_default_tags=True)
@dataclass
class ChatLogItem:
    # The console's position before execution of this item's code
    console_pos: int
    # Log of tool results/errors — str for success, ToolError for failure
    tool_log: Optional[Union[str, ToolError, List[Union[str, ToolError]]]] = None
    # Date and time of generating this log item
    timestamp: datetime = field(default_factory=datetime.now)

    def push_tool_result(self, tool_result: Union[str, ToolError]) -> int:
        """Append a tool result or error to tool_log and return its index.

        Args:
            tool_result: the tool-generated result string, or ToolError on failure

        Returns:
            The index of the appended result within tool_log.
        """
        if self.tool_log is None:
            self.tool_log = tool_result
            return 0
        if isinstance(self.tool_log, str) or isinstance(self.tool_log, ToolError):
            self.tool_log = [self.tool_log, tool_result]
            return 1
        self.tool_log.append(tool_result)
        return len(self.tool_log) - 1

    def get_tool_result(self, tool_call_id: int) -> Union[str, ToolError]:
        """Retrieve a tool result by index.

        Args:
            tool_call_id: the index of the tool call within tool_log

        Returns:
            The tool result string or ToolError.

        Raises:
            KeyError: if tool_log is None
            IndexError: if tool_call_id is out of range
        """
        if self.tool_log is None:
            raise KeyError(tool_call_id)
        if isinstance(self.tool_log, (str, ToolError)):
            if tool_call_id != 0:
                raise IndexError(
                    f"tool_call_id {tool_call_id} out of range for single result")
            return self.tool_log
        return self.tool_log[tool_call_id]


@db0.memo(no_default_tags=True)
@dataclass
class LLM_LogItem(ChatLogItem):
    # Response received from the LLM — plain code string or CodeBlock with tool calls
    llm_resp: Union[str, CodeBlock] = None


@db0.memo(no_default_tags=True)
@dataclass
class WarmupLogItem(ChatLogItem):
    # 0-based index into JobDef.warmup_code
    warmup_block_num: int = 0


@db0.memo(no_default_tags=True)
@dataclass(kw_only=True)
class ReminderLogItem(ChatLogItem):
    """The triggering reminder."""
    reminder: "Reminder"


@db0.memo(no_default_tags=True)
@dataclass(kw_only=True)
class SubTaskLogItem(ChatLogItem):
    """The triggering subtask handler."""
    handler: "SubTaskHandler"


@db0.memo(no_default_tags=True)
@dataclass
class UserLogItem:
    """User message submitted for job initiation or as a push notification."""
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
