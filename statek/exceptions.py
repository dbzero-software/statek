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
