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

"""Chat style enum — separated from settings to avoid cyclic imports."""

from dbzero import enum


@enum(values=["CONSOLE", "MARKDOWN", "MD_DIALOG", "DIRECT"])
class ChatStyle:  # pylint: disable=too-few-public-methods
    """Defines how console outputs are presented to the LLM.

    CONSOLE  - console results are prefixed with ">".
    MARKDOWN - console output is presented as-is.
    MD_DIALOG - dialog mode with markdown code blocks for code execution.
    DIRECT   - like MD_DIALOG but python code blocks in LLM responses
               are ignored; code execution only via python_cli tool.
    """
