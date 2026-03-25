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
