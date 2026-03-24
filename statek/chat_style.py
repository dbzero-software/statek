"""Chat style enum — separated from settings to avoid cyclic imports."""

from dbzero import enum


@enum(values=["CONSOLE", "MARKDOWN", "MD_DIALOG"])
class ChatStyle:  # pylint: disable=too-few-public-methods
    """Defines how console outputs are presented to the LLM.

    CONSOLE - console results are prefixed with ">".
    MARKDOWN - console output is presented as-is.
    """
