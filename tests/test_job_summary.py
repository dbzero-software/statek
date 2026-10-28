"""Tests for formatting compact job summaries."""

# pylint: disable=no-member

import pytest

from statek.chat_style import ChatStyle
from statek.executors.job import (
    JobSummary,
    PyHandleDesc,
    StateSummary,
    format_job_summary,
)
from statek.pyenv import PyEnv


PREAMBLE = (
    "Below is a summary of past communication and actions in this session. "
    "It includes Input Summary, Output Summary, System Summary, and State "
    "(available Python CLI handles)."
)


@pytest.mark.parametrize(
    ("chat_style", "heading_prefix", "xml_box"),
    [
        (ChatStyle.CONSOLE, "", False),
        (ChatStyle.MARKDOWN, "# ", False),
        (ChatStyle.MD_DIALOG, "", True),
        (ChatStyle.DIRECT, "# ", False),
    ],
)
def test_format_job_summary_styles(chat_style, heading_prefix, xml_box, db0_fixture):  # pylint: disable=unused-argument
    """Every chat style presents the same sections using its own markup."""
    summary = JobSummary(
        input_summary="Adam asked about on-calls.",
        output_summary="I retrieved his calendar.",
        system_summary="A notification arrived.",
        state=StateSummary(locals={
            PyHandleDesc("missing", "no longer available"),
            PyHandleDesc("local", "local value"),
            PyHandleDesc("global_name", "global value"),
            PyHandleDesc("shared", "local takes precedence"),
        }),
    )
    env = PyEnv(
        local_state={"local": "hello", "shared": None},
        global_state={"global_name": 42, "shared": 1},
    )
    headings = heading_prefix
    expected = (
        f"{PREAMBLE}\n\n"
        f"{headings}Input Summary\nAdam asked about on-calls.\n\n"
        f"{headings}Output Summary\nI retrieved his calendar.\n\n"
        f"{headings}System Summary\nA notification arrived.\n\n"
        f"{headings}State\nVariables:\n"
        "    global_name (int): global value\n"
        "    local (str): local value\n"
        "    shared (NoneType): local takes precedence"
    )
    if xml_box:
        expected = f"<job_summary>\n{expected}\n</job_summary>"

    assert format_job_summary(summary, env, chat_style) == expected


def test_format_job_summary_empty_optional_sections(db0_fixture):  # pylint: disable=unused-argument
    """Optional sections remain visible without inventing content."""
    summary = JobSummary(input_summary="Input", output_summary="Output")

    assert format_job_summary(summary, PyEnv(), ChatStyle.CONSOLE) == (
        f"{PREAMBLE}\n\n"
        "Input Summary\nInput\n\n"
        "Output Summary\nOutput\n\n"
        "System Summary\n\n"
        "State\nVariables:"
    )


def test_format_job_summary_omits_description_when_absent(db0_fixture):  # pylint: disable=unused-argument
    """A handle without description still includes its resolved type."""
    summary = JobSummary(
        input_summary="Input",
        output_summary="Output",
        state=StateSummary(locals={PyHandleDesc("answer")}),
    )

    result = format_job_summary(summary, PyEnv(local_state={"answer": 42}), ChatStyle.MARKDOWN)

    assert result.endswith("# State\nVariables:\n    answer (int)")
