# pylint: disable=no-member
"""Tests for prompt configuration parsing."""

import pytest

from statek.prompt_config import (
    PromptStyle,
    PromptSection,
    PromptSectionData,
    SystemPrompt,
    SystemPromptData,
    compare_prompts,
    format_system_prompt,
    parse_system_prompt,
)
from statek.task_difficulty import TaskDifficulty


def test_parse_system_prompt_returns_intro_without_sections():
    """Plain text without delimiters is returned as the prompt intro."""
    assert parse_system_prompt("System prompt introduction") == SystemPromptData(
        intro="System prompt introduction",
        sections=[],
    )


def test_parse_system_prompt_handles_all_section_styles():
    """XML, dash, star, and markdown section delimiters are parsed in order."""
    result = parse_system_prompt(
        "System prompt introduction\n"
        "<identity>You're world class programmer</identity>\n"
        "--- CONFIDENTIALITY ---\n"
        "Keep secrets.\n"
        "*** RULES ***\n"
        "Be direct.\n"
        "## Worth noting\n"
        "Details matter.\n"
    )

    assert result == SystemPromptData(
        intro="System prompt introduction",
        sections=[
            PromptSectionData(
                title="identity",
                contents="You're world class programmer",
            ),
            PromptSectionData(
                title="CONFIDENTIALITY",
                contents="Keep secrets.",
            ),
            PromptSectionData(
                title="RULES",
                contents="Be direct.",
            ),
            PromptSectionData(
                title="Worth noting",
                contents="Details matter.",
            ),
        ],
    )


def test_parse_system_prompt_handles_target_difficulties():
    """Target difficulty metadata is parsed for XML attrs and prefixed titles."""
    result = parse_system_prompt(
        '<identity target_difficulties="LMH">Identity.</identity>\n'
        "--- MH:CONFIDENTIALITY ---\n"
        "Medium and high.\n"
        "*** L:RULES ***\n"
        "Low only.\n"
    )

    assert result.sections[0].target_difficulties == {
        TaskDifficulty.low,
        TaskDifficulty.medium,
        TaskDifficulty.high,
    }
    assert result.sections[1] == PromptSectionData(
        title="CONFIDENTIALITY",
        contents="Medium and high.",
        target_difficulties={TaskDifficulty.medium, TaskDifficulty.high},
    )
    assert result.sections[2] == PromptSectionData(
        title="RULES",
        contents="Low only.",
        target_difficulties={TaskDifficulty.low},
    )


def test_parse_system_prompt_accepts_xml_difficulty_attribute_alias():
    """The documented XML difficulty attribute is accepted."""
    result = parse_system_prompt("<identity difficulty='high'>Identity.</identity>")

    assert result.sections == [
        PromptSectionData(
            title="identity",
            contents="Identity.",
            target_difficulties={TaskDifficulty.high},
        )
    ]


def test_parse_system_prompt_multiline_xml_section():
    """XML section contents can span multiple lines."""
    result = parse_system_prompt(
        "Intro.\n"
        "<identity target_difficulties=\"M\">\n"
        "Line one.\n"
        "Line two.\n"
        "</identity>\n"
    )

    assert result.intro == "Intro."
    assert result.sections == [
        PromptSectionData(
            title="identity",
            contents="Line one.\nLine two.",
            target_difficulties={TaskDifficulty.medium},
        ),
    ]


def test_parse_system_prompt_ignores_non_difficulty_colon_titles():
    """A colon in a title is preserved unless the prefix is a valid difficulty list."""
    result = parse_system_prompt("## API: Contracts\nDetails.")

    assert result.sections == [
        PromptSectionData(title="API: Contracts", contents="Details.")
    ]


def test_parse_system_prompt_invalid_xml_difficulty_raises():
    """Invalid XML difficulty metadata is rejected."""
    with pytest.raises(ValueError, match="Invalid task difficulty"):
        parse_system_prompt("<identity difficulty='urgent'>Identity.</identity>")


def test_format_system_prompt_defaults_to_dashed_sections():
    """The default formatter renders the intro and eligible dashed sections."""
    prompt = SystemPromptData(
        intro="Intro.",
        sections=[
            PromptSectionData(title="identity", contents="Identity."),
            PromptSectionData(
                title="advanced",
                contents="High only.",
                target_difficulties={TaskDifficulty.high},
            ),
            PromptSectionData(
                title="rules",
                contents="Low and medium.",
                target_difficulties={TaskDifficulty.low, TaskDifficulty.medium},
            ),
        ],
    )

    assert format_system_prompt(prompt, TaskDifficulty.medium) == (
        "Intro.\n\n"
        "--- identity ---\n"
        "Identity.\n\n"
        "--- rules ---\n"
        "Low and medium."
    )


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        (PromptStyle.XML, "Intro.\n\n<identity>Identity.</identity>"),
        (PromptStyle.DASHED, "Intro.\n\n--- identity ---\nIdentity."),
        (PromptStyle.ASTERISK, "Intro.\n\n*** identity ***\nIdentity."),
        (PromptStyle.MARKDOWN, "Intro.\n\n## identity\nIdentity."),
    ],
)
def test_format_system_prompt_supports_section_styles(style, expected):
    """Supported styles control section delimiters."""
    prompt = SystemPromptData(
        intro="Intro.",
        sections=[PromptSectionData(title="identity", contents="Identity.")],
    )

    assert format_system_prompt(prompt, TaskDifficulty.low, style=style) == expected


def test_format_system_prompt_uses_section_formatter():
    """A custom section formatter can transform section contents before render."""
    prompt = SystemPromptData(
        intro="Intro.",
        sections=[PromptSectionData(title="identity", contents="{{agent}}")],
    )

    assert format_system_prompt(
        prompt,
        TaskDifficulty.low,
        section_formatter=lambda contents: contents.replace("{{agent}}", "Codex"),
    ) == "Intro.\n\n--- identity ---\nCodex"


def test_format_system_prompt_accepts_persistent_prompt(db0_fixture):  # pylint: disable=unused-argument
    """Persistent SystemPrompt objects are formatted like volatile data prompts."""
    prompt = SystemPrompt(
        intro="Intro.",
        sections=[
            PromptSection(
                title="identity",
                contents="Identity.",
                target_difficulties={TaskDifficulty.high},
            ),
        ],
    )

    assert format_system_prompt(
        prompt,
        TaskDifficulty.high,
        style=PromptStyle.MARKDOWN,
    ) == "Intro.\n\n## identity\nIdentity."


def test_compare_prompts_matches_data_and_memo_prompts(db0_fixture):  # pylint: disable=unused-argument
    """Equivalent volatile and persistent prompts compare as identical."""
    data_prompt = SystemPromptData(
        intro="Intro.",
        sections=[
            PromptSectionData(
                title="identity",
                contents="Identity.",
                target_difficulties={TaskDifficulty.low, TaskDifficulty.high},
            ),
            PromptSectionData(title="rules", contents="Rules."),
        ],
    )
    memo_prompt = SystemPrompt(
        intro="Intro.",
        sections=[
            PromptSection(
                title="identity",
                contents="Identity.",
                target_difficulties={TaskDifficulty.high, TaskDifficulty.low},
            ),
            PromptSection(title="rules", contents="Rules."),
        ],
    )

    assert compare_prompts(data_prompt, memo_prompt)
    assert compare_prompts(memo_prompt, data_prompt)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            SystemPromptData(intro="Intro.", sections=[]),
            SystemPromptData(intro="Different.", sections=[]),
        ),
        (
            SystemPromptData(
                intro="Intro.",
                sections=[PromptSectionData(title="one", contents="Contents.")],
            ),
            SystemPromptData(
                intro="Intro.",
                sections=[PromptSectionData(title="two", contents="Contents.")],
            ),
        ),
        (
            SystemPromptData(
                intro="Intro.",
                sections=[PromptSectionData(title="one", contents="Contents.")],
            ),
            SystemPromptData(
                intro="Intro.",
                sections=[PromptSectionData(title="one", contents="Different.")],
            ),
        ),
        (
            SystemPromptData(
                intro="Intro.",
                sections=[
                    PromptSectionData(
                        title="one",
                        contents="Contents.",
                        target_difficulties={TaskDifficulty.low},
                    )
                ],
            ),
            SystemPromptData(
                intro="Intro.",
                sections=[
                    PromptSectionData(
                        title="one",
                        contents="Contents.",
                        target_difficulties={TaskDifficulty.medium},
                    )
                ],
            ),
        ),
        (
            SystemPromptData(
                intro="Intro.",
                sections=[
                    PromptSectionData(title="one", contents="One."),
                    PromptSectionData(title="two", contents="Two."),
                ],
            ),
            SystemPromptData(
                intro="Intro.",
                sections=[
                    PromptSectionData(title="two", contents="Two."),
                    PromptSectionData(title="one", contents="One."),
                ],
            ),
        ),
    ],
)
def test_compare_prompts_detects_differences(left, right):
    """Any field, section count, or section order difference is significant."""
    assert not compare_prompts(left, right)
