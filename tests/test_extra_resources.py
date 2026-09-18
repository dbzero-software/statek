"""Tests for incremental difficulty-specific extra resource declarations."""

import pytest

from statek.extra_resources import parse_extra_resources


@pytest.mark.parametrize(
    ("definition", "expected"),
    [
        ("", (None, None, None)),
        (" \n\t ", (None, None, None)),
        ("(H:assistant)", (None, None, ["assistant"])),
        ("H:assistant", (None, None, ["assistant"])),
        ("LM:assistant", (["assistant"], None, None)),
        ("LH:assistant", (["assistant"], None, None)),
        ("MH:assistant", (None, ["assistant"], None)),
        ("HML:assistant", (["assistant"], None, None)),
        ("LL:assistant", (["assistant"], None, None)),
        (" ( lm : Assistant , researcher ) , ( h : helpdesk ) ",
         (["Assistant", "researcher"], None, ["helpdesk"])),
        ("L:a,a,M:a,b,H:b,c,c", (["a"], ["b"], ["c"])),
        ("H:a,b,M:a,L:b", (["b"], ["a"], None)),
        ("H:a,L:b,a", (["b", "a"], None, None)),
        ("(H:a),(L:a)", (["a"], None, None)),
        ("L:a,L:b", (["a", "b"], None, None)),
        ("L:A,a", (["A", "a"], None, None)),
        ("M:team.assistant,help-desk", (None, ["team.assistant", "help-desk"], None)),
    ],
)
def test_parse_extra_resources(definition, expected):
    """Resources occur once, at the lowest declared difficulty."""
    assert parse_extra_resources(definition) == expected


@pytest.mark.parametrize(
    "definition",
    [
        "(LM:schedule_assistant,researcher),(H:helpdesk)",
        "LM:schedule_assistant,researcher,H:helpdesk",
        "(LM:schedule_assistant,researcher),H:helpdesk",
        "LM:schedule_assistant,researcher,(H:helpdesk)",
    ],
)
def test_parse_extra_resources_optional_group_parentheses(definition):
    """Parenthesized, bare and mixed forms have identical semantics."""
    assert parse_extra_resources(definition) == (
        ["schedule_assistant", "researcher"], None, ["helpdesk"]
    )


@pytest.mark.parametrize(
    "definition",
    [
        "Q:a", "LQ:a", ":a", "a", "a,L:b", "L:", "()", "( )",
        "L:a,,b", "L:a,", ",L:a", "(L:a,)", "(L:a", "L:a)",
        "((L:a))", "(L:a)(H:b)", "(L:a),b", "(L:a,H:b)",
        "L:a:b", "L:a,Q:b", "L:a,(Q:b)", "L:a H:b", "L:a b",
        "{L:a}", "[L:a]", "L:'a'", 'L:"a"', "L:a\\b",
        "H:assistant()", "L:__import__('os')", "(L:a)garbage",
    ],
)
def test_parse_extra_resources_rejects_malformed_definitions(definition):
    """Invalid groups and trailing input must not be silently discarded."""
    with pytest.raises(ValueError, match="EXTRA_RESOURCES"):
        parse_extra_resources(definition)


def test_parse_extra_resources_results_own_their_lists():
    """Mutating one parse result cannot affect another result or level."""
    first = parse_extra_resources("L:a,M:b")
    second = parse_extra_resources("L:a,M:b")
    first[0].append("changed")
    assert first[1] == ["b"]
    assert second == (["a"], ["b"], None)
