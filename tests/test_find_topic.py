"""Tests for find_topic."""

import pytest

from statek.document import Document, Topic, find_topic


def _doc(ord_no, topic, title, audience=None):
    meta = {"ord_no": ord_no, "topic": topic, "title": title}
    if audience is not None:
        meta["audience"] = audience
    return Document(document_metadata=meta, body=["body"])


TOPICS = [
    Topic(ord_no=0, topic="Preferences", documents=[
        _doc(0, "Preferences", "Overview"),
        _doc(1, "Preferences", "Details", audience=["agent_a"]),
    ]),
    Topic(ord_no=1, topic="Tools", documents=[
        _doc(0, "Tools", "Tool Guide", audience=["agent_a"]),
    ]),
    Topic(ord_no=2, topic="Preferred Actions", documents=[
        _doc(0, "Preferred Actions", "Actions", audience=["agent_b"]),
    ]),
]


def test_find_by_index():
    assert find_topic(0, "agent_a", TOPICS).topic == "Preferences"
    assert find_topic(1, "agent_a", TOPICS).topic == "Tools"


def test_find_by_exact_name_case_insensitive():
    assert find_topic("tools", "agent_a", TOPICS).topic == "Tools"
    assert find_topic("TOOLS", "agent_a", TOPICS).topic == "Tools"


def test_find_by_name_fragment():
    assert find_topic("tool", "agent_a", TOPICS).topic == "Tools"


def test_find_returns_none_when_no_match():
    assert find_topic("nonexistent", "agent_a", TOPICS) is None
    assert find_topic(99, "agent_a", TOPICS) is None


def test_find_raises_on_ambiguous_fragment():
    # "prefer" matches both "Preferences" and "Preferred Actions" for agent_b
    with pytest.raises(ValueError, match="Preferences"):
        find_topic("prefer", "agent_b", TOPICS)


def test_find_excludes_topic_with_no_accessible_docs():
    # agent_b has no accessible docs in "Tools" (audience=["agent_a"])
    assert find_topic("Tools", "agent_b", TOPICS) is None
