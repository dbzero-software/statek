"""Tests for find_document."""

import pytest

from statek.document import Document, Topic, find_document


def _doc(ord_no, title, audience=None):
    meta = {"ord_no": ord_no, "topic": "T", "title": title}
    if audience is not None:
        meta["audience"] = audience
    return Document(document_metadata=meta, body=["body"])


TOPIC = Topic(ord_no=0, topic="Preferences", documents=[
    _doc(0, "Overview"),
    _doc(1, "Details", audience=["agent_a"]),
    _doc(2, "Detailed Settings", audience=["agent_a", "agent_b"]),
])


def test_find_by_index():
    assert find_document(0, "agent_a", TOPIC).document_metadata["title"] == "Overview"
    assert find_document(1, "agent_a", TOPIC).document_metadata["title"] == "Details"


def test_find_by_name_fragment_case_insensitive():
    assert find_document("overview", "agent_a", TOPIC).document_metadata["title"] == "Overview"


def test_find_returns_none_when_no_match():
    assert find_document("nonexistent", "agent_a", TOPIC) is None
    assert find_document(99, "agent_a", TOPIC) is None


def test_find_raises_on_ambiguous_fragment():
    # "detail" matches both "Details" and "Detailed Settings" for agent_a
    with pytest.raises(ValueError, match="Details"):
        find_document("detail", "agent_a", TOPIC)


def test_find_exact_title_takes_priority():
    # "Details" matches exactly, even though "Detailed Settings" also contains it
    doc = find_document("Details", "agent_a", TOPIC)
    assert doc.document_metadata["title"] == "Details"


def test_find_excludes_inaccessible_docs():
    # agent_c can only see "Overview" (no audience restriction)
    assert find_document("Details", "agent_c", TOPIC) is None
