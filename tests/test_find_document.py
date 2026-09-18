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


def test_find_rejects_duplicate_ids_exposed_by_combined_audiences():
    """Combined resource audiences cannot silently select one of two identical IDs."""
    topic = Topic(ord_no=0, topic="T", documents=[
        _doc(1, "Agent A", audience=["agent_a"]),
        _doc(1, "Agent B", audience=["agent_b"]),
    ])

    with pytest.raises(ValueError, match="Ambiguous document ID '1'") as error:
        find_document(1, ["agent_a", "agent_b"], topic)

    assert "Agent A (#1)" in str(error.value)
    assert "Agent B (#1)" in str(error.value)


def test_duplicate_ids_allow_unique_exact_title_lookup():
    """An unrelated numeric collision does not block a unique exact title."""
    topic = Topic(ord_no=0, topic="T", documents=[
        _doc(1, "Receiver Guide", audience=["agent_a"]),
        _doc(1, "Donor Guide", audience=["agent_b"]),
    ])

    document = find_document("Donor Guide", ["agent_a", "agent_b"], topic)

    assert document.document_metadata["title"] == "Donor Guide"


def test_duplicate_ids_allow_unique_title_fragment_lookup():
    """An unrelated numeric collision does not block a unique title fragment."""
    topic = Topic(ord_no=0, topic="T", documents=[
        _doc(1, "Receiver Guide", audience=["agent_a"]),
        _doc(1, "Donor Manual", audience=["agent_b"]),
    ])

    document = find_document("Manual", ["agent_a", "agent_b"], topic)

    assert document.document_metadata["title"] == "Donor Manual"


def test_duplicate_exact_titles_are_ambiguous():
    """Exact title priority does not silently select one of multiple exact matches."""
    topic = Topic(ord_no=0, topic="T", documents=[
        _doc(1, "Shared Guide", audience=["agent_a"]),
        _doc(2, "Shared Guide", audience=["agent_b"]),
    ])

    with pytest.raises(ValueError, match="Ambiguous document 'Shared Guide'"):
        find_document("Shared Guide", ["agent_a", "agent_b"], topic)
