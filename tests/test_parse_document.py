"""Tests for parse_document and Document.match_audience."""

import pytest

from statek.document import Document, parse_document


def test_parse_document_basic():
    """Required metadata and body are parsed correctly."""
    text = (
        "# ord_no: 0\n"
        "# topic: Preferences\n"
        "# audience: information_retriever\n"
        "# title: Overview\n"
        "First line of body.\n"
        "Second line."
    )
    doc = parse_document(text)
    assert doc.document_metadata["ord_no"] == 0
    assert doc.document_metadata["topic"] == "Preferences"
    assert doc.document_metadata["title"] == "Overview"
    assert doc.document_metadata["audience"] == ["information_retriever"]
    assert doc.body == ["First line of body.", "Second line."]


def test_parse_document_missing_required_header_raises():
    """Missing any of ord_no, topic, or title raises ValueError."""
    text = "# ord_no: 0\n# topic: T\nBody."
    with pytest.raises(ValueError, match="title"):
        parse_document(text)


def test_parse_document_no_audience_defaults_to_none():
    text = "# ord_no: 0\n# topic: T\n# title: T\nBody."
    doc = parse_document(text)
    assert doc.document_metadata.get("audience") is None


def test_match_audience_no_audience_matches_all():
    doc = Document(
        document_metadata={"ord_no": 0, "topic": "T", "title": "T"},
        body=["x"],
    )
    assert doc.match_audience("any_agent") is True


def test_match_audience_checks_list():
    doc = Document(
        document_metadata={
            "ord_no": 0, "topic": "T", "title": "T",
            "audience": ["agent_a", "agent_b"],
        },
        body=["x"],
    )
    assert doc.match_audience("agent_a") is True
    assert doc.match_audience("other") is False
