"""Tests for load_documents and Topic."""

import os

import pytest

from statek.document import Document, load_documents, Topic


DOC_PREFS_0 = """\
# ord_no: 0
# topic: Preferences
# title: Overview
First line.
Second line."""

DOC_PREFS_1 = """\
# ord_no: 1
# topic: Preferences
# audience: agent_a
# title: Details
Details body."""

DOC_TOOLS_0 = """\
# ord_no: 0
# topic: Tools
# title: Tool Guide
Tool body."""


@pytest.fixture(autouse=True)
def clear_cache():
    load_documents.cache_clear()
    yield
    load_documents.cache_clear()


def test_load_documents_groups_by_topic(temp_dir):
    """Documents are grouped into topics; topics get consecutive ord_no."""
    prefs_dir = os.path.join(temp_dir, "prefs")
    os.makedirs(prefs_dir, exist_ok=True)
    _write(prefs_dir, "doc0.txt", DOC_PREFS_0)
    _write(prefs_dir, "doc1.md", DOC_PREFS_1)
    _write(temp_dir, "tools.txt", DOC_TOOLS_0)

    topics = load_documents(temp_dir)

    assert len(topics) == 2
    # Topics get consecutive ord_no starting at 0
    assert topics[0].ord_no == 0
    assert topics[1].ord_no == 1
    topic_names = {t.topic for t in topics}
    assert topic_names == {"Preferences", "Tools"}


def test_load_documents_orders_docs_by_ord_no(temp_dir):
    """Documents within a topic are sorted by their ord_no."""
    os.makedirs(temp_dir, exist_ok=True)
    # Write in reverse filename order to verify sorting by ord_no, not name
    _write(temp_dir, "z.txt", DOC_PREFS_1)
    _write(temp_dir, "a.txt", DOC_PREFS_0)

    topics = load_documents(temp_dir)
    assert len(topics) == 1
    docs = topics[0].documents
    assert docs[0].document_metadata["ord_no"] == 0
    assert docs[1].document_metadata["ord_no"] == 1


def test_load_documents_ignores_non_txt_md_files(temp_dir):
    """Only .txt and .md files are loaded."""
    os.makedirs(temp_dir, exist_ok=True)
    _write(temp_dir, "doc.txt", DOC_PREFS_0)
    _write(temp_dir, "notes.json", '{"key": "value"}')

    topics = load_documents(temp_dir)
    assert len(topics) == 1
    assert len(topics[0].documents) == 1


def test_topic_dataclass_fields():
    """Topic has ord_no, topic name, and documents list."""
    t = Topic(ord_no=0, topic="T", documents=[])
    assert t.ord_no == 0
    assert t.topic == "T"
    assert t.documents == []


def test_topic_count_filters_by_audience():
    """count returns only documents accessible to the given agent."""
    doc_all = Document(
        document_metadata={"ord_no": 0, "topic": "T", "title": "A"},
        body=["x"],
    )
    doc_restricted = Document(
        document_metadata={"ord_no": 1, "topic": "T", "title": "B",
                           "audience": ["agent_a"]},
        body=["x"],
    )
    topic = Topic(ord_no=0, topic="T", documents=[doc_all, doc_restricted])
    assert topic.count("agent_a") == 2
    assert topic.count("other") == 1


def _write(directory, filename, content):
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)
