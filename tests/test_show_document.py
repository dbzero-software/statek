"""Tests for show_document tool implementation."""

# pylint: disable=redefined-outer-name

import os

import pytest

from tests.conftest import StatekContextJob, run_with_statek_job
from statek.document import load_documents
from statek.agents.list_of_documents import show_document


DOC_LONG = """\
# ord_no: 0
# topic: Guide
# title: Full Guide
""" + "\n".join(f"Line {i}" for i in range(100))

DOC_RESTRICTED = """\
# ord_no: 1
# topic: Guide
# audience: agent_a
# title: Secret Guide
Restricted content."""


@pytest.fixture(autouse=True)
def clear_cache():
    load_documents.cache_clear()
    yield
    load_documents.cache_clear()


@pytest.fixture
def docs_dir(temp_dir):
    _write(temp_dir, "doc0.txt", DOC_LONG)
    _write(temp_dir, "doc1.txt", DOC_RESTRICTED)
    return temp_dir


def test_show_document_by_id(docs_dir, capsys):
    job = StatekContextJob()
    run_with_statek_job(job, lambda: show_document("agent_a", docs_dir, key=0, topic="Guide"))
    out = capsys.readouterr().out
    assert "Line 0" in out


def test_show_document_by_title_fragment(docs_dir, capsys):
    job = StatekContextJob()
    run_with_statek_job(
        job,
        lambda: show_document("agent_a", docs_dir, key="Full", topic="Guide"),
    )
    out = capsys.readouterr().out
    assert "Line 0" in out


def test_show_document_uses_last_topic_id(docs_dir, capsys):
    """Falls back to last_topic_id when topic is not specified."""
    job = StatekContextJob()
    job.py_env.local_state["_PERM_CTX"] = {"last_topic_id": 0}
    run_with_statek_job(job, lambda: show_document("agent_a", docs_dir, key=0))
    out = capsys.readouterr().out
    assert "Line 0" in out


def test_show_document_limit_and_start(docs_dir, capsys):
    job = StatekContextJob()
    run_with_statek_job(
        job,
        lambda: show_document("agent_a", docs_dir, key=0, topic="Guide", start_from=5, limit=3),
    )
    out = capsys.readouterr().out
    assert "Line 5" in out
    assert "Line 7" in out
    assert "Line 8" not in out


def test_show_document_no_topic_no_last_raises(docs_dir):
    job = StatekContextJob()
    with pytest.raises(ValueError, match="topic"):
        run_with_statek_job(job, lambda: show_document("agent_a", docs_dir, key=0))


def test_show_document_fuzzy_match(docs_dir, capsys):
    """Fuzzy match finds a document when key is close to the title (>=90%)."""
    job = StatekContextJob()
    # "Ful Guide" is close enough to "Full Guide"
    run_with_statek_job(
        job,
        lambda: show_document("agent_a", docs_dir, key="Ful Guide", topic="Guide"),
    )
    out = capsys.readouterr().out
    assert "Line 0" in out


def test_show_document_fuzzy_match_below_threshold(docs_dir, capsys):
    """Fuzzy match rejects keys below 90% similarity."""
    job = StatekContextJob()
    run_with_statek_job(
        job,
        lambda: show_document("agent_a", docs_dir, key="Xyz Abc", topic="Guide"),
    )
    out = capsys.readouterr().out
    assert "not found" in out


def _write(directory, filename, content):
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)
