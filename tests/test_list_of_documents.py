"""Tests for list_of_documents tool implementation."""

# pylint: disable=redefined-outer-name

import os

import pytest

from tests.conftest import StatekContextJob, run_with_statek_job
from statek.document import load_documents
from statek.agents.list_of_documents import list_of_documents as _impl


DOC_PREFS_0 = """\
# ord_no: 0
# topic: Preferences
# title: Overview
Body."""

DOC_PREFS_1 = """\
# ord_no: 1
# topic: Preferences
# audience: agent_a
# title: Details
Body."""

DOC_TOOLS_0 = """\
# ord_no: 0
# topic: Tools
# title: Tool Guide
Body."""


@pytest.fixture(autouse=True)
def clear_cache():
    load_documents.cache_clear()
    yield
    load_documents.cache_clear()


@pytest.fixture
def docs_dir(temp_dir):
    _write(temp_dir, "prefs0.txt", DOC_PREFS_0)
    _write(temp_dir, "prefs1.txt", DOC_PREFS_1)
    _write(temp_dir, "tools0.txt", DOC_TOOLS_0)
    return temp_dir


def test_list_topics(docs_dir, capsys):
    """No topic arg lists all topics."""
    _PERM_CTX = {}  # noqa: F841
    _impl("agent_a", docs_dir, topic=None)
    out = capsys.readouterr().out
    assert "Preferences" in out
    assert "Tools" in out


def test_list_documents_in_topic(docs_dir, capsys):
    """Specifying topic lists documents within it."""
    job = StatekContextJob()
    run_with_statek_job(job, lambda: _impl("agent_a", docs_dir, topic=0))
    out = capsys.readouterr().out
    assert "Overview" in out
    assert "Details" in out


def test_list_documents_filters_by_audience(docs_dir, capsys):
    """Documents not matching audience are excluded."""
    job = StatekContextJob()
    run_with_statek_job(job, lambda: _impl("agent_b", docs_dir, topic="Preferences"))
    out = capsys.readouterr().out
    assert "Overview" in out
    assert "Details" not in out


def test_list_documents_sets_last_topic_id(docs_dir):
    """Accessing a topic stores its ID in _PERM_CTX."""
    job = StatekContextJob()
    run_with_statek_job(job, lambda: _impl("agent_a", docs_dir, topic=0))
    assert job.py_env.local_state["_PERM_CTX"]["last_topic_id"] == 0


def test_list_topics_respects_limit(docs_dir, capsys):
    """Limit parameter caps the number of listed topics."""
    _PERM_CTX = {}  # noqa: F841
    _impl("agent_a", docs_dir, topic=None, limit=1)
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
    assert len(lines) == 1


def _write(directory, filename, content):
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)
