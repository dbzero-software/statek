"""Tests for merging job summaries into committed chat history."""

import pytest

from statek.executors.chat_log_item import LLM_LogItem, NotificationLogItem, UserLogItem
from statek.executors.job import JobCompactionResult, JobSummary


def _summary(label):
    return JobSummary(input_summary=label, output_summary=label)


def test_get_compacted_chat_log_without_summaries_preserves_raw_items(job_factory):
    """The iterator keeps all committed entries, including pending LLM markers."""
    job = job_factory()
    user = UserLogItem("hello")
    response = LLM_LogItem(console_pos=0, llm_resp="print(1)")
    pending_response = LLM_LogItem(console_pos=1, llm_resp=None)
    job.chat_log = ["initial", user, response, pending_response]
    queued = NotificationLogItem(console_pos=1, message="queued")
    job._Job__pending_chat_log = [queued]  # pylint: disable=protected-access

    assert list(job.get_compacted_chat_log()) == ["initial", user, response, pending_response]


def test_get_compacted_chat_log_replaces_one_slice(job_factory):
    """A single result replaces exactly its affected raw entries."""
    job = job_factory()
    raw = [
        UserLogItem("first"),
        LLM_LogItem(console_pos=0, llm_resp="first action"),
        UserLogItem("second"),
        LLM_LogItem(console_pos=1, llm_resp="second action"),
    ]
    job.chat_log = raw
    summary = _summary("middle")
    job._Job__job_summaries = JobCompactionResult(1, 2, summary)  # pylint: disable=protected-access

    assert list(job.get_compacted_chat_log()) == [raw[0], summary, raw[3]]
    assert list(job.chat_log) == raw


def test_get_compacted_chat_log_merges_multiple_out_of_order_results(job_factory):
    """Stored result order does not change chronological output."""
    job = job_factory()
    raw = [UserLogItem(str(index)) for index in range(7)]
    job.chat_log = raw
    first = _summary("first")
    second = _summary("second")
    job._Job__job_summaries = [  # pylint: disable=protected-access
        JobCompactionResult(4, 2, second),
        JobCompactionResult(1, 2, first),
    ]

    assert list(job.get_compacted_chat_log()) == [raw[0], first, raw[3], second, raw[6]]


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 0)],
        [(-1, 1)],
        [(2, 2)],
        [(0, 2), (1, 1)],
    ],
)
def test_get_compacted_chat_log_rejects_invalid_ranges(job_factory, ranges):
    """Invalid or overlapping spans cannot silently corrupt the visible history."""
    job = job_factory()
    job.chat_log = [UserLogItem("first"), UserLogItem("second"), UserLogItem("third")]
    job._Job__job_summaries = [  # pylint: disable=protected-access
        JobCompactionResult(start, length, _summary(str(start)))
        for start, length in ranges
    ]

    with pytest.raises(ValueError):
        list(job.get_compacted_chat_log())
