"""Tests for compacted LLM-facing job history."""

# pylint: disable=no-member

import pytest

from statek.chat_history import ChatRole, ContentSource
from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import (
    LLM_LogItem, NotificationLogItem, UserLogItem, WarmupLogItem,
)
from statek.executors.job import JobCompactionResult, JobSummary, format_job_summary


def _summary(label="summary"):
    return JobSummary(input_summary=label, output_summary=label)


def test_compacted_history_preserves_initial_warmup_and_uncovered_turns(job_factory):
    """The default view substitutes a summary without leaking covered outputs."""
    job = job_factory(warmup_code="warmup_code()")
    job.job_def.set_chat_style(ChatStyle.MARKDOWN)
    job.chat_log = [
        UserLogItem("initial request"),
        WarmupLogItem(console_pos=0, warmup_block_num=0),
        LLM_LogItem(console_pos=1, llm_resp="hidden_action()"),
        LLM_LogItem(console_pos=2, llm_resp="visible_action()"),
    ]
    job.py_env.console = ["warmup output", "hidden output", "visible output"]
    job.py_env.push_log = {0: "initial push", 2: "hidden push", 3: "visible push"}
    summary = _summary()
    job._Job__job_summaries = JobCompactionResult(2, 1, summary)  # pylint: disable=protected-access

    history = list(job.get_chat_history())

    assert history[0].role == ChatRole.USER
    assert "initial request" in history[0].content
    assert "initial push" in history[0].content
    assert history[1].role == ChatRole.ASSISTANT
    assert history[1].content == "warmup_code()"
    assert history[2].content == "warmup output"
    assert history[3].role == ChatRole.USER
    assert history[3].content_src == ContentSource.SYSTEM
    assert history[3].content == format_job_summary(summary, job.py_env, ChatStyle.MARKDOWN)
    assert history[4].content == "visible_action()"
    assert history[5].content == "visible output"
    assert history[6].content == "visible push"
    assert len(history) == 7

    raw = list(job.get_chat_history(compacted=False))
    assert any(item.content == "hidden_action()" for item in raw)
    assert any(item.content == "hidden output" for item in raw)
    assert any(item.content == "hidden push" for item in raw)


def test_compacted_history_preserves_initial_user_even_when_covered(job_factory):
    """A summary covering index zero still follows the initial user message."""
    job = job_factory()
    job.chat_log = [UserLogItem("hello"), LLM_LogItem(console_pos=0, llm_resp="next()")]
    summary = _summary()
    job._Job__job_summaries = JobCompactionResult(0, 1, summary)  # pylint: disable=protected-access

    history = list(job.get_chat_history())

    assert history[0].content == "hello"
    assert history[1].content_src == ContentSource.SYSTEM
    assert history[2].content == "next()"


def test_compacted_history_places_covered_initial_summary_after_warmup(job_factory):
    """Initial input stays first and a covered summary follows the full warmup."""
    job = job_factory(warmup_code="setup()")
    job.chat_log = [
        UserLogItem("hello"),
        WarmupLogItem(console_pos=0, warmup_block_num=0),
        LLM_LogItem(console_pos=1, llm_resp="next()"),
    ]
    job.py_env.console = ["ready"]
    job._Job__job_summaries = JobCompactionResult(0, 1, _summary())  # pylint: disable=protected-access

    history = list(job.get_chat_history())

    assert [item.content for item in history[:3]] == ["hello", "setup()", "ready"]
    assert history[3].content_src == ContentSource.SYSTEM
    assert history[4].content == "next()"


def test_compacted_history_rejects_summary_covering_warmup(job_factory):
    """Warmup output must remain available in its original form."""
    job = job_factory(warmup_code="setup()")
    job.chat_log = [WarmupLogItem(console_pos=0, warmup_block_num=0)]
    job._Job__job_summaries = JobCompactionResult(0, 1, _summary())  # pylint: disable=protected-access

    with pytest.raises(ValueError):
        list(job.get_chat_history())


def test_compacted_history_renders_multiple_summaries_in_order(job_factory):
    """Separate compacted slices each become a system-authored user block."""
    job = job_factory()
    job.chat_log = [LLM_LogItem(console_pos=0, llm_resp=f"step{idx}()") for idx in range(4)]
    job._Job__job_summaries = [  # pylint: disable=protected-access
        JobCompactionResult(2, 1, _summary("second")),
        JobCompactionResult(0, 1, _summary("first")),
    ]

    history = list(job.get_chat_history())

    assert [item.content_src for item in history] == [
        ContentSource.SYSTEM,
        ContentSource.ASSISTANT,
        ContentSource.SYSTEM,
        ContentSource.ASSISTANT,
    ]
    assert "first" in history[0].content
    assert history[1].content == "step1()"
    assert "second" in history[2].content
    assert history[3].content == "step3()"


def test_compacted_history_renders_summary_starting_at_later_string(job_factory):
    """A summary stays visible when its first raw entry is an ignored string."""
    job = job_factory()
    job.chat_log = [UserLogItem("first"), "later raw message"]
    job._Job__job_summaries = JobCompactionResult(1, 1, _summary())  # pylint: disable=protected-access

    history = list(job.get_chat_history())

    assert len(history) == 2
    assert history[0].content == "first"
    assert history[1].content_src == ContentSource.SYSTEM


def test_compacted_history_keeps_synthetic_call_ids_stable(job_factory):
    """Compaction leaves identifiers for surviving notification calls intact."""
    job = job_factory()
    job.chat_log = [
        LLM_LogItem(console_pos=0, llm_resp="old()"),
        NotificationLogItem(console_pos=0, message="notice"),
    ]
    raw = list(job.get_chat_history(compacted=False))
    raw_call_id = next(item.tool_calls[0].id for item in raw if item.tool_calls)
    job._Job__job_summaries = JobCompactionResult(0, 1, _summary())  # pylint: disable=protected-access

    compacted = list(job.get_chat_history())
    compacted_call_id = next(item.tool_calls[0].id for item in compacted if item.tool_calls)

    assert compacted_call_id == raw_call_id


def test_compacted_history_is_lazy_and_used_for_next_request(job_factory):
    """The current request uses summaries added before history iteration."""
    job = job_factory()
    job.chat_log = [LLM_LogItem(console_pos=0, llm_resp="old()")]
    history_iter = job.get_chat_history()
    summary = _summary()
    job._Job__job_summaries = JobCompactionResult(0, 1, summary)  # pylint: disable=protected-access

    assert list(history_iter)[0].content_src == ContentSource.SYSTEM
    assert list(job.get_next_request()["chat_history"])[0].content_src == ContentSource.SYSTEM


def test_historical_request_keeps_raw_history(job_factory):
    """A request snapshot for an old turn does not apply current summaries."""
    job = job_factory()
    job.chat_log = [
        LLM_LogItem(console_pos=0, llm_resp="old()"),
        LLM_LogItem(console_pos=0, llm_resp="new()"),
    ]
    job._Job__job_summaries = JobCompactionResult(0, 1, _summary())  # pylint: disable=protected-access

    history = list(job.get_request_data(1)["chat_history"])

    assert [item.content for item in history] == ["old()"]
