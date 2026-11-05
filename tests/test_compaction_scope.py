"""Compaction cutoffs retain the newest sufficiently small user interaction."""
# pylint: disable=no-member,protected-access

import pytest

from statek.executors.chat_log_item import LLM_LogItem, NotificationLogItem, UserLogItem
from statek.executors.forked_job import ForkedJob
from statek.executors.job import JobSummary
from statek.executors.summarization import get_compaction_scope
from statek.model_card import get_model_card, set_model_card
from statek.task_difficulty import TaskDifficulty


def _user(tokens, text="request"):
    return UserLogItem(text, est_token_usage=tokens)


def _response(tokens):
    return LLM_LogItem(0, llm_resp="response", est_token_usage=tokens)


@pytest.mark.parametrize("prefix,tail,rate,expected", [
    (90, 10, 0.9, 2), (89, 11, 0.9, 4), (91, 9, 0.9, 2),
    (50, 50, 0.5, 2), (50, 50, 0.6, 4), (50, 50, 0, 2), (99, 1, 1, 4),
])
def test_user_boundary_by_token_fraction(job_factory, prefix, tail, rate, expected):
    job = job_factory()
    job.chat_log = [_user(1), _response(prefix - 1), _user(1), _response(tail - 1)]
    assert get_compaction_scope(job, split_rate=rate) == expected


def test_latest_user_boundary_wins(job_factory):
    job = job_factory()
    job.chat_log = [_user(1), _response(89), _user(1), _response(4), _user(1), _response(4)]
    assert get_compaction_scope(job) == 4


@pytest.mark.parametrize("case", ["empty", "no_user", "zero", "empty_user", "large_user"])
def test_entire_log_when_no_qualifying_boundary(job_factory, case):
    job = job_factory()
    job.chat_log = {
        "empty": [], "no_user": [_response(100)],
        "zero": [_user(0), _response(0), _user(0)],
        "empty_user": [_user(1), _response(99), _user(0, "")],
        "large_user": [_user(1), _response(89), _user(100), _response(1)],
    }[case]
    assert get_compaction_scope(job) == len(job.chat_log)


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_split_rate(job_factory, rate):
    with pytest.raises(ValueError, match="split_rate"):
        get_compaction_scope(job_factory(), split_rate=rate)


def test_legacy_strings_use_current_model_high_estimate(job_factory):
    job = job_factory(model="openai//scope-model")
    set_model_card("openai", "scope-model", token_to_bytes=4, token_estimation_margin=0.1)
    job.chat_log = [_response(90), "ż" * 20]
    assert get_compaction_scope(job) == 2  # The high estimate is 11 tokens, not 10.
    job.chat_log[0].est_token_usage = 100
    assert get_compaction_scope(job) == 1


def test_missing_card_for_legacy_string_is_not_created(job_factory):
    job = job_factory(model="openai//missing-scope")
    job.chat_log = [_response(100), "hello"]
    with pytest.raises(ValueError, match="missing-scope"):
        get_compaction_scope(job)
    assert get_model_card("openai", "missing-scope", no_create=True) is None


def test_cached_items_need_no_card_and_legacy_fields_default_to_zero(job_factory):
    job = job_factory(model="openai//missing-scope")
    old = _response(100)
    del old.est_token_usage
    user = _user(100)
    del user.est_token_usage
    job.chat_log = [old, _response(90), user, _response(10)]
    assert get_compaction_scope(job) == 2


def test_raw_indices_and_existing_summaries_are_preserved(job_factory):
    job = job_factory()
    job.chat_log = [_user(1), _response(89), _user(1), _response(9)]
    job.compact(0, 2, JobSummary("request", "actions"))
    job._Job__pending_chat_log = [NotificationLogItem(console_pos=0, message="queued" * 1000)]
    before = list(job.chat_log), job._Job__job_summaries, job.status, job.get_current_difficulty()
    assert get_compaction_scope(job) == 2
    assert get_compaction_scope(ForkedJob(job, 4)) == 2
    assert (list(job.chat_log), job._Job__job_summaries, job.status,
            job.get_current_difficulty()) == before
    assert [item.est_token_usage for item in job.chat_log] == [1, 89, 1, 9]


@pytest.mark.parametrize("field,value", [("token_to_bytes", 0), ("token_estimation_margin", 1)])
def test_invalid_card_for_legacy_string(job_factory, field, value):
    job = job_factory(model="openai//scope-model")
    card = set_model_card("openai", "scope-model", token_to_bytes=4)
    setattr(card, field, value)
    job.chat_log = [_response(100), "hello"]
    with pytest.raises(ValueError, match=field):
        get_compaction_scope(job)


def test_legacy_text_uses_recorded_difficulty_without_resolving_it(job_factory, monkeypatch):
    job = job_factory(model="L:small,M:medium,H:large")
    job._Job__last_difficulty = TaskDifficulty.medium
    job.chat_log = [_response(100), "hello"]

    def unexpected_difficulty_resolution(self):
        raise AssertionError("Scope selection must not trigger difficulty transitions")

    monkeypatch.setattr(type(job), "get_current_difficulty", unexpected_difficulty_resolution)
    assert get_compaction_scope(job) == 1
    assert job._Job__last_difficulty == TaskDifficulty.medium
    assert len(job.chat_log) == 2
