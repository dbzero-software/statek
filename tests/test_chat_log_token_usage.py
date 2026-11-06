"""Stored conservative estimates follow appends and tool-result updates."""
# pylint: disable=no-member,protected-access,unused-argument,redefined-outer-name

from unittest.mock import AsyncMock, patch

import dbzero as db0
import pytest

from statek.agents.dialog_agent import RecurringReminder
from statek.executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, PostProcessedItem,
    SubTaskLogItem, SystemLogItem, ToolError, UserLogItem, WarmupLogItem,
)
from statek.executors.forked_job import ForkedJob
from statek.executors.job import Job, JobStatus
from statek.executors.post_processor import FinalCheck
from statek.executors.summarization import push_summarization_instr
from statek.executors.utils import run_job_step
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from statek.model_card import get_model_card, set_model_card
from statek.future import FutureResult
from statek.exceptions import FutureError
from statek.chat_style import ChatStyle
from statek.task import SubTaskHandler
from statek.task_difficulty import TaskDifficulty
from statek.utils import CallSpec, CodeBlock
from tests.conftest import DB0_DIR


def _future_ready(future):
    return future.deps["ready"]


def _future_value(future):
    if not _future_ready(future):
        raise FutureError(future)
    return future.deps["value"]


@pytest.fixture
def estimated_job(job_factory):
    job = job_factory(model="openai//estimate-test")
    card = set_model_card("openai", "estimate-test", token_to_bytes=4)
    return job, card


def test_field_preserves_positional_constructors(db0_fixture):
    item = LLM_LogItem(12, "result", llm_resp="response")
    assert item.est_token_usage == 0
    assert item.llm_resp == "response"


def test_append_and_update_use_high_bound(estimated_job):
    job, card = estimated_job
    item = LLM_LogItem(0, llm_resp="x" * 400)
    job._append_chat_log_item(item)
    assert item.est_token_usage == 105
    item.push_tool_result("y" * 400)
    assert item.est_token_usage == 210
    item.set_tool_result(0, ToolError("z" * 800))
    assert item.est_token_usage == card.est_token_count(item)[1] == 315


def test_missing_card_fails_before_append(job_factory):
    job = job_factory(model="openai//missing-estimate")
    with pytest.raises(ValueError, match="missing-estimate"):
        job._append_chat_log_item(SystemLogItem(console_pos=0, message="hello"))
    assert not job.chat_log
    assert get_model_card("openai", "missing-estimate", no_create=True) is None


def test_invalid_card_fails_before_append(estimated_job):
    job, card = estimated_job
    card.token_to_bytes = 0
    with pytest.raises(ValueError, match="token_to_bytes"):
        job._append_chat_log_item(SystemLogItem(console_pos=0, message="hello"))
    assert not job.chat_log


def test_llm_request_model_is_retained_for_refresh(estimated_job):
    job, _ = estimated_job
    card = set_model_card("openrouter", "family/model", "family", token_to_bytes=2)
    response = LLM_Response(LLM_StepData(text="x" * 400, call_requests=[]), LLM_Stats(0, 0, None))
    job.append_chat_log({"model": "openrouter/family/model"}, response)
    item = job.chat_log[-1]
    assert item.est_token_usage == 210
    item.push_tool_result("y" * 400)
    assert item.est_token_usage == card.est_token_count(item)[1] == 420


def test_notification_reminder_postprocessor_and_summary(estimated_job):
    job, card = estimated_job
    job.send_notification("notice" * 20)
    job._process_pending_notifications()
    job.handle_reminder(RecurringReminder(text="remember" * 20))
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.handle_post_processor(FinalCheck(), LLM_StepData(text="done", call_requests=[]))
    push_summarization_instr(job)
    assert any(isinstance(item, PostProcessedItem) for item in job.chat_log)
    for item in job.chat_log:
        assert item.est_token_usage == card.est_token_count(item)[1] > 0


@pytest.mark.asyncio
async def test_regular_and_cli_results_refresh_estimate(estimated_job):
    job, card = estimated_job
    job.set_status(JobStatus.STARTED)
    item = LLM_LogItem(0, llm_resp=CodeBlock(tool_calls=[
        CallSpec("regular", "lookup"),
        CallSpec("cli", "python_cli", kwargs={"code": "print('x' * 400); exit('done')"}),
    ]))
    job._append_chat_log_item(item)
    before = item.est_token_usage
    with patch("statek.executors.utils.exec_tool", new=AsyncMock(return_value=("y" * 400, None))):
        assert await run_job_step(job)
    assert item.tool_log[0] == "y" * 400
    assert "x" * 400 in item.tool_log[1]
    assert item.est_token_usage == card.est_token_count(item)[1] > before


@pytest.mark.asyncio
async def test_missing_card_prevents_tool_side_effects(job_factory):
    job = job_factory(model="openai//missing-estimate", warmup_code="side_effect = 1")
    with pytest.raises(ValueError, match="missing-estimate"):
        await run_job_step(job)
    assert "side_effect" not in dict(job.py_env.iter_locals())
    assert not job.chat_log


def test_fork_preserves_estimate_and_card(estimated_job):
    job, card = estimated_job
    item = LLM_LogItem(0, llm_resp="x" * 400)
    job._append_chat_log_item(item)
    fork = ForkedJob(job, 1)
    inherited = fork.chat_log[0]
    assert db0.uuid(inherited) == db0.uuid(item)
    assert inherited.est_token_usage == item.est_token_usage == 105
    item.push_tool_result("y" * 400)
    assert inherited.est_token_usage == card.est_token_count(inherited)[1] == 210
    local = LLM_LogItem(0, llm_resp="z" * 400)
    fork._append_chat_log_item(local)
    local.push_tool_result("w" * 800)
    assert local.est_token_usage == card.est_token_count(local)[1] == 315
    assert item.est_token_usage == 210
    assert len(job.chat_log) == 1


def test_warmup_estimate_excludes_job_source(estimated_job):
    job, _ = estimated_job
    item = WarmupLogItem(0)
    job._append_chat_log_item(item)
    assert item.est_token_usage == 0
    item.push_tool_result("x" * 400)
    assert item.est_token_usage == 105


def test_legacy_item_tool_results_remain_usable(db0_fixture):
    item = ChatLogItem(0)
    item.push_tool_result("legacy result")
    assert item.get_tool_result(0) == "legacy result"
    assert item.est_token_usage == 0


@pytest.mark.asyncio
async def test_cli_estimate_survives_suspension_and_reopen(estimated_job):
    job, card = estimated_job
    job.set_status(JobStatus.STARTED)
    future = FutureResult(db0.dict({"ready": False, "value": "resumed" * 100}), 0)
    future.set_complement_functions(complement=_future_value, condition=_future_ready)
    job.py_env.update_state(pending=future)
    item = LLM_LogItem(0, llm_resp=CodeBlock(tool_calls=[CallSpec(
        "cli", "python_cli", kwargs={
            "code": "print('partial' * 100)\nprint(pending)\nexit('done')",
        },
    )]))
    job._append_chat_log_item(item)
    initial_estimate = item.est_token_usage
    assert not await run_job_step(job)
    assert item.est_token_usage == card.est_token_count(item)[1] > initial_estimate
    assert "partial" * 100 in item.tool_log[0]
    identifier = db0.uuid(job)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    job = db0.fetch(identifier)
    item = job.chat_log[-1]
    job.py_env.get_local("pending").deps["ready"] = True
    assert await run_job_step(job)
    assert "resumed" * 100 in item.tool_log[0]
    assert item.est_token_usage == item._token_estimation_card.est_token_count(item)[1]


def test_subtask_notification_estimates(estimated_job, job_factory):
    job, card = estimated_job
    handler = SubTaskHandler(job_factory(), id="child")
    handler.complete(result="result" * 100)
    job.push_notification(handler)
    assert job._process_pending_notifications()
    item = job.chat_log[-1]
    assert isinstance(item, SubTaskLogItem)
    assert item.est_token_usage == card.est_token_count(item)[1] > 0


def test_difficulty_transition_needs_no_model_card(job_factory):
    job = job_factory(model="openai//missing-estimate")
    item = DifficultyChangeLogItem(console_pos=0, previous_difficulty=TaskDifficulty.low,
                                   difficulty=TaskDifficulty.medium, cumulative_token_usage=100)
    job._append_chat_log_item(item)
    assert item.est_token_usage == 0


@pytest.mark.parametrize("style", [ChatStyle.DIRECT, ChatStyle.MD_DIALOG])
def test_user_messages_store_estimates_and_preserve_timestamps(estimated_job, style):
    job, card = estimated_job
    job.job_def.set_chat_style(style)
    job.push_user_message("ż" * 200)
    first = job.chat_log[0]
    assert first.timestamp == job.created_at
    assert first.est_token_usage == card.est_token_count(first)[1] == 105
    job.push_user_message("x" * 800)
    assert job.chat_log[-1].est_token_usage == 210
    assert first.wait_finished_at is None
    assert UserLogItem("legacy", job.created_at).est_token_usage == 0


def test_constructor_estimates_normalized_user_message(estimated_job):
    source, card = estimated_job
    source.job_def.set_chat_style(ChatStyle.DIRECT)
    job = Job(job_def=source.job_def, chat_log=["x" * 400])
    item = job.chat_log[0]
    assert isinstance(item, UserLogItem)
    assert item.timestamp == job.created_at
    assert item.est_token_usage == card.est_token_count(item)[1] == 105


def test_user_estimate_uses_model_margin_and_survives_fork_and_reopen(estimated_job):
    job, card = estimated_job
    card.token_estimation_margin = 0.2
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.push_user_message("x" * 400)
    fork = ForkedJob(job, 1)
    assert fork.chat_log[0].est_token_usage == 120
    identifier = db0.uuid(fork)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    assert db0.fetch(identifier).chat_log[0].est_token_usage == 120


def test_missing_card_prevents_user_append(job_factory):
    job = job_factory(model="openai//missing-user-estimate")
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    with pytest.raises(ValueError, match="missing-user-estimate"):
        job.push_user_message("hello")
    assert not job.chat_log
