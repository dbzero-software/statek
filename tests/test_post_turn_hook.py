"""Post-turn callbacks run once after successful LLM response processing."""
# pylint: disable=no-member
import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import LLM_LogItem, PostProcessedItem
from statek.executors.job import Job, JobStatus
from statek.executors.utils import job_worker, run_job_step
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from tests.test_run_job_step import RuntimeMessagePostProcessor


@pytest.fixture(name="llm_turn")
def fixture_llm_turn():
    """A successful model response with observable budget and delivery checks."""
    api = MagicMock()
    api.process_request = AsyncMock(return_value=LLM_Response(
        LLM_StepData('answer', None), LLM_Stats(10, 20, None)))
    harness = MagicMock()
    with ExitStack() as stack:
        stack.enter_context(patch('statek.executors.utils.LLM_API.get', return_value=api))
        stack.enter_context(patch('statek.executors.utils.get_llm_harness', return_value=harness))
        dialog = stack.enter_context(patch('statek.executors.utils.handle_dialog',
                                           new_callable=AsyncMock))
        yield api, harness, dialog


def test_base_hook_is_noop(job_factory):
    job = job_factory()
    before = job.status, list(job.chat_log), job.py_env.exit_status
    assert job.post_turn_hook() is None
    assert (job.status, list(job.chat_log), job.py_env.exit_status) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('branch', ['continue', 'consumed', 'reminder', 'notification', 'done'])
@pytest.mark.parametrize('finish', [False, True])
async def test_hook_after_each_completed_turn(job_def_factory, llm_turn, branch, finish):
    job = Job(job_def=job_def_factory(post_processing=(RuntimeMessagePostProcessor('Review')
                  if branch == 'consumed' else None)), job_status=JobStatus.STARTED)
    api, harness, dialog = llm_turn
    events = []
    harness.check_after_step.side_effect = lambda _: events.append('checked')
    if branch in ('reminder', 'notification', 'done'):
        job.job_def.set_chat_style(ChatStyle.DIRECT)

    def hook(current):
        assert current is job
        assert events == ['checked']
        assert job.usage.total_bytes_received == 20
        assert isinstance(job.chat_log[-1],
                          PostProcessedItem if branch == 'consumed' else LLM_LogItem)
        if branch == 'done':
            assert job.status == JobStatus.DONE
            dialog.assert_awaited_once()
        if finish:
            job.set_status(JobStatus.DONE)
        events.append('hook')

    with ExitStack() as stack:
        stack.enter_context(patch.object(Job, 'post_turn_hook', hook, create=True))
        if branch == 'reminder':
            stack.enter_context(patch.object(type(job.job_def.agent), 'reminder',
                                              new=object(), create=True))
            stack.enter_context(patch.object(Job, 'handle_reminder', return_value=True))
        if branch == 'notification':
            stack.enter_context(patch.object(Job, '_process_pending_notifications',
                                              side_effect=[False, True]))
        result = await run_job_step(job)
    assert result is (finish or branch == 'done')
    assert events == ['checked', 'hook']
    api.process_request.assert_awaited_once()
    harness.check_after_step.assert_called_once_with(job)


@pytest.mark.asyncio
@pytest.mark.parametrize('branch', ['done', 'waiting', 'warmup', 'exit', 'api_error'])
async def test_no_hook_without_completed_llm_turn(job_factory, llm_turn, branch):
    job = job_factory(warmup_code=['a = 1', 'b = 2'] if branch == 'warmup' else None)
    api, _, _ = llm_turn
    with ExitStack() as stack:
        hook = stack.enter_context(patch.object(Job, 'post_turn_hook', create=True))
        if branch == 'done':
            job.set_status(JobStatus.DONE)
        elif branch == 'waiting':
            stack.enter_context(patch('statek.executors.utils._release_ready_continuation',
                                      return_value=False))
        elif branch == 'exit':
            job.set_status(JobStatus.STARTED)
            job.chat_log.append(LLM_LogItem(0, llm_resp='exit("done")'))
        elif branch == 'api_error':
            api.process_request.side_effect = RuntimeError('provider failed')
        if branch == 'api_error':
            with pytest.raises(RuntimeError, match='provider failed'):
                await run_job_step(job)
        else:
            await run_job_step(job)
        hook.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('worker', [False, True])
async def test_hook_failure_is_critical(job_factory, llm_turn, worker):
    job = job_factory()
    error = ValueError('hook failure')
    with patch.object(Job, 'post_turn_hook', side_effect=error, create=True) as hook, \
         patch('statek.executors.utils.handle_critical_error') as notify:
        if worker:
            await job_worker(asyncio.Semaphore(1), job)
            assert job.status == JobStatus.DONE
            assert job.error is not None
            assert any('hook failure' in line for line in job.py_env.console)
            notify.assert_called_once_with(error)
        else:
            with pytest.raises(ValueError, match='hook failure'):
                await run_job_step(job)
            notify.assert_not_called()
        hook.assert_called_once()
    llm_turn[0].process_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_hook_can_resume_automatically_completed_job(job_factory, llm_turn):
    job = job_factory()
    job.job_def.set_chat_style(ChatStyle.DIRECT)

    def resume(current):
        assert current.status == JobStatus.DONE
        current.set_status(JobStatus.STARTED)

    with patch.object(Job, 'post_turn_hook', resume):
        assert await run_job_step(job) is False
    assert job.status == JobStatus.STARTED
    llm_turn[1].check_after_step.assert_called_once_with(job)
