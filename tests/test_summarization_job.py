"""Summarization forks apply validated results and allow model corrections."""
# pylint: disable=no-member,protected-access
import json
from unittest.mock import AsyncMock, MagicMock, patch

import dbzero as db0
import pytest

from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import LLM_LogItem, SystemLogItem, UserLogItem
from statek.executors.job import JobStatus, JobSummary
from statek.executors.summarization import push_summarization_instr
from statek.executors.summarization_job import SummarizationJob
from statek.executors.utils import run_job_step
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from tests.conftest import DB0_DIR


def source_job(job_factory):
    job = job_factory()
    job.chat_log = [UserLogItem('request'), LLM_LogItem(0, llm_resp='old action'),
                    LLM_LogItem(0, llm_resp='recent action')]
    return job


def result_text():
    return json.dumps({'input_summary': 'Request', 'output_summary': 'Actions',
                       'system_summary': None, 'state': {'locals': []}})


def append_result(job, text):
    job.chat_log.append(LLM_LogItem(job.py_env.console_len, llm_resp=text))


@pytest.mark.parametrize('separate_source', [False, True])
def test_success_preserves_previous_summary_and_target_state(job_factory, separate_source):
    target = source_job(job_factory)
    target.compact(0, 1, JobSummary('prior input', 'prior actions'))
    target.set_status(JobStatus.SUSPENDED)
    source = source_job(job_factory) if separate_source else target
    job = SummarizationJob(source, 2, target_job=target, chat_log_pos=1, chat_log_len=1)
    assert job.status == JobStatus.STARTED
    assert job.forked_at == 2
    assert job in list(db0.find(SummarizationJob, db0.as_tag(target)))
    assert job in list(db0.find(SummarizationJob, db0.as_tag(source)))
    assert list(job.chat_log) == list(source.chat_log[:2])
    assert db0.uuid(job.chat_log[1]) == db0.uuid(source.chat_log[1])
    append_result(job, result_text())
    assert job.post_turn_hook() is None
    assert job.status == JobStatus.DONE
    assert target.status == JobStatus.STARTED
    results = list(target._Job__job_summaries)
    assert [(r.chat_log_pos, r.chat_log_len) for r in results] == [(0, 1), (1, 1)]
    assert results[0].job_summary.input_summary == 'prior input'
    assert results[1].job_summary.output_summary == 'Actions'
    assert len(target.chat_log) == 3
    if separate_source:
        assert source._Job__job_summaries is None


@pytest.mark.parametrize('status', ['READY', 'STARTED', 'DONE'])
def test_success_only_resumes_suspended_targets(job_factory, status):
    target = source_job(job_factory)
    target.set_status(getattr(JobStatus, status))
    job = SummarizationJob(target, 2, target_job=target, chat_log_pos=1, chat_log_len=1)
    append_result(job, result_text())
    job.post_turn_hook()
    assert target.status == getattr(JobStatus, status)
    assert job.status == JobStatus.DONE


def test_invalid_result_can_be_corrected(job_factory):
    target = source_job(job_factory)
    target.set_status(JobStatus.SUSPENDED)
    job = SummarizationJob(target, 2, target_job=target, chat_log_pos=1, chat_log_len=1)
    append_result(job, 'invalid')
    job.set_status(JobStatus.DONE)
    job.py_env.exit_status = 'previous completion'
    with patch('statek.executors.summarization_job.parse_summarization_result',
               side_effect=ValueError('state.locals is missing')):
        job.post_turn_hook()
    assert job.status == JobStatus.STARTED
    assert job.py_env.exit_status is None
    assert target.status == JobStatus.SUSPENDED
    assert target._Job__job_summaries is None
    assert isinstance(job.chat_log[-1], SystemLogItem)
    assert 'ValueError: state.locals is missing' in job.chat_log[-1].message
    assert job.chat_log[-1].console_pos == job.py_env.console_len
    assert list(job.get_next_request()['chat_history'])[-1].content == job.chat_log[-1].message
    assert job.get_next_code_block() is None
    append_result(job, result_text())
    job.post_turn_hook()
    assert target.status == JobStatus.STARTED
    assert job.status == JobStatus.DONE


@pytest.mark.parametrize('failure', ['parser', 'compact'])
def test_unrecoverable_errors_leave_target_suspended(job_factory, failure):
    target = source_job(job_factory)
    target.set_status(JobStatus.SUSPENDED)
    job = SummarizationJob(target, 2, target_job=target, chat_log_pos=1, chat_log_len=1)
    append_result(job, result_text())
    count = len(job.chat_log)
    if failure == 'compact':
        target.compact(1, 1, JobSummary('already compacted', ''))
        with pytest.raises(ValueError, match='overlap'):
            job.post_turn_hook()
    else:
        with patch('statek.executors.summarization_job.parse_summarization_result',
                   side_effect=RuntimeError('storage unavailable')):
            with pytest.raises(RuntimeError, match='storage unavailable'):
                job.post_turn_hook()
    assert target.status == JobStatus.SUSPENDED
    assert len(job.chat_log) == count


@pytest.mark.parametrize('pos, length, error', [
    (-1, 1, ValueError), (0, 0, ValueError), (2, 2, ValueError),
    (True, 1, TypeError), (0, False, TypeError), (0.5, 1, TypeError),
])
def test_invalid_target_range_before_fork_initialization(job_factory, pos, length, error):
    target = source_job(job_factory)
    with patch('statek.executors.summarization_job.ForkedJob.__init__') as initialize:
        with pytest.raises(error):
            SummarizationJob(target, 2, target_job=target,
                             chat_log_pos=pos, chat_log_len=length)
        initialize.assert_not_called()


def test_invalid_target_type(job_factory):
    with pytest.raises(TypeError, match='target_job'):
        SummarizationJob(job_factory(), 0, target_job=None, chat_log_pos=0, chat_log_len=1)


def test_restored_job_can_apply_summary(job_factory):
    source = source_job(job_factory)
    target = source_job(job_factory)
    target.set_status(JobStatus.SUSPENDED)
    job = SummarizationJob(source, 2, target_job=target, chat_log_pos=1, chat_log_len=2)
    job_uuid, target_uuid, source_uuid = db0.uuid(job), db0.uuid(target), db0.uuid(source)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(job_uuid)
    restored_target = db0.fetch(target_uuid)
    assert restored in list(db0.find(SummarizationJob, db0.as_tag(restored_target)))
    assert restored in list(db0.find(SummarizationJob, db0.as_tag(db0.fetch(source_uuid))))
    append_result(restored, result_text())
    restored.post_turn_hook()
    assert restored.status == JobStatus.DONE
    assert restored_target.status == JobStatus.STARTED
    record = restored_target._Job__job_summaries
    assert (record.chat_log_pos, record.chat_log_len) == (1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('style', ['CONSOLE', 'MARKDOWN', 'MD_DIALOG', 'DIRECT'])
async def test_execution_engine_retries_without_executing_json(job_factory, style):
    target = source_job(job_factory)
    target.job_def.set_chat_style(getattr(ChatStyle, style))
    target.set_status(JobStatus.SUSPENDED)
    job = SummarizationJob(target, 2, target_job=target, chat_log_pos=1, chat_log_len=1)
    push_summarization_instr(job)
    api = MagicMock()
    api.process_request = AsyncMock(side_effect=[
        LLM_Response(LLM_StepData(text, None), LLM_Stats(0, 0, None))
        for text in ('invalid', result_text())])
    with patch('statek.executors.utils.LLM_API.get', return_value=api), \
         patch('statek.executors.utils.get_llm_harness', return_value=MagicMock()), \
         patch('statek.executors.utils.handle_dialog', new_callable=AsyncMock), \
         patch('statek.executors.utils.exec_all_steps', new_callable=AsyncMock) as execute:
        assert await run_job_step(job) is False
        assert target.status == JobStatus.SUSPENDED
        assert job.status == JobStatus.STARTED
        assert await run_job_step(job) is True
        execute.assert_not_awaited()
    assert target.status == JobStatus.STARTED
    assert job.status == JobStatus.DONE
    assert api.process_request.await_count == 2
