"""Independent continuations of existing jobs."""
# pylint: disable=no-member,protected-access
from unittest.mock import AsyncMock, MagicMock, patch

import dbzero as db0
import pytest
from statek.llm_harness import LLM_Harness
from statek.executors.chat_log_item import ToolError
from statek.utils import CallSpec, CodeBlock
from statek.executors.utils import run_job_step
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from statek.executors.forked_job import ForkedJob
from statek.executors.job import Job, JobStatus, JobCompactionResult, JobSummary
from statek.executors.chat_log_item import LLM_LogItem, UserLogItem, WarmupLogItem
from statek.pyenv import PyEnv
from tests.conftest import DB0_DIR


def source_job(job_factory):
    job = job_factory(warmup_code="setup()")
    job.chat_log = [UserLogItem("initial"), WarmupLogItem(0),
                    LLM_LogItem(1, llm_resp="first()"),
                    UserLogItem("follow-up"), LLM_LogItem(2, llm_resp="second()")]
    job.py_env.console = ["ready", "first output", "second output"]
    job.py_env.push_log = {2: "pushed follow-up", 3: "pending"}
    return job


@pytest.mark.parametrize("boundary, expected", [
    (0, ["initial", "setup()", "ready"]),
    (1, ["initial", "setup()", "ready", "first()", "first output"]),
    (2, ["initial", "setup()", "ready", "first()", "first output",
         "pushed follow-up", "follow-up", "second()", "second output"]),
])
def test_fork_history_boundary(job_factory, boundary, expected):
    source = source_job(job_factory)
    fork = ForkedJob(source, boundary)
    assert [item.content for item in fork.get_chat_history()] == expected
    assert fork.num_turns == boundary
    assert fork.num_execution_turns == 0
    assert fork.status == JobStatus.STARTED
    assert fork.last_response is None
    assert fork.get_next_code_block() is None
    assert fork in list(db0.find(Job, JobStatus.STARTED))


def test_fork_preserves_history_and_owns_execution(job_factory):
    source = source_job(job_factory)
    handle = PyEnv()
    source.add_locals(shared=handle, value=1)
    fork = ForkedJob(source, 1)
    before = [item.content for item in fork.get_chat_history()]
    source.chat_log[2].llm_resp = "changed"
    source.py_env.console[1] = "changed output"
    source.py_env.push_log[0] = "new initial"
    assert [item.content for item in fork.get_chat_history()] == before
    fork.add_locals(value=2)
    assert source.py_env.local_state['value'] == 1
    assert fork.py_env.local_state['shared'] == handle
    fork.chat_log.append(LLM_LogItem(2, llm_resp="fork code"))
    fork.console_append("fork output")
    assert fork.num_turns == 2
    assert fork.num_execution_turns == 1
    assert fork.get_next_code_block() == "fork code"
    assert len(source.py_env.console) == 3
    nested = ForkedJob(fork, 2)
    assert nested.num_execution_turns == 0
    assert list(nested.get_chat_history())[-1].content == "fork output"
    uuid = db0.uuid(nested)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(uuid)
    assert restored.forked_at == 2
    assert restored.get_next_code_block() is None
    assert list(restored.get_chat_history())[-1].content == "fork output"


def test_fork_compaction(job_factory):
    source = source_job(job_factory)
    source._Job__job_summaries = JobCompactionResult(2, 3, JobSummary('input', 'output'))
    partial = ForkedJob(source, 1)
    assert list(partial.get_chat_history())[-2].content == 'first()'
    full = ForkedJob(source, 2)
    assert 'Input Summary' in list(full.get_chat_history())[-1].content


@pytest.mark.parametrize('boundary', [-1, 3, 0.5, True])
def test_invalid_boundary(job_factory, boundary):
    with pytest.raises((ValueError, TypeError)):
        ForkedJob(source_job(job_factory), boundary)


def test_fork_has_fresh_limits_and_copies_tool_history(job_factory):
    source = job_factory()
    source.chat_log = [LLM_LogItem(0, llm_resp=CodeBlock(tool_calls=[
        CallSpec('call', 'tool', kwargs={'x': [1]})]), tool_log=[ToolError('old error')])]
    source.usage.total_bytes_sent = 1000
    fork = ForkedJob(source, 1)
    source.chat_log[0].tool_log[0].err_message = 'changed'
    source.chat_log[0].llm_resp.tool_calls[0].kwargs['x'].append(2)
    assert fork.chat_log[0].tool_log[0].err_message == 'old error'
    assert list(fork.chat_log[0].llm_resp.tool_calls[0].kwargs['x']) == [1]
    assert fork.exception_count == 0
    assert fork.count_llm_actions() == 0
    assert fork.approx_token_usage == 0
    harness = LLM_Harness(max_turns=0, max_exceptions=0,
                          max_consecutive_exceptions=0, max_token_usage=0)
    harness.check_before_step(fork)
    fork.chat_log.append(LLM_LogItem(0, llm_resp='new', tool_log=ToolError('new error')))
    assert fork.exception_count == 1


def test_pending_response_and_missing_tool_result_are_rejected(job_factory):
    source = job_factory()
    source.chat_log = [LLM_LogItem(0)]
    with pytest.raises(ValueError, match='pending'):
        ForkedJob(source, 1)
    source.chat_log[0].llm_resp = CodeBlock(tool_calls=[CallSpec('call', 'tool')])
    with pytest.raises(ValueError, match='incomplete'):
        ForkedJob(source, 1)


@pytest.mark.asyncio
async def test_engine_starts_with_inference_then_runs_only_fork_code(job_factory):
    source = source_job(job_factory)
    fork = ForkedJob(source, 2)
    response = LLM_Response(
        step_data=LLM_StepData(text='x = 42', call_requests=None),
        stats=LLM_Stats(0, 0, None))
    api = MagicMock()
    api.process_request = AsyncMock(return_value=response)
    with patch('statek.executors.utils.LLM_API.get', return_value=api), \
         patch('statek.executors.utils.exec_all_steps', new_callable=AsyncMock) as execute:
        await run_job_step(fork)
        execute.assert_not_called()
        await run_job_step(fork)
        execute.assert_awaited_once()
        assert execute.call_args.args[0].code == 'x = 42'
    assert fork.num_execution_turns == 2
    assert source.num_turns == 2
    assert len(source.py_env.console) == 3


def test_fork_requests_and_slices_use_combined_timeline(job_factory):
    source = source_job(job_factory)
    fork = ForkedJob(source, 1)
    fork.chat_log.extend([UserLogItem('fork prompt'), LLM_LogItem(2, llm_resp='fork answer')])
    request = fork.get_request_data(1)
    contents = [item.content for item in request['chat_history']]
    assert contents == ['initial', 'setup()', 'ready', 'first()', 'first output', 'fork prompt']
    assert [item.content for item in fork.get_chat_history(start_at=1, num_turns=1)] == [
        'fork prompt', 'fork answer']
