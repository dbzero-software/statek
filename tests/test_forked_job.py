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
from statek.executors.job import Job, JobStatus, JobCompactionResult, JobSummary, TaskDifficulty
from statek.executors.chat_log_item import LLM_LogItem, UserLogItem, WarmupLogItem
from statek.pyenv import Error, ErrorKind, PyEnv, PyEnvForked
from tests.conftest import DB0_DIR, seed_pyenv


@pytest.mark.parametrize('initial_difficulty', [None, TaskDifficulty.medium])
def test_fork_initializes_difficulty_without_temporary_members(job_def_factory,
                                                              initial_difficulty):
    source = Job(job_def_factory(metadata={
        'MODEL': 'L:small,M:medium,H:large', 'DEFAULT_DIFFICULTY': 'low',
    }))
    source._Job__last_difficulty = initial_difficulty
    source.py_env.update_state(**{'default_example_id': 1})
    with patch('statek.executors.job._get_example_difficulty_for_job',
               return_value=TaskDifficulty.high):
        fork = ForkedJob(source, 0)
    assert fork.get_current_difficulty() == TaskDifficulty.high
    assert len(fork.chat_log) == 1
    assert fork.chat_log[0].previous_difficulty == (initial_difficulty or TaskDifficulty.low)
    assert fork.chat_log[0].difficulty == TaskDifficulty.high
    assert fork.chat_log[0].cumulative_token_usage == 0
    assert fork.difficulty_token_extension == 0
    assert not source.chat_log
    assert source._Job__last_difficulty == initial_difficulty
    assert not hasattr(fork, '_ForkedJob__initial_difficulty')
    assert not hasattr(fork, '_ForkedJob__initializing')


def test_fork_keeps_source_difficulty_after_reopen(job_def_factory):
    source = Job(job_def_factory(metadata={
        'MODEL': 'L:small,M:medium,H:large', 'DEFAULT_DIFFICULTY': 'low',
    }))
    source.panic()
    fork = ForkedJob(source, len(source.chat_log))
    assert fork.get_current_model() == 'medium'
    assert not fork.chat_log.tail
    identifier = db0.uuid(fork)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(identifier)
    assert restored.get_current_model() == 'medium'
    assert not restored.chat_log.tail
    assert not hasattr(restored, '_ForkedJob__initial_difficulty')
    assert not hasattr(restored, '_ForkedJob__initializing')


def source_job(job_factory):
    job = job_factory(warmup_code="setup()")
    job.chat_log = [UserLogItem("initial"), WarmupLogItem(0),
                    LLM_LogItem(1, llm_resp="first()"),
                    UserLogItem("follow-up"), LLM_LogItem(2, llm_resp="second()")]
    job.py_env = seed_pyenv(
        job.py_env, console=["ready", "first output", "second output"],
        push_log={2: "pushed follow-up", 3: "pending"})
    return job


@pytest.mark.parametrize("boundary, expected", [
    (0, []),
    (1, ["initial"]),
    (2, ["initial", "setup()", "ready"]),
    (3, ["initial", "setup()", "ready", "first()", "first output"]),
    (4, ["initial", "setup()", "ready", "first()", "first output", "follow-up"]),
    (5, ["initial", "setup()", "ready", "first()", "first output",
         "pushed follow-up", "follow-up", "second()", "second output", "pending"]),
])
def test_fork_history_boundary(job_factory, boundary, expected):
    source = source_job(job_factory)
    fork = ForkedJob(source, boundary)
    assert [item.content for item in fork.get_chat_history()] == expected
    assert fork.forked_at == boundary
    assert len(fork.chat_log) == boundary
    assert fork.num_turns == sum(isinstance(item, LLM_LogItem)
                                 for item in source.chat_log[:boundary])
    assert fork.num_execution_turns == 0
    assert fork.status == JobStatus.STARTED
    assert fork.last_response is None
    assert fork.get_next_code_block() is None
    assert fork in list(db0.find(Job, JobStatus.STARTED))


def test_fork_views_history_and_owns_execution(job_factory):
    source = source_job(job_factory)
    handle = PyEnv()
    source.add_locals(shared=handle, value=1)
    fork = ForkedJob(source, 3)
    source.chat_log[2].llm_resp = "changed"
    source.py_env._console[1] = "changed output"
    source.py_env._push_log[0] = "new initial"
    assert [item.content for item in fork.get_chat_history()] == [
        'initial\nnew initial', 'setup()', 'ready', 'changed', 'changed output']
    assert db0.uuid(fork.chat_log[2]) == db0.uuid(source.chat_log[2])
    assert not fork.chat_log.tail
    assert not fork.py_env._console
    fork.add_locals(value=2)
    assert source.py_env.get_local('value') == 1
    assert fork.py_env.get_local('shared') == handle
    fork.chat_log.append(LLM_LogItem(2, llm_resp="fork code"))
    fork.console_append("fork output")
    assert fork.num_turns == 2
    assert fork.num_execution_turns == 1
    assert fork.get_next_code_block() == "fork code"
    assert source.py_env.console_len == 3
    nested = ForkedJob(fork, len(fork.chat_log))
    assert nested.num_execution_turns == 0
    assert list(nested.get_chat_history())[-1].content == "fork output"
    uuid = db0.uuid(nested)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(uuid)
    assert restored.forked_at == 4
    assert restored.get_next_code_block() is None
    assert list(restored.get_chat_history())[-1].content == "fork output"


def test_fork_compaction(job_factory):
    source = source_job(job_factory)
    source._Job__job_summaries = JobCompactionResult(2, 3, JobSummary('input', 'output'))
    partial = ForkedJob(source, 3)
    assert list(partial.get_chat_history())[-2].content == 'first()'
    full = ForkedJob(source, len(source.chat_log))
    assert 'Input Summary' in list(full.get_chat_history())[-1].content


@pytest.mark.parametrize('boundary', [-1, 6, 0.5, True])
def test_invalid_boundary(job_factory, boundary):
    with pytest.raises((ValueError, TypeError)):
        ForkedJob(source_job(job_factory), boundary)


def test_fork_has_fresh_limits_and_views_tool_history(job_factory):
    source = job_factory()
    source.chat_log = [LLM_LogItem(0, llm_resp=CodeBlock(tool_calls=[
        CallSpec('call', 'tool', kwargs={'x': [1]})]), tool_log=[ToolError('old error')])]
    source.usage.total_bytes_sent = 1000
    fork = ForkedJob(source, 1)
    source.chat_log[0].tool_log[0].err_message = 'changed'
    source.chat_log[0].llm_resp.tool_calls[0].kwargs['x'].append(2)
    assert fork.chat_log[0].tool_log[0].err_message == 'changed'
    assert list(fork.chat_log[0].llm_resp.tool_calls[0].kwargs['x']) == [1, 2]
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
    fork = ForkedJob(source, len(source.chat_log))
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
    assert source.py_env.console_len == 3


def test_fork_requests_and_slices_use_combined_timeline(job_factory):
    source = source_job(job_factory)
    fork = ForkedJob(source, 3)
    fork.chat_log.extend([UserLogItem('fork prompt'), LLM_LogItem(2, llm_resp='fork answer')])
    request = fork.get_request_data(1)
    contents = [item.content for item in request['chat_history']]
    assert contents == ['initial', 'setup()', 'ready', 'first()', 'first output', 'fork prompt']
    assert [item.content for item in fork.get_chat_history(start_at=1, num_turns=1)] == [
        'fork prompt', 'fork answer']


def test_fork_views_warmup_baseline(job_factory):
    source = source_job(job_factory)
    source.add_locals(warmup_name=1)
    source._Job__init_warmup_summary()
    source.add_locals(later_name=2)
    fork = ForkedJob(source, 3)
    summary = fork._Job__warmup_summary
    assert summary is not None
    assert db0.uuid(summary) == db0.uuid(source._Job__warmup_summary)
    assert {item.var_name for item in summary.state.locals} == {'warmup_name'}
    source._Job__warmup_summary.state.locals.clear()
    assert not summary.state.locals
    assert ForkedJob(source, 1)._Job__warmup_summary is None


def test_raw_cutoff_does_not_hide_local_exceptions(job_factory):
    source = source_job(job_factory)
    source.chat_log[2].tool_log = ToolError('source error')
    fork = ForkedJob(source, 3)
    fork.chat_log.append(LLM_LogItem(2, llm_resp='new', tool_log=ToolError('fork error')))
    assert fork.num_execution_turns == 1
    assert fork.exception_count == 1
    assert fork.max_consecutive_exceptions == 1


def test_full_fork_keeps_trailing_user_messages(job_factory):
    source = source_job(job_factory)
    source.chat_log.append(UserLogItem('last message'))
    fork = ForkedJob(source, len(source.chat_log))
    assert [item.content for item in fork.get_chat_history()] == [
        item.content for item in source.get_chat_history()]


def test_push_log_is_bounded_by_console_position(job_factory):
    source = job_factory()
    source.chat_log = [LLM_LogItem(0, llm_resp='first'),
                       LLM_LogItem(2, llm_resp='second')]
    source.py_env = seed_pyenv(
        source.py_env, console=['out1', 'out2', 'out3'],
        push_log={0: ['initial'], 1: ['during first'],
                  2: 'cutoff message', 3: 'trailing'})
    partial = ForkedJob(source, 1)
    assert [item.content for item in partial.get_chat_history()] == [
        'initial', 'first', 'out1\nout2', 'during first']
    assert list(partial.py_env.iter_console()) == ['out1', 'out2']
    full = ForkedJob(source, len(source.chat_log))
    assert [item.content for item in full.get_chat_history()] == [
        item.content for item in source.get_chat_history()]
    assert not list(ForkedJob(source, 0).get_chat_history())


def test_fork_reports_raw_boundary_after_reopen(job_factory):
    source = source_job(job_factory)
    fork = ForkedJob(source, 3)
    fork.chat_log.append(LLM_LogItem(2, llm_resp='new', tool_log=ToolError('local error')))
    uuid = db0.uuid(fork)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(uuid)
    assert restored.forked_at == 3
    assert restored.exception_count == 1
    assert restored.num_execution_turns == 1


def test_view_stays_bounded_and_is_live_after_reopen(job_factory):
    source = source_job(job_factory)
    source.py_env._push_log[1] = "original push"
    fork = ForkedJob(source, 3)
    nested = ForkedJob(fork, 3)
    fork.chat_log.append(LLM_LogItem(2, llm_resp='local'))
    fork.console_append('local output')
    source.chat_log.append(LLM_LogItem(3, llm_resp='excluded'))
    source.console_append('excluded output')
    source.chat_log[2].llm_resp = 'updated'
    source.py_env._console[1] = 'updated output'
    source.py_env._push_log[1] = 'updated push'
    uuid = db0.uuid(nested)
    source_uuid = db0.uuid(source)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(uuid)
    assert len(restored.chat_log) == 3
    assert list(restored.py_env.iter_console()) == ['ready', 'updated output']
    assert restored.chat_log[-1].llm_resp == 'updated'
    assert dict(restored.py_env.iter_push_log())[1] == 'updated push'
    db0.fetch(source_uuid).chat_log[2].llm_resp = 'changed after reopen'
    assert restored.chat_log[-1].llm_resp == 'changed after reopen'
    assert restored.chat_log[::-1] == list(reversed(restored.chat_log))
    assert not restored.chat_log.tail
    with pytest.raises(TypeError, match='inherited'):
        restored.chat_log[0] = UserLogItem('overwrite')


def test_fork_views_summaries_and_keeps_local_logs_separate(job_factory):
    source = source_job(job_factory)
    result = JobCompactionResult(2, 3, JobSummary('input', 'output'))
    source._Job__job_summaries = result
    fork = ForkedJob(source, len(source.chat_log))
    result.job_summary.input_summary = 'updated summary'
    assert 'updated summary' in list(fork.get_chat_history())[-1].content
    fork.py_env.push_append('fork push')
    assert dict(source.py_env.iter_push_log())[3] == 'pending'
    assert dict(fork.py_env.iter_push_log())[3] == ['pending', 'fork push']
    fork.chat_log.append(LLM_LogItem(3, llm_resp='original'))
    fork.chat_log[-1] = LLM_LogItem(3, llm_resp='replaced')
    assert fork.chat_log[-1].llm_resp == 'replaced'
    assert len(source.chat_log) == 5


def test_partial_environment_view_keeps_local_errors_at_cutoff(job_factory):
    source = source_job(job_factory)
    source.py_env = seed_pyenv(
        source.py_env, exceptions={1: Error(ErrorKind.EXECUTION, 'inherited'),
                                   2: Error(ErrorKind.HARNESS, 'excluded')})
    fork = ForkedJob(source, 3)
    assert isinstance(fork.py_env, PyEnvForked)
    assert fork.py_env.get_exception(2) is None
    fork.py_env.push_append('fork push')
    fork.py_env.console_append('fork error', error=Error(ErrorKind.EXECUTION, 'local'))
    assert [(pos, error.message) for pos, error in fork.py_env.iter_exceptions()] == [
        (1, 'inherited'), (2, 'local')]
    assert dict(fork.py_env.iter_push_log())[2] == 'fork push'
    assert source.py_env.get_exception(2).message == 'excluded'
    assert dict(source.py_env.iter_push_log())[2] == 'pushed follow-up'
