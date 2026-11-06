"""Instructions for summarizing job history without executing the original task."""
# pylint: disable=no-member,protected-access
import json

import dbzero as db0
import pytest

from statek.chat_history import ChatRole, ContentSource, format_chat_history_item
from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import LLM_LogItem, SystemLogItem, UserLogItem
from statek.executors.forked_job import ForkedJob
from statek.executors.job import JobCompactionResult, JobSummary, PyHandleDesc, StateSummary
from statek.executors.summarization import push_summarization_instr
from tests.conftest import DB0_DIR, seed_pyenv


@pytest.mark.parametrize('style_name', ['CONSOLE', 'MARKDOWN', 'MD_DIALOG', 'DIRECT'])
@pytest.mark.parametrize('forked', [False, True])
def test_instruction_is_system_message_in_next_request(job_factory, style_name, forked):
    source = job_factory()
    style = getattr(ChatStyle, style_name)
    source.job_def.set_chat_style(style)
    source.chat_log = [UserLogItem('original request'), LLM_LogItem(0, llm_resp='work')]
    source.console_append('work result')
    job = ForkedJob(source, len(source.chat_log)) if forked else source
    history_before = list(job.get_chat_history())
    status_before = job.status
    assert push_summarization_instr(job) is None
    history = list(job.get_next_request()['chat_history'])
    assert history[:-1] == history_before
    instruction = history[-1]
    assert instruction.role == ChatRole.SYSTEM
    assert instruction.content_src == ContentSource.SYSTEM
    assert format_chat_history_item(instruction, style) == {
        'role': 'system', 'content': instruction.content}
    assert isinstance(job.chat_log[-1], SystemLogItem)
    assert job.chat_log[-1].console_pos == 1
    assert job.status == status_before
    assert job.num_turns == 1
    if forked:
        assert len(source.chat_log) == 2
        assert job.get_next_code_block() is None
    for field in ('input_summary', 'output_summary', 'system_summary', 'state',
                  'locals', 'var_name', 'desc'):
        assert f'"{field}"' in instruction.content
    assert 'English' in instruction.content
    assert 'JSON' in instruction.content
    assert 'no code' in instruction.content
    assert 'tool calls' in instruction.content


def test_instruction_lists_only_new_names_without_values(job_factory):
    job = job_factory()
    job.add_locals(warmup_handle=1)
    job._Job__init_warmup_summary()
    job.add_locals(previous_handle=2, zebra='SECRET VALUE', alpha=3)
    job.add_locals(**{'quoted"name\n': 'ANOTHER SECRET'})
    job._Job__job_summaries = JobCompactionResult(
        0, 1, JobSummary('', '', state=StateSummary({PyHandleDesc('previous_handle')})))
    job.py_env = seed_pyenv(job.py_env, global_state={'global_only': 4})
    push_summarization_instr(job)
    message = job.chat_log[-1].message
    names = json.loads(message.split('New state variable names (JSON):\n')[1])
    assert names == ['alpha', 'quoted"name\n', 'zebra']
    for excluded in ('warmup_handle', 'previous_handle', 'global_only',
                     'SECRET VALUE', 'ANOTHER SECRET'):
        assert excluded not in message
    assert dict(job.py_env.iter_locals())['zebra'] == 'SECRET VALUE'


def test_empty_names_and_repeated_instructions(job_factory):
    job = job_factory()
    push_summarization_instr(job)
    first = job.chat_log[-1].message
    assert json.loads(first.split('New state variable names (JSON):\n')[1]) == []
    assert '"locals": []' in first
    size = len(job.chat_log)
    job.add_locals(new_handle=1)
    push_summarization_instr(job)
    assert len(job.chat_log) == size + 1
    assert job.chat_log[-2].message == first
    assert json.loads(job.chat_log[-1].message.split(
        'New state variable names (JSON):\n')[1]) == ['new_handle']


def test_console_boundaries_and_historical_request(job_factory):
    job = job_factory()
    job.chat_log = [UserLogItem('request'), LLM_LogItem(0, llm_resp='first')]
    job.console_append('first result')
    job.py_env.push_append('follow-up')
    before = list(job.get_chat_history())
    push_summarization_instr(job)
    message = job.chat_log[-1].message
    job.console_append('later output')
    assert list(job.get_chat_history())[:-1] == before
    job.chat_log.append(LLM_LogItem(2, llm_resp='summary response'))
    history = list(job.get_request_data(1)['chat_history'])
    assert history[:-1] == before
    assert history[-1].content == message
    assert history[-1].role == ChatRole.SYSTEM


def test_instruction_uses_source_view_after_database_reopen(job_factory):
    source = job_factory()
    push_summarization_instr(source)
    fork = ForkedJob(source, len(source.chat_log))
    source.chat_log[-1].message = 'changed'
    message = source.chat_log[-1].message
    assert fork.chat_log[-1].message == message
    uuid = db0.uuid(fork)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(uuid)
    assert isinstance(restored.chat_log[-1], SystemLogItem)
    history = list(restored.get_next_request()['chat_history'])
    assert history[-1].role == ChatRole.SYSTEM
    assert history[-1].content == message
