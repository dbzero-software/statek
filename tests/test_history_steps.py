"""Incremental history must not materialize later turn payloads."""
# pylint: disable=no-member,protected-access
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from statek.executors.chat_log_item import (
    LLM_LogItem, UserLogItem, WarmupLogItem, NotificationLogItem,
)
from statek.executors.job import JobCompactionResult, JobSummary
from statek.executors.job_history import JobHistory
from statek.chat_style import ChatStyle
from statek.pyenv import PyEnv
from tests.conftest import seed_pyenv


def test_history_steps_are_bounded_and_match_history(job_factory, monkeypatch):
    job = job_factory()
    for number in range(100):
        job.chat_log.append(LLM_LogItem(console_pos=number, llm_resp=f'print({number})'))
        job.console_append(str(number))
    expected = list(job.get_chat_history())
    reads = []
    original = PyEnv.iter_console

    def console_slice(env, from_pos=0, to_pos=None):
        assert to_pos is not None, 'unbounded console read'
        reads.append((from_pos, to_pos))
        yield from original(env, from_pos, to_pos)

    monkeypatch.setattr(PyEnv, 'iter_console', console_slice)
    steps = job.iter_history_steps()
    assert not reads
    first = next(steps)
    assert first.turn_num == 0
    assert reads == [(0, 1)]
    assert first.log_item == job.chat_log[0]
    actual = list(first.messages)
    for step in steps:
        actual.extend(step.messages)
    assert actual == expected
    assert reads == [(n, n + 1) for n in range(100)]


def test_history_steps_capture_append_boundaries(job_factory):
    job = job_factory()
    job.chat_log.append(LLM_LogItem(console_pos=0, llm_resp='first'))
    job.console_append('first output')
    steps = job.iter_history_steps()
    job.chat_log.append(LLM_LogItem(console_pos=1, llm_resp='later'))
    job.console_append('later output')
    result = [step for step in steps if not step.final]
    assert len(result) == 1
    assert [m.content for m in result[0].messages] == ['first', 'first output']


def test_history_steps_match_mixed_compacted_history(job_factory):
    job = job_factory(warmup_code='setup()')
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.chat_log = [UserLogItem('hello'), WarmupLogItem(console_pos=0, warmup_block_num=0),
                    LLM_LogItem(console_pos=1, llm_resp='hidden'),
                    NotificationLogItem(console_pos=2, message='notice'),
                    LLM_LogItem(console_pos=2, llm_resp='visible')]
    job.py_env = seed_pyenv(job.py_env, console=['ready', 'hidden output', 'visible output'],
                           push_log={0: 'initial', 2: 'hidden push', 3: 'last push'})
    job._Job__job_summaries = JobCompactionResult(2, 1, JobSummary('input', 'output'))
    for compacted in (False, True):
        expected = list(job.get_chat_history(compacted=compacted))
        steps = list(job.iter_history_steps(compacted=compacted))
        assert [message for step in steps for message in step.messages] == expected
        assert steps[-1].final


def test_first_step_log_reads_do_not_grow_with_job_size(job_factory):
    reads = []

    class Log:
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            if index >= self.length:
                raise IndexError(index)
            reads.append(index)
            return LLM_LogItem(console_pos=index, llm_resp='same first turn')

    job = job_factory()
    for length in (10, 10000):
        reads.clear()
        env = MagicMock(console_len=length)
        env.iter_console.return_value = iter(['output'])
        fake = SimpleNamespace(chat_log=Log(length), py_env=env, job_def=job.job_def)
        next(JobHistory(fake).iter_history_steps())
        assert max(reads) == 1
        assert len(reads) <= 6
        env.iter_console.assert_called_once_with(0, 1)


def test_failed_read_retries_same_snapshot(job_factory, monkeypatch):
    job = job_factory()
    for number in range(2):
        job.chat_log.append(LLM_LogItem(console_pos=number, llm_resp=str(number)))
        job.console_append(str(number))
    steps = job.iter_history_steps()
    assert next(steps).turn_num == 0
    original = PyEnv.iter_console
    failed = False

    def read(env, start=0, stop=None):
        nonlocal failed
        if start == 1 and not failed:
            failed = True
            raise RuntimeError('transient read failure')
        yield from original(env, start, stop)

    monkeypatch.setattr(PyEnv, 'iter_console', read)
    with pytest.raises(RuntimeError, match='transient'):
        next(steps)
    job.chat_log.append(LLM_LogItem(console_pos=2, llm_resp='new'))
    job.console_append('new')
    remaining = [step for step in steps if not step.final]
    assert [step.turn_num for step in remaining] == [1]
    assert remaining[0].console == ('1',)
