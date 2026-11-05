"""Local names not yet described by warmup or compaction summaries."""
# pylint: disable=protected-access,no-member
from types import SimpleNamespace

import dbzero as db0
import pytest

from statek.executors.chat_log_item import LLM_LogItem, WarmupLogItem
from statek.executors.forked_job import ForkedJob
from statek.executors.job import Job, JobCompactionResult, JobSummary, PyHandleDesc, StateSummary
from tests.conftest import DB0_DIR


def _summary(*names):
    return JobSummary('', '', state=StateSummary({PyHandleDesc(name) for name in names}))


@pytest.mark.parametrize('local_state, expected', [
    (None, set()), ({}, set()), ({'value': None, '_private': 2}, {'value', '_private'}),
])
def test_current_local_names_without_summaries(job_factory, local_state, expected):
    job = job_factory()
    job.py_env.local_state = local_state
    job.py_env.global_state = {'global_only': 1}
    result = job.get_diff_locals()
    assert isinstance(result, set)
    assert result == expected


@pytest.mark.parametrize('multiple', [False, True])
def test_subtracts_warmup_and_all_compactions_without_mutation(job_factory, multiple):
    job = job_factory()
    job.add_locals(warmup=1, first=2, second=3, new=4)
    job._Job__warmup_summary = _summary('warmup', 'removed')
    first = JobCompactionResult(0, 1, _summary('first', 'warmup'))
    second = JobCompactionResult(1, 1, _summary('second', 'first'))
    no_state = JobCompactionResult(2, 1, JobSummary('', ''))
    job._Job__job_summaries = [first, second, no_state] if multiple else first
    count_before = len(db0.find(JobSummary))
    assert job.get_diff_locals() == ({'new'} if multiple else {'second', 'new'})
    assert dict(job.py_env.local_state) == {'warmup': 1, 'first': 2, 'second': 3, 'new': 4}
    assert {item.var_name for item in first.job_summary.state.locals} == {'first', 'warmup'}
    assert {item.var_name for item in job._Job__warmup_summary.state.locals} == {
        'warmup', 'removed'}
    assert len(db0.find(JobSummary)) == count_before


def test_known_name_stays_known_when_reassigned_or_reintroduced(job_factory):
    job = job_factory()
    job.add_locals(known=1)
    job._Job__init_warmup_summary()
    job.add_locals(known=2, new=3)
    assert job.get_diff_locals() == {'new'}
    job.py_env.local_state.pop('known')
    job.py_env.local_state.pop('new')
    assert job.get_diff_locals() == set()
    job.add_locals(known=4)
    assert job.get_diff_locals() == set()


def test_old_job_without_summary_attributes():
    old_job = SimpleNamespace(py_env=SimpleNamespace(local_state={'existing': 1}))
    assert Job.get_diff_locals(old_job) == {'existing'}


def test_empty_summary_states(job_factory):
    job = job_factory()
    job.add_locals(new=1)
    job._Job__warmup_summary = JobSummary('', '')
    job._Job__job_summaries = [JobCompactionResult(0, 1, _summary())]
    assert job.get_diff_locals() == {'new'}


def test_fork_and_reopened_job_use_inherited_baselines(job_factory):
    source = job_factory(warmup_code='warmup = 1')
    source.add_locals(warmup=1)
    source._Job__init_warmup_summary()
    source.add_locals(compacted=2, new=3)
    source.chat_log = [WarmupLogItem(0), LLM_LogItem(0, llm_resp='compacted = 2')]
    source._Job__job_summaries = JobCompactionResult(1, 1, _summary('compacted'))
    fork = ForkedJob(source, len(source.chat_log))
    fork.add_locals(fork_only=4)
    assert source.get_diff_locals() == {'new'}
    assert fork.get_diff_locals() == {'new', 'fork_only'}
    uuid = db0.uuid(fork)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    assert db0.fetch(uuid).get_diff_locals() == {'new', 'fork_only'}
