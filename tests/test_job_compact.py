"""Register compacted slices without modifying raw history or execution state."""
# pylint: disable=no-member,protected-access
import dbzero as db0
import pytest

from statek.executors.chat_log_item import LLM_LogItem, UserLogItem, WarmupLogItem
from statek.executors.forked_job import ForkedJob
from statek.executors.job import JobCompactionResult, JobSummary, StateSummary, PyHandleDesc
from tests.conftest import DB0_DIR


def test_register_summary_preserves_raw_history_and_state(job_factory):
    job = job_factory(warmup_code='setup()')
    job.chat_log = [UserLogItem('request'), WarmupLogItem(0),
                    LLM_LogItem(1, llm_resp='old()'), LLM_LogItem(2, llm_resp='recent()')]
    job.py_env.console = ['ready', 'old output', 'recent output']
    job.add_locals(handle=42, new=1)
    summary = JobSummary('request', 'old action', state=StateSummary({PyHandleDesc('handle')}))
    raw = list(job.get_chat_history(compacted=False))
    status = job.status
    console = list(job.py_env.console)
    history = job.get_chat_history()
    assert job.compact(2, 1, summary) is None
    record = job._Job__job_summaries
    assert isinstance(record, JobCompactionResult)
    assert (record.chat_log_pos, record.chat_log_len, record.job_summary) == (2, 1, summary)
    assert list(job.get_compacted_chat_log()) == [*job.chat_log[:2], summary, job.chat_log[3]]
    contents = [item.content for item in history]
    assert contents[:3] == ['request', 'setup()', 'ready']
    assert 'old action' in contents[3]
    assert contents[-2:] == ['recent()', 'recent output']
    assert list(job.get_chat_history(compacted=False)) == raw
    assert list(job.py_env.console) == console
    assert dict(job.py_env.local_state) == {'handle': 42, 'new': 1}
    assert job.status == status
    assert job.get_diff_locals() == {'new'}


def test_repeated_registration_appends_without_replacing(job_factory):
    job = job_factory()
    job.chat_log = [LLM_LogItem(0, llm_resp=str(i)) for i in range(4)]
    summaries = [JobSummary(str(i), '') for i in range(3)]
    job.compact(2, 1, summaries[0])
    first = job._Job__job_summaries
    job.compact(0, 1, summaries[1])
    job.compact(1, 1, summaries[2])
    assert list(job._Job__job_summaries)[0] == first
    assert [r.job_summary for r in job._Job__job_summaries] == summaries
    assert list(job.get_compacted_chat_log()) == [
        summaries[1], summaries[2], summaries[0], job.chat_log[3]]


@pytest.mark.parametrize('pos, length, error', [
    (-1, 1, ValueError), (0, 0, ValueError), (0, -1, ValueError),
    (4, 1, ValueError), (3, 2, ValueError),
    (True, 1, TypeError), (0, True, TypeError), (0.5, 1, TypeError),
    (0, '1', TypeError), (None, 1, TypeError),
])
def test_invalid_range_leaves_summaries_unchanged(job_factory, pos, length, error):
    job = job_factory()
    job.chat_log = [LLM_LogItem(0, llm_resp='work') for _ in range(4)]
    summary = JobSummary('', '')
    before = len(db0.find(JobCompactionResult))
    with pytest.raises(error):
        job.compact(pos, length, summary)
    assert job._Job__job_summaries is None
    assert len(db0.find(JobCompactionResult)) == before


@pytest.mark.parametrize('pos, length', [(0, 2), (1, 1), (2, 1), (2, 2), (0, 4)])
def test_overlap_rejected_without_replacing_previous(job_factory, pos, length):
    job = job_factory()
    job.chat_log = [LLM_LogItem(0, llm_resp='work') for _ in range(4)]
    summary = JobSummary('', '')
    job.compact(1, 2, summary)
    original = job._Job__job_summaries
    with pytest.raises(ValueError, match='overlap'):
        job.compact(pos, length, summary)
    assert job._Job__job_summaries == original
    assert len(db0.find(JobCompactionResult)) == 1


def test_warmup_and_wrong_summary_rejected(job_factory):
    job = job_factory(warmup_code='setup()')
    job.chat_log = [UserLogItem('request'), WarmupLogItem(0)]
    with pytest.raises(TypeError, match='JobSummary'):
        job.compact(0, 1, None)
    with pytest.raises(ValueError, match='warmup'):
        job.compact(0, 2, JobSummary('', ''))
    assert job._Job__job_summaries is None
    # Initial user messages can be summarized independently of warmup.
    job.compact(0, 1, JobSummary('request', ''))


def test_fork_empty_list_registration_and_persistence(job_factory):
    source = job_factory()
    source.chat_log = [LLM_LogItem(0, llm_resp='first'), LLM_LogItem(0, llm_resp='second')]
    fork = ForkedJob(source, 2)
    fork.compact(0, 1, JobSummary('fork summary', ''))
    assert source._Job__job_summaries is None
    assert 'fork summary' in list(fork.get_chat_history())[0].content
    source.compact(0, 1, JobSummary('source summary', ''))
    inherited = ForkedJob(source, 2)
    inherited.compact(1, 1, JobSummary('second summary', ''))
    assert isinstance(source._Job__job_summaries, JobCompactionResult)
    uuid = db0.uuid(inherited)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(uuid)
    assert [r.job_summary.input_summary for r in restored._Job__job_summaries] == [
        'source summary', 'second summary']


def test_compaction_record_uses_job_prefix(job_factory):
    job = job_factory()
    job.chat_log = [LLM_LogItem(0, llm_resp='work')]
    summary = JobSummary('', '')
    prefix = db0.get_prefix_of(job).name
    db0.open('other-prefix', 'rw')
    job.compact(0, 1, summary)
    assert db0.get_prefix_of(job._Job__job_summaries).name == prefix
