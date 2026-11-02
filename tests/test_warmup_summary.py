"""Warmup state is captured once, only after the last block completes."""
# pylint: disable=no-member,protected-access

from unittest.mock import AsyncMock, MagicMock, patch

import dbzero as db0
import pytest

from statek.executors.utils import run_job_step
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from tests.conftest import DB0_DIR
from tests.test_warmup_blocks import (
    create_future_not_ready, _fetch_result_from_deps, _check_condition_true,
)


def _descriptions(job):
    return {item.var_name: item.desc for item in job._Job__warmup_summary.state.locals}


@pytest.mark.parametrize('locals_state', [None, {}, {'injected': 1, '_private': None}])
def test_summary_records_names_once_and_survives_reopen(job_factory, locals_state):
    job = job_factory()
    assert job._Job__warmup_summary is None
    job.py_env.local_state = locals_state
    job.py_env.global_state = {'global_only': 42}
    job._Job__init_warmup_summary()
    summary = job._Job__warmup_summary
    expected = {key: None for key in locals_state or {}}
    assert _descriptions(job) == expected
    assert summary.input_summary == ''
    assert summary.output_summary == ''
    assert summary.system_summary is None
    assert not list(job.get_chat_history())
    assert not list(job.get_compacted_chat_log())
    summary_id = db0.uuid(summary)
    job.add_locals(after_warmup=42)
    job._Job__init_warmup_summary()
    assert db0.uuid(job._Job__warmup_summary) == summary_id
    assert _descriptions(job) == expected
    job_id = db0.uuid(job)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open('test_prefix', 'rw')
    restored = db0.fetch(job_id)
    assert _descriptions(restored) == expected


@pytest.mark.asyncio
async def test_summary_after_last_block_exit(job_factory):
    job = job_factory(warmup_code=['temporary = 1\nkept = 2',
                                   'final = 3\nexit("done")'])
    job.add_locals(injected=0)
    assert not await run_job_step(job)
    assert job._Job__warmup_summary is None
    job.py_env.local_state.pop('temporary')
    assert await run_job_step(job)
    assert _descriptions(job) == {'injected': None, 'kept': None, 'final': None}


@pytest.mark.asyncio
async def test_summary_exists_before_first_llm_request(job_factory):
    job = job_factory(warmup_code='warmup_value = 42')

    async def process(**request):
        assert _descriptions(job) == {'warmup_value': None}
        assert all('Input Summary' not in (item.content or '')
                   for item in request['chat_history'])
        return LLM_Response(LLM_StepData('later = 1', None), LLM_Stats(0, 0, None))

    api = MagicMock()
    api.process_request = AsyncMock(side_effect=process)
    with patch('statek.executors.utils.LLM_API.get', return_value=api):
        await run_job_step(job)
    api.process_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_last_block_waits_for_future_before_summary(job_factory):
    job = job_factory(warmup_code='value = future_value\nexit("done")')
    future = create_future_not_ready()
    job.add_locals(future_value=future)
    assert not await run_job_step(job)
    assert job._Job__warmup_summary is None
    future.deps.value = 42
    future.set_complement_functions(complement=_fetch_result_from_deps,
                                    condition=_check_condition_true)
    assert await run_job_step(job)
    assert _descriptions(job) == {'future_value': None, 'value': None}


@pytest.mark.asyncio
@pytest.mark.parametrize('warmup', [
    ['exit("early")', 'unreached = 1'],
    'partial = 1\nraise RuntimeError("failed warmup")',
])
async def test_unfinished_warmup_has_no_summary(job_factory, warmup):
    job = job_factory(warmup_code=warmup)
    assert await run_job_step(job)
    assert job._Job__warmup_summary is None


@pytest.mark.asyncio
async def test_no_warmup_leaves_summary_unset(job_factory):
    job = job_factory()
    api = MagicMock()
    api.process_request = AsyncMock(return_value=LLM_Response(
        LLM_StepData('later = 1', None), LLM_Stats(0, 0, None)))
    with patch('statek.executors.utils.LLM_API.get', return_value=api):
        await run_job_step(job)
    assert job._Job__warmup_summary is None
