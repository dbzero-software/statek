"""Tests for run_agentic_loop / _make_start_jobs_func."""
# pylint: disable=no-member

import pytest
import dbzero as db0

from statek.agents.agent import Agent
from statek.prompt_config import make_system_prompt
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.job import parse_warmup_code
from statek.executors.utils import _make_start_jobs_func, find_existing_job_def, run_agentic_loop
from statek.pyenv import PyEnv


def _make_job_def(agent, warmup_code=None):
    return JobDef(
        agent=agent,
        metadata={"MODEL": "test-model"},
        job_params=None,
        warmup_code=warmup_code,
    )


def _make_job(job_def, status=JobStatus.READY):
    job = Job(
        job_def=job_def,
        model_family="test",
        model="test-model",
        job_status=status,
        py_env=PyEnv(local_state={}),
    )
    return job


class TestMakeStartJobsFunc:
    """Tests for the _make_start_jobs_func factory."""

    def test_circuit_breaker_raises_when_job_def_has_errors(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """start_jobs_func raises RuntimeError when job_def has errors."""
        job_def = _make_job_def(agent)
        try:
            raise ValueError("warmup failed")
        except ValueError as exc:
            job_def.set_error(exc)

        func = _make_start_jobs_func(agent, job_def, lambda: 5, provider=None)
        with pytest.raises(RuntimeError):
            func(capacity=10)

    def test_no_jobs_created_when_capacity_zero(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """No jobs are created when capacity is 0."""
        job_def = _make_job_def(agent)

        func = _make_start_jobs_func(agent, job_def, lambda: 5, provider=None)
        func(capacity=0)

        jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(jobs) == 0

    def test_no_jobs_created_when_queue_empty(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """No jobs are created when task_queue_size returns 0."""
        job_def = _make_job_def(agent)

        func = _make_start_jobs_func(agent, job_def, lambda: 0, provider=None)
        func(capacity=10)

        jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(jobs) == 0

    def test_creates_jobs_up_to_capacity(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Creates min(capacity, queue_size) jobs when no ready jobs exist."""
        job_def = _make_job_def(agent)

        func = _make_start_jobs_func(agent, job_def, lambda: 10, provider=None)
        func(capacity=3)

        jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(jobs) == 3

    def test_creates_jobs_limited_by_queue_size(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Creates min(capacity, queue_size) jobs when queue_size < capacity."""
        job_def = _make_job_def(agent)

        func = _make_start_jobs_func(agent, job_def, lambda: 2, provider=None)
        func(capacity=10)

        jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(jobs) == 2

    def test_deducts_existing_ready_jobs(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """ready_jobs already in READY state count against new jobs to create."""
        job_def = _make_job_def(agent)
        # Pre-create 2 READY jobs for this agent
        _make_job(job_def, status=JobStatus.READY)
        _make_job(job_def, status=JobStatus.READY)

        # queue=5, ready=2 → should create min(10, 5-2) = 3
        func = _make_start_jobs_func(agent, job_def, lambda: 5, provider=None)
        func(capacity=10)

        all_jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(all_jobs) == 5  # 2 pre-existing + 3 new

    def test_deducts_warming_up_jobs(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Jobs in WARMING_UP state also count against new jobs to create."""
        job_def = _make_job_def(agent)
        # Pre-create 1 WARMING_UP job
        _make_job(job_def, status=JobStatus.WARMING_UP)

        # queue=4, ready_or_warming_up=1 → min(10, 4-1) = 3
        func = _make_start_jobs_func(agent, job_def, lambda: 4, provider=None)
        func(capacity=10)

        all_jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(all_jobs) == 4  # 1 pre-existing + 3 new

    def test_no_jobs_when_ready_jobs_cover_queue(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """No new jobs created when ready_jobs >= queue_size."""
        job_def = _make_job_def(agent)
        _make_job(job_def, status=JobStatus.READY)
        _make_job(job_def, status=JobStatus.READY)
        _make_job(job_def, status=JobStatus.READY)

        # queue=2, ready=3 → min(10, 2-3) = min(10, -1) → 0
        func = _make_start_jobs_func(agent, job_def, lambda: 2, provider=None)
        func(capacity=10)

        all_jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(all_jobs) == 3  # no new jobs created

    def test_only_counts_this_agents_ready_jobs(self, db0_fixture):  # pylint: disable=unused-argument
        """Only READY/WARMING_UP jobs for this specific agent are counted."""
        agent_a = Agent(role="agent_a", _system_prompt=make_system_prompt("A"), _tools=[])
        agent_b = Agent(role="agent_b", _system_prompt=make_system_prompt("B"), _tools=[])

        job_def_a = _make_job_def(agent_a)
        job_def_b = _make_job_def(agent_b)

        # Create 3 READY jobs for agent_b — should not affect agent_a's count
        for _ in range(3):
            _make_job(job_def_b, status=JobStatus.READY)

        # queue=4, agent_a ready=0 → min(10, 4) = 4
        func = _make_start_jobs_func(agent_a, job_def_a, lambda: 4, provider=None)
        func(capacity=10)

        agent_a_jobs = list(db0.find(Job, db0.as_tag(agent_a)))
        assert len(agent_a_jobs) == 4


def _make_tagged_job_def(agent, warmup_code=None):
    """Create a JobDef (automatically tagged with its agent via __post_init__)."""
    parsed = parse_warmup_code(warmup_code)
    return JobDef(
        agent=agent,
        metadata={"MODEL": "test-model"},
        job_params=None,
        warmup_code=parsed,
    )


class TestFindExistingJobDef:
    """Tests for find_existing_job_def."""

    def test_returns_none_when_no_job_defs(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Returns None when no JobDefs are tagged with the agent."""
        result = find_existing_job_def(agent, "print('hello')")
        assert result is None

    def test_finds_matching_job_def(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Returns existing JobDef when agent and warmup_code both match."""
        job_def = _make_tagged_job_def(agent, "print('hello')")

        result = find_existing_job_def(agent, "print('hello')")
        assert result is job_def

    def test_returns_none_when_warmup_code_differs(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Returns None when a JobDef for the agent exists but warmup_code differs."""
        _make_tagged_job_def(agent, "print('hello')")

        result = find_existing_job_def(agent, "print('world')")
        assert result is None

    def test_returns_none_when_different_agent(self, db0_fixture):  # pylint: disable=unused-argument
        """Returns None when the JobDef belongs to a different agent."""
        agent_a = Agent(role="agent_a", _system_prompt=make_system_prompt("A"), _tools=[])
        agent_b = Agent(role="agent_b", _system_prompt=make_system_prompt("B"), _tools=[])

        _make_tagged_job_def(agent_a, "print('hello')")

        result = find_existing_job_def(agent_b, "print('hello')")
        assert result is None

    def test_handles_none_warmup_code(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Matches when both stored and input warmup_code parse to None."""
        job_def = _make_tagged_job_def(agent, None)

        result = find_existing_job_def(agent, None)
        assert result is job_def

    def test_matches_equivalent_warmup_code_forms(self, db0_fixture, agent):  # pylint: disable=unused-argument
        """Matches when warmup_code is stored parsed but input is the equivalent raw string."""
        _make_tagged_job_def(agent, "x = 1\nprint(x)")

        # The same code passed as a raw string should resolve to the same parsed form
        result = find_existing_job_def(agent, "x = 1\nprint(x)")
        assert result is not None


class TestRunAgenticLoop:
    """Tests for run_agentic_loop."""

    @pytest.mark.asyncio
    async def test_new_job_def_initially_shares_agent_metadata(self, db0_fixture):  # pylint: disable=unused-argument
        """Loop-created JobDefs initially reference the agent metadata dict."""
        agent = Agent(
            role="loop_agent",
            _system_prompt=make_system_prompt("Test"),
            _metadata={"MODEL": "test-model", "TEMPERATURE": "0.3"},
            _tools=[],
        )

        await run_agentic_loop(
            agent=agent,
            warmup_code=None,
            task_queue_size_func=lambda: 0,
            queue_prefixes=[db0.get_current_prefix().name],
            auto_terminate=True,
        )

        job_def = find_existing_job_def(agent, None)
        assert job_def is not None
        assert job_def.metadata is agent._metadata  # pylint: disable=protected-access
