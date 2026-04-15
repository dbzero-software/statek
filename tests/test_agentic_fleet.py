"""Tests for AgentLoopDef and run_agentic_fleet."""
# pylint: disable=no-member

import pytest
import dbzero as db0

from statek.agents.agent import SupervisedAgent
from statek.executors.job import Job, JobDef
from statek.executors.utils import AgentLoopDef, run_agentic_fleet, find_existing_job_def


def _make_supervised_agent(role="fleet_agent"):
    return SupervisedAgent(
        role=role,
        _system_prompt="Test",
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )


class TestAgentLoopDef:
    """Tests for the AgentLoopDef dataclass."""

    def test_creates_with_required_fields(self, db0_fixture):  # pylint: disable=unused-argument
        """AgentLoopDef can be created with all required fields."""
        agent = _make_supervised_agent()

        def queue_func():
            return 0

        loop_def = AgentLoopDef(
            agent=agent,
            warmup_code="print('hello')",
            task_queue_size_func=queue_func,
        )

        assert loop_def.agent is agent
        assert loop_def.warmup_code == "print('hello')"
        assert loop_def.task_queue_size_func is queue_func

    def test_creates_with_none_warmup_code(self, db0_fixture):  # pylint: disable=unused-argument
        """AgentLoopDef accepts None warmup_code."""
        agent = _make_supervised_agent()
        loop_def = AgentLoopDef(agent=agent, warmup_code=None, task_queue_size_func=lambda: 0)
        assert loop_def.warmup_code is None

    def test_creates_with_sequence_warmup_code(self, db0_fixture):  # pylint: disable=unused-argument
        """AgentLoopDef accepts a list of warmup code blocks."""
        agent = _make_supervised_agent()
        blocks = ["x = 1", "print(x)"]
        loop_def = AgentLoopDef(agent=agent, warmup_code=blocks, task_queue_size_func=lambda: 0)
        assert loop_def.warmup_code == blocks


class TestRunAgenticFleet:
    """Tests for run_agentic_fleet."""

    @pytest.mark.asyncio
    async def test_auto_terminates_with_empty_queues(self, db0_fixture):  # pylint: disable=unused-argument
        """Fleet auto-terminates when all task queues are empty."""
        agent = _make_supervised_agent("fleet_agent_a")
        loop_def = AgentLoopDef(
            agent=agent,
            warmup_code=None,
            task_queue_size_func=lambda: 0,
        )

        await run_agentic_fleet([loop_def], max_concurrency=1, auto_terminate=True)

        # No jobs should have been created
        jobs = list(db0.find(Job, db0.as_tag(agent)))
        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_creates_job_def_for_each_agent(self, db0_fixture):  # pylint: disable=unused-argument
        """Fleet creates a JobDef for each AgentLoopDef."""
        agent_a = _make_supervised_agent("fleet_agent_b")
        agent_b = _make_supervised_agent("fleet_agent_c")

        loop_defs = [
            AgentLoopDef(agent=agent_a, warmup_code=None, task_queue_size_func=lambda: 0),
            AgentLoopDef(agent=agent_b, warmup_code=None, task_queue_size_func=lambda: 0),
        ]

        await run_agentic_fleet(loop_defs, auto_terminate=True)

        assert find_existing_job_def(agent_a, None) is not None
        assert find_existing_job_def(agent_b, None) is not None

    @pytest.mark.asyncio
    async def test_combined_start_jobs_func_creates_jobs_for_all_agents(self, db0_fixture):  # pylint: disable=unused-argument
        """Fleet calls start_jobs_func for all agents on each loop iteration."""
        agent_a = _make_supervised_agent("fleet_agent_d")
        agent_b = _make_supervised_agent("fleet_agent_e")

        calls = []

        def queue_a():
            calls.append("a")
            return 0

        def queue_b():
            calls.append("b")
            return 0

        loop_defs = [
            AgentLoopDef(agent=agent_a, warmup_code=None, task_queue_size_func=queue_a),
            AgentLoopDef(agent=agent_b, warmup_code=None, task_queue_size_func=queue_b),
        ]

        await run_agentic_fleet(loop_defs, max_concurrency=10, auto_terminate=True)

        # Both queue funcs should have been called at least once
        assert "a" in calls
        assert "b" in calls

    @pytest.mark.asyncio
    async def test_reuses_existing_job_def(self, db0_fixture):  # pylint: disable=unused-argument
        """Fleet reuses an existing JobDef when one already matches."""
        agent = _make_supervised_agent("fleet_agent_f")
        loop_def = AgentLoopDef(
            agent=agent,
            warmup_code="x = 1",
            task_queue_size_func=lambda: 0,
        )

        # Run once to create the job_def
        await run_agentic_fleet([loop_def], auto_terminate=True)
        job_def_first = find_existing_job_def(agent, "x = 1")
        assert job_def_first is not None

        # Run again — should reuse, not create a new one
        await run_agentic_fleet([loop_def], auto_terminate=True)
        job_def_second = find_existing_job_def(agent, "x = 1")
        assert job_def_second is job_def_first

        all_job_defs = list(db0.find(JobDef, db0.as_tag(agent)))
        assert len(all_job_defs) == 1
