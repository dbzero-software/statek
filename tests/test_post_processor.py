"""Tests for post-processor primitives and JobDef identity."""

# pylint: disable=unused-argument,no-member,too-few-public-methods

import dbzero as db0

from statek.agents.agent import SupervisedAgent
from statek.executors.job import JobDef
from statek.executors.post_processor import PostProcessor, post_processor_identity
from statek.executors.utils import find_existing_job_def
from statek.llm_api import LLM_StepData
from statek.prompt_config import make_system_prompt
from statek.task import create_new_job


@db0.memo
class IdentityPostProcessor(PostProcessor):
    """Minimal concrete post-processor for identity tests."""

    def __init__(self, label: str = "default"):
        self.label = label

    def process(self, llm_step: LLM_StepData, job):
        """Return the step unchanged."""
        del job
        return llm_step


def _supervised_agent(role: str = "post_processor_agent") -> SupervisedAgent:
    """Create a supervised agent with required model metadata."""
    return SupervisedAgent(
        role=role,
        _system_prompt=make_system_prompt("Test agent"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )


def test_concrete_post_processor_has_stable_value_identity(db0_fixture):
    """Concrete processors compare by class and stored state."""
    first = IdentityPostProcessor("same")
    second = IdentityPostProcessor("same")

    assert post_processor_identity(first) == post_processor_identity(second)


def test_create_job_def_accepts_post_processing_instance(db0_fixture):
    """SupervisedAgent.create_job_def stores resolved post-processor instances."""
    agent = _supervised_agent()
    processor = IdentityPostProcessor("check")

    job_def = agent.create_job_def(post_processing=processor)

    assert job_def.post_processing == [processor]


def test_find_existing_job_def_matches_single_and_list_post_processing(db0_fixture):
    """A single processor and a one-item list are the same JobDef identity."""
    agent = _supervised_agent()
    processor = IdentityPostProcessor("check")
    job_def = JobDef(
        agent=agent,
        metadata={"MODEL": "test-model"},
        job_params=None,
        warmup_code=None,
        post_processing=processor,
    )

    found = find_existing_job_def(
        agent,
        None,
        model=job_def.model,
        job_params=None,
        locale=None,
        chat_style=None,
        post_processing=[processor],
    )

    assert found is job_def


def test_create_new_job_reuses_matching_post_processing_job_def(db0_fixture):
    """Repeated jobs with equivalent post-processing reuse one JobDef."""
    agent = _supervised_agent()
    first = create_new_job(
        agent,
        post_processing=IdentityPostProcessor("check"),
    )
    second = create_new_job(
        agent,
        post_processing=[IdentityPostProcessor("check")],
    )

    assert second.job_def is first.job_def
    assert len(db0.find(JobDef, db0.as_tag(agent))) == 1


def test_create_new_job_distinguishes_different_post_processing(db0_fixture):
    """Different post-processing configuration creates a distinct JobDef."""
    agent = _supervised_agent()
    base = create_new_job(agent, post_processing=IdentityPostProcessor("first"))
    different = create_new_job(agent, post_processing=IdentityPostProcessor("second"))

    assert different.job_def is not base.job_def
    assert len(db0.find(JobDef, db0.as_tag(agent))) == 2
