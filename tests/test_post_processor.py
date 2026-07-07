"""Tests for post-processor primitives and JobDef identity."""

# pylint: disable=unused-argument,no-member,too-few-public-methods

import dbzero as db0

from statek.chat_history import ChatRole, ContentSource, format_chat_history_item
from statek.agents.agent import SupervisedAgent
from statek.executors.chat_log_item import LLM_LogItem, PostProcessedItem
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.post_processor import PostProcessor, post_processor_identity
from statek.executors.utils import find_existing_job_def
from statek.llm_api import LLM_StepData
from statek.prompt_config import make_system_prompt
from statek.settings import ChatStyle
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


def test_post_processed_item_yields_system_chat_history_item(db0_fixture):
    """Post-processor messages are included as system chat history."""
    agent = _supervised_agent()
    job = Job(
        job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}),
        job_status=JobStatus.READY,
    )
    processor = IdentityPostProcessor("check")

    job.chat_log.append(PostProcessedItem(
        console_pos=0,
        post_processor=processor,
        message="Review the draft answer.",
    ))

    history = list(job.get_next_request()["chat_history"])

    assert len(history) == 1
    assert history[0].role == ChatRole.SYSTEM
    assert history[0].content_src == ContentSource.SYSTEM
    assert history[0].content == "Review the draft answer."


def test_post_processed_item_formats_as_system_payload(db0_fixture):
    """The existing formatter turns post-processed history into a system payload."""
    agent = _supervised_agent()
    job = Job(
        job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}),
        job_status=JobStatus.READY,
    )

    job.chat_log.append(PostProcessedItem(
        console_pos=0,
        post_processor=IdentityPostProcessor("check"),
        message="Review the draft answer.",
    ))
    history = list(job.get_next_request()["chat_history"])

    assert format_chat_history_item(history[0], ChatStyle.DIRECT) == {
        "role": "system",
        "content": "Review the draft answer.",
    }


def test_post_processed_item_delimits_previous_console_output(db0_fixture):
    """Post-processed items use console_pos when slicing prior console output."""
    agent = _supervised_agent()
    job = Job(
        job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}),
        job_status=JobStatus.READY,
    )
    job.py_env.console = ["Out 1", "Out 2", "Out 3"]
    job.chat_log.append(LLM_LogItem(console_pos=0, llm_resp="First response"))
    job.chat_log.append(PostProcessedItem(
        console_pos=2,
        post_processor=IdentityPostProcessor("check"),
        message="Review the draft answer.",
    ))

    history = list(job.get_next_request()["chat_history"])

    assert [item.role for item in history] == [
        ChatRole.ASSISTANT,
        ChatRole.USER,
        ChatRole.SYSTEM,
    ]
    assert history[1].content == "Out 1\nOut 2"
    assert history[2].content == "Review the draft answer."
