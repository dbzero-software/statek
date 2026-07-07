"""Tests for post-processor primitives and JobDef identity."""

# pylint: disable=unused-argument,no-member,too-few-public-methods

import dbzero as db0
import pytest

from statek.chat_history import ChatRole, ContentSource, format_chat_history_item
from statek.agents.agent import SupervisedAgent
from statek.executors.chat_log_item import LLM_LogItem, PostProcessedItem
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.post_processor import FinalCheck, PostProcessor, post_processor_identity
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


@db0.memo
class StaticReturnPostProcessor(PostProcessor):
    """Post-processor returning a fixed value for result-normalization tests."""

    def __init__(self, result):
        self.result = result

    def process(self, llm_step: LLM_StepData, job):
        """Return the configured result."""
        del llm_step, job
        return self.result


@db0.memo
class ReplacementPostProcessor(PostProcessor):
    """Post-processor returning a replacement LLM step."""

    def __init__(self, text: str):
        self.text = text

    def process(self, llm_step: LLM_StepData, job):
        """Return a replacement step."""
        del llm_step, job
        return LLM_StepData(text=self.text, call_requests=None)


@db0.memo
class ClonePostProcessor(PostProcessor):
    """Post-processor returning a value-equal replacement LLM step."""

    def process(self, llm_step: LLM_StepData, job):
        """Return a new step with the same values as the input step."""
        del job
        return LLM_StepData(text=llm_step.text, call_requests=llm_step.call_requests)


@db0.memo
class TupleMessageAndStepPostProcessor(PostProcessor):
    """Post-processor returning a system message and replacement step."""

    def __init__(self, message: str, text: str):
        self.message = message
        self.text = text

    def process(self, llm_step: LLM_StepData, job):
        """Return a message and replacement step."""
        del llm_step, job
        return self.message, LLM_StepData(text=self.text, call_requests=None)


@db0.memo
class MultipleStepPostProcessor(PostProcessor):
    """Post-processor returning two LLM steps."""

    def process(self, llm_step: LLM_StepData, job):
        """Return an invalid multiple-step tuple."""
        del llm_step, job
        return (
            LLM_StepData(text="First", call_requests=None),
            LLM_StepData(text="Second", call_requests=None),
        )


@db0.memo
class TupleMessagesPostProcessor(PostProcessor):
    """Post-processor returning multiple system messages."""

    def process(self, llm_step: LLM_StepData, job):
        """Return two messages without an LLM step."""
        del llm_step, job
        return "Review once.", "Review twice."


def _supervised_agent(role: str = "post_processor_agent") -> SupervisedAgent:
    """Create a supervised agent with required model metadata."""
    return SupervisedAgent(
        role=role,
        _system_prompt=make_system_prompt("Test agent"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )


def _job() -> Job:
    """Create a job for post-processor method tests."""
    agent = _supervised_agent()
    job = Job(
        job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}),
        job_status=JobStatus.READY,
    )
    job.job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
    return job


def _final_step(text: str = "Draft final answer") -> LLM_StepData:
    """Return a final DIRECT step for FinalCheck tests."""
    return LLM_StepData(text=text, call_requests=None)


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


def test_handle_post_processor_passes_through_llm_step(db0_fixture):
    """An LLM_StepData result continues without activating the processor."""
    job = _job()
    llm_step = LLM_StepData(text="Draft answer", call_requests=None)
    processor = IdentityPostProcessor("identity")

    result = job.handle_post_processor(processor, llm_step)

    assert result == (llm_step, False)
    assert job.chat_log == []


def test_handle_post_processor_replaces_llm_step(db0_fixture):
    """A replacement LLM_StepData result becomes the next chained step."""
    job = _job()
    llm_step = LLM_StepData(text="Draft answer", call_requests=None)
    processor = ReplacementPostProcessor("Better answer")

    result = job.handle_post_processor(processor, llm_step)

    assert result == (LLM_StepData(text="Better answer", call_requests=None), True)
    assert job.chat_log == []


def test_handle_post_processor_value_equal_replacement_activates(db0_fixture):
    """A new LLM_StepData instance activates even when it has equal values."""
    job = _job()
    llm_step = LLM_StepData(text="Draft answer", call_requests=None)
    processor = ClonePostProcessor()

    processed_step, activated = job.handle_post_processor(processor, llm_step)

    assert processed_step == llm_step
    assert processed_step is not llm_step
    assert activated is True
    assert job.chat_log == []


def test_handle_post_processor_message_only_suppresses_llm_step(db0_fixture):
    """A string result records a system message and drops the current LLM step."""
    job = _job()
    job.py_env.console = ["existing output"]
    processor = StaticReturnPostProcessor("Review the draft answer.")

    result = job.handle_post_processor(
        processor,
        LLM_StepData(text="Draft answer", call_requests=None),
    )

    assert result == (None, True)
    assert len(job.chat_log) == 1
    assert isinstance(job.chat_log[0], PostProcessedItem)
    assert job.chat_log[0].post_processor is processor
    assert job.chat_log[0].message == "Review the draft answer."
    assert job.chat_log[0].console_pos == 1


def test_handle_post_processor_tuple_message_and_step(db0_fixture):
    """Tuple results may combine a system message with one LLM step."""
    job = _job()
    processor = TupleMessageAndStepPostProcessor("Review the draft answer.", "Better answer")

    result = job.handle_post_processor(
        processor,
        LLM_StepData(text="Draft answer", call_requests=None),
    )

    assert result == (LLM_StepData(text="Better answer", call_requests=None), True)
    assert len(job.chat_log) == 1
    assert isinstance(job.chat_log[0], PostProcessedItem)
    assert job.chat_log[0].message == "Review the draft answer."


def test_handle_post_processor_tuple_message_without_step_suppresses(db0_fixture):
    """Tuple results without LLM_StepData suppress the current LLM step."""
    job = _job()
    processor = TupleMessagesPostProcessor()

    result = job.handle_post_processor(
        processor,
        LLM_StepData(text="Draft answer", call_requests=None),
    )

    assert result == (None, True)
    assert [item.message for item in job.chat_log] == ["Review once.", "Review twice."]


def test_handle_post_processor_rejects_invalid_return_value(db0_fixture):
    """Unsupported processor return values fail clearly."""
    job = _job()
    processor = StaticReturnPostProcessor(42)

    with pytest.raises(TypeError, match="Unsupported post-processor return value"):
        job.handle_post_processor(
            processor,
            LLM_StepData(text="Draft answer", call_requests=None),
        )


def test_handle_post_processor_rejects_multiple_llm_steps(db0_fixture):
    """A tuple cannot contain more than one replacement LLM step."""
    job = _job()
    processor = MultipleStepPostProcessor()

    with pytest.raises(ValueError, match="at most one LLM_StepData"):
        job.handle_post_processor(
            processor,
            LLM_StepData(text="Draft answer", call_requests=None),
        )


def test_final_check_activates_for_early_final_step(db0_fixture):
    """FinalCheck suppresses an early final answer and asks for reflection."""
    job = _job()
    processor = FinalCheck(max_llm_actions=3)
    llm_step = _final_step("The answer is 42.")

    result = processor.process(llm_step, job)

    assert isinstance(result, str)
    assert "The answer is 42." in result
    assert "correct, complete, and directly answers the user" in result
    assert "Revise if needed; otherwise return it unchanged." in result


def test_final_check_inactive_at_action_threshold(db0_fixture):
    """FinalCheck passes through once the job has enough LLM actions."""
    job = _job()
    job.chat_log.extend([
        LLM_LogItem(console_pos=0, llm_resp="First answer"),
        LLM_LogItem(console_pos=1, llm_resp="Second answer"),
        LLM_LogItem(console_pos=2, llm_resp="Third answer"),
    ])
    processor = FinalCheck(max_llm_actions=3)
    llm_step = _final_step()

    assert processor.process(llm_step, job) is llm_step


def test_final_check_inactive_for_non_final_step(db0_fixture):
    """FinalCheck does not fire for steps that still need tool execution."""
    job = _job()
    processor = FinalCheck(max_llm_actions=3)
    llm_step = LLM_StepData(
        text="I need data.",
        call_requests=[{"name": "python_cli"}],
    )

    assert processor.process(llm_step, job) is llm_step


def test_final_check_does_not_repeat_for_same_instance(db0_fixture):
    """A FinalCheck instance fires at most once for a job."""
    job = _job()
    processor = FinalCheck(max_llm_actions=3)
    llm_step = _final_step()

    assert job.handle_post_processor(processor, llm_step) == (None, True)
    assert job.handle_post_processor(processor, llm_step) == (llm_step, False)


def test_final_check_separate_instance_can_fire_independently(db0_fixture):
    """Fired-once detection is per exact FinalCheck instance."""
    job = _job()
    first = FinalCheck(max_llm_actions=3)
    second = FinalCheck(max_llm_actions=4)
    llm_step = _final_step()

    assert job.handle_post_processor(first, llm_step) == (None, True)
    assert job.handle_post_processor(second, llm_step) == (None, True)
    assert [item.post_processor for item in job.chat_log] == [first, second]


def test_final_check_md_dialog_internal_text_activation_profile(db0_fixture):
    """MD_DIALOG header-less final text can activate while prior action count is zero."""
    job = _job()
    job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
    job.chat_log.append(LLM_LogItem(console_pos=0, llm_resp="internal reasoning"))
    processor = FinalCheck(max_llm_actions=1)

    result = processor.process(LLM_StepData(text="internal final", call_requests=None), job)

    assert job.count_llm_actions() == 0
    assert isinstance(result, str)
