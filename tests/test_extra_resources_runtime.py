"""Job-local capability donors resolve from configuration, not conversations."""
# pylint: disable=protected-access,no-member,redefined-outer-name

from collections.abc import Callable
from unittest.mock import patch

import dbzero as db0
import pytest

from statek.agents.agent import (
    Agent,
    SupervisedAgent,
    list_of_documents as list_documents_wrapper,
    list_of_examples as list_examples_wrapper,
    show_document as show_document_wrapper,
    show_example as show_example_wrapper,
)
from statek.agents.dialog_agent import DialogAgent
from statek.agents.researcher import Researcher
from statek.executors.job import Job, JobDef
from statek.executors.utils import exec_cli_step, exec_step, exec_tool
from statek.extra_resources import parse_extra_resources
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats, select_request_tools
from statek.prompt_config import make_system_prompt
from statek.system import tool
from statek.utils import CallSpec, get_current_agent, get_current_job


DIALOG_MESSAGES: list[tuple[str, object]] = []


def record_dialog_message(body: str, media=None) -> None:
    """Record dialog output without accepting framework context keywords."""
    DIALOG_MESSAGES.append((body, media))


def research_message(message: str) -> str:
    """Return a researcher response without accepting framework context keywords."""
    return f"response:{message}"


def context_observing_message(message: str) -> str:
    """Read the active job without accepting framework context keywords."""
    return f"{get_current_job().job_def.agent.role}:{message}"


@tool
def receiver_action(value: str, **kwargs) -> str:  # pylint: disable=unused-argument
    """Return a value from the receiving agent.

    Args:
        value: Value to return.
    """
    return f"receiver:{value}"


@tool
def preference_action(value: str, **kwargs) -> str:  # pylint: disable=unused-argument
    """Return a preference action with the active execution identity.

    Args:
        value: Value to return with the active identity.
    """
    agent = get_current_agent()
    job = get_current_job()
    return f"{agent.role}:{job.job_def.agent.role}:{value}"


@tool
def schedule_action(value: str, **kwargs) -> str:  # pylint: disable=unused-argument
    """Return a schedule action value.

    Args:
        value: Value to return.
    """
    return f"schedule:{value}"


@tool
def _preference_internal(value: str, **kwargs) -> str:  # pylint: disable=unused-argument
    """Return an internal preference helper value."""
    return f"internal:{value}"


@tool(hidden=True)
def hidden_preference_action(**kwargs) -> str:  # pylint: disable=unused-argument
    """Return a value that must remain unavailable to formal tool calls."""
    return "hidden preference"


@pytest.fixture
def resource_agents(agent_factory: Callable[..., Agent]) -> dict[str, Agent]:
    """Configure donors in a different order from their activation order."""
    return {
        role: agent_factory(role=role)
        for role in ("receiver", "sounding_board", "schedule_assistant", "preferences_assistant")
    }


@pytest.fixture
def resource_tool_job(resource_agents: dict[str, Agent]) -> Job:
    """Create a low-difficulty job with tool-bearing configured donors."""
    receiver = resource_agents["receiver"]
    receiver._system_prompt = make_system_prompt("Receiver rules only.\n{tools}")
    receiver._tools.append(receiver_action)
    preferences = resource_agents["preferences_assistant"]
    preferences._system_prompt = make_system_prompt("Donor rules must not be merged.")
    preferences._tools.append(preference_action)
    preferences._internal_tools.append(_preference_internal)
    resource_agents["schedule_assistant"]._tools.append(schedule_action)
    return Job(job_def=JobDef(
        agent=receiver,
        metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": "L"},
        extra_resources=parse_extra_resources(
            "L:preferences_assistant,M:schedule_assistant,H:sounding_board"
        ),
    ))


@pytest.mark.parametrize("difficulty,count", [("L", 2), ("M", 3), ("H", 4)])
def test_resource_agents_inherit_in_declaration_order(
    resource_agents: dict[str, Agent], difficulty: str, count: int,
) -> None:
    """Each difficulty includes cumulative donors in declaration order without creating jobs."""
    definition = JobDef(
        agent=resource_agents["receiver"],
        metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": difficulty},
        extra_resources=parse_extra_resources(
            "L:preferences_assistant,M:schedule_assistant,H:sounding_board"
        ),
    )
    job = Job(job_def=definition)
    expected = [resource_agents[role] for role in (
        "receiver", "preferences_assistant", "schedule_assistant", "sounding_board"
    )]

    assert job.get_resource_agents() == expected[:count]
    assert len(db0.find(Job)) == 1


def test_resource_agents_without_declaration_return_receiver(
    resource_agents: dict[str, Agent],
) -> None:
    """A job without extra resources uses only its receiving agent's configuration."""
    receiver = resource_agents["receiver"]
    job = Job(job_def=JobDef(agent=receiver))

    assert job.get_resource_agents() == [receiver]


def test_resource_agents_deduplicate_receiver_and_donors(
    resource_agents: dict[str, Agent],
) -> None:
    """Repeated donor and self-references produce only one entry per configured role."""
    receiver = resource_agents["receiver"]
    job = Job(job_def=JobDef(
        agent=receiver,
        metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": "H"},
        extra_resources=parse_extra_resources(
            "L:receiver,preferences_assistant,M:receiver,preferences_assistant"
        ),
    ))

    assert job.get_resource_agents() == [receiver, resource_agents["preferences_assistant"]]


def test_resource_agents_do_not_expand_donor_declarations(
    resource_agents: dict[str, Agent],
) -> None:
    """Only explicitly named donors are included, not their own extra resources."""
    donor = resource_agents["preferences_assistant"]
    donor.update_metadata({"MODEL": "test-model", "EXTRA_RESOURCES": "L:missing"})
    receiver = resource_agents["receiver"]
    job = Job(job_def=JobDef(
        agent=receiver, extra_resources=parse_extra_resources("L:preferences_assistant"),
    ))

    assert job.get_resource_agents() == [receiver, donor]


def test_resource_agents_follow_job_escalation_without_shared_mutation(
    resource_agents: dict[str, Agent],
) -> None:
    """Panic expands one job's resources without changing other jobs or shared configuration."""
    receiver = resource_agents["receiver"]
    definition = JobDef(
        agent=receiver,
        metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": "L"},
        extra_resources=parse_extra_resources("M:schedule_assistant,H:sounding_board"),
    )
    first = Job(job_def=definition)
    second = Job(job_def=definition)

    assert first.get_resource_agents() == [receiver]
    first.panic()
    assert first.get_resource_agents() == [receiver, resource_agents["schedule_assistant"]]
    first.panic()
    expanded = first.get_resource_agents()
    assert expanded == [
        receiver, resource_agents["schedule_assistant"], resource_agents["sounding_board"]
    ]
    expanded.clear()

    assert len(first.get_resource_agents()) == 3
    assert second.get_resource_agents() == [receiver]
    assert definition.extra_resources == (None, ["schedule_assistant"], ["sounding_board"])
    assert receiver._tools == []
    assert receiver._metadata == {"MODEL": "test-model"}


def test_resource_agents_ignore_running_donor_jobs(resource_agents: dict[str, Agent]) -> None:
    """Creating donor jobs does not affect which configured agent supplies resources."""
    receiver = resource_agents["receiver"]
    donor = resource_agents["schedule_assistant"]
    job = Job(job_def=JobDef(
        agent=receiver, extra_resources=parse_extra_resources("L:schedule_assistant"),
    ))
    expected = job.get_resource_agents()
    donor_definition = JobDef(agent=donor)
    for _ in range(5):
        Job(job_def=donor_definition)

    assert job.get_resource_agents() == expected == [receiver, donor]
    assert len(db0.find(Job)) == 6


def test_resource_agents_resolve_configured_subclasses(
    resource_agents: dict[str, Agent],
) -> None:
    """Persisted Agent subclasses can supply resources just like base Agent configurations."""
    receiver = resource_agents["receiver"]
    donor = SupervisedAgent(
        role="supervised_donor", _system_prompt=None, _tools=[], _metadata={"MODEL": "test-model"},
    )
    job = Job(job_def=JobDef(
        agent=receiver, extra_resources=parse_extra_resources("L:supervised_donor"),
    ))

    assert job.get_resource_agents() == [receiver, donor]


def test_resource_agents_missing_definition_fails_only_when_activated(
    resource_agents: dict[str, Agent],
) -> None:
    """A missing donor raises an error only after its configured difficulty is reached."""
    receiver = resource_agents["receiver"]
    job = Job(job_def=JobDef(
        agent=receiver,
        metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": "L"},
        extra_resources=parse_extra_resources("M:missing_agent"),
    ))

    assert job.get_resource_agents() == [receiver]
    job.panic()
    with pytest.raises(ValueError, match="EXTRA_RESOURCES.*missing_agent"):
        job.get_resource_agents()


def test_resource_agents_use_receiver_prefix_not_current_prefix(
    resource_agents: dict[str, Agent],
) -> None:
    """Lookup uses the receiver's prefix without changing the caller's current prefix."""
    receiver = resource_agents["receiver"]
    donor = resource_agents["schedule_assistant"]
    job = Job(job_def=JobDef(
        agent=receiver, extra_resources=parse_extra_resources("L:schedule_assistant"),
    ))
    db0.open("unrelated_config", "rw")
    Agent(role="schedule_assistant", _system_prompt=None, _tools=[])

    assert job.get_resource_agents() == [receiver, donor]
    assert db0.get_current_prefix().name == "unrelated_config"


def test_resource_agents_do_not_resolve_missing_role_from_another_prefix(
    resource_agents: dict[str, Agent],
) -> None:
    """A donor in another prefix cannot satisfy a missing local configuration."""
    job = Job(job_def=JobDef(
        agent=resource_agents["receiver"], extra_resources=parse_extra_resources("L:other_donor"),
    ))
    db0.open("unrelated_config", "rw")
    Agent(role="other_donor", _system_prompt=None, _tools=[])

    with pytest.raises(ValueError, match="EXTRA_RESOURCES.*other_donor"):
        job.get_resource_agents()


def test_resource_tools_expand_request_and_receiver_prompt(resource_tool_job: Job) -> None:
    """Active donor tools appear in requests and receiver tool placeholders only."""
    request = resource_tool_job.get_next_request()
    names = [tool_fn.__name__ for tool_fn in request["available_tools"]]

    assert names[:2] == ["receiver_action", "preference_action"]
    assert "_preference_internal" not in names
    assert "schedule_action" not in names
    assert "receiver_action" in request["system_prompt"]
    assert "preference_action" in request["system_prompt"]
    assert "_preference_internal" not in request["system_prompt"]
    assert "schedule_action" not in request["system_prompt"]
    assert "Donor rules must not be merged." not in request["system_prompt"]


def test_resource_tools_are_available_to_formal_provider_scope(resource_tool_job: Job) -> None:
    """An inherited action can be explicitly selected for a formal provider tool call."""
    request = resource_tool_job.get_next_request()
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "ALL"},
        available_tools=request["available_tools"],
    )
    names = [tool_fn.__name__ for tool_fn in selected]

    assert "preference_action" in names
    assert "_preference_internal" not in names


def test_resource_tools_follow_panic_without_mutating_sibling_job(
    resource_tool_job: Job,
) -> None:
    """Panic adds the next donor's tools only to the escalating job."""
    sibling = Job(job_def=resource_tool_job.job_def)

    resource_tool_job.panic()

    expanded = [tool_fn.__name__ for tool_fn in resource_tool_job.get_resource_tools()]
    unchanged = [tool_fn.__name__ for tool_fn in sibling.get_resource_tools()]
    assert "schedule_action" in expanded
    assert "schedule_action" not in unchanged
    assert resource_tool_job.job_def.agent._tools == [receiver_action]


@pytest.mark.asyncio
async def test_exec_step_calls_inherited_action_in_receiver_context(
    resource_tool_job: Job,
) -> None:
    """Python execution can call a donor action while retaining receiver job identity."""
    await exec_step("result = preference_action('saved')", resource_tool_job)

    assert resource_tool_job.py_env.local_state["result"] == "receiver:receiver:saved"


@pytest.mark.asyncio
async def test_exec_step_refreshes_inherited_tools_after_panic_in_same_block(
    resource_tool_job: Job,
) -> None:
    """A tool activated by panic is callable by the following statement."""
    await exec_step(
        "panic()\nresult = schedule_action('expanded')",
        resource_tool_job,
    )

    assert resource_tool_job.py_env.local_state["result"] == "schedule:expanded"


@pytest.mark.asyncio
async def test_exec_tool_calls_inherited_action_in_receiver_context(
    resource_tool_job: Job,
) -> None:
    """Formal tool calls resolve donor actions against the receiving execution context."""
    result, error = await exec_tool(
        CallSpec(
            id="inherited-action", func_name="preference_action",
            args=[], kwargs={"value": "saved"},
        ),
        resource_tool_job,
    )

    assert error is None
    assert result == "receiver:receiver:saved"


@pytest.mark.asyncio
async def test_exec_tool_rejects_inherited_hidden_action(
    resource_tool_job: Job, resource_agents: dict[str, Agent],
) -> None:
    """Hidden donor tools remain unavailable after capability expansion."""
    resource_agents["preferences_assistant"]._tools.append(hidden_preference_action)

    result, error = await exec_tool(
        CallSpec(id="hidden-donor", func_name="hidden_preference_action", args=[], kwargs={}),
        resource_tool_job,
    )

    assert error is not None
    assert "hidden_preference_action" in result


@pytest.mark.asyncio
async def test_exec_step_calls_dialog_answer_without_context_keyword(
    db0_fixture,  # pylint: disable=unused-argument
) -> None:
    """Name-registered dialog tools retain their original callable signature."""
    DIALOG_MESSAGES.clear()
    agent = DialogAgent(
        send_message=record_dialog_message,
        _metadata={"MODEL": "test-model"},
    )
    job = Job(job_def=agent.create_job_def())

    running = await exec_step("answer('finished')", job)

    assert running is False
    assert DIALOG_MESSAGES == [("finished", None)]
    assert job.py_env.exit_status == "Success"


@pytest.mark.asyncio
async def test_exec_cli_step_calls_researcher_tool_without_context_keyword(
    db0_fixture,  # pylint: disable=unused-argument
) -> None:
    """Python CLI can call a name-registered tool that has no framework kwargs."""
    agent = Researcher(send_message=research_message)
    job = Job(job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}))

    await exec_cli_step("result = ask('question')", job, lambda _output: None)

    assert job.py_env.local_state["result"] == "response:question"


@pytest.mark.asyncio
async def test_exec_tool_calls_researcher_tool_without_context_keyword(
    db0_fixture,  # pylint: disable=unused-argument
) -> None:
    """Formal calls preserve signatures of name-registered context tools."""
    agent = Researcher(send_message=research_message)
    job = Job(job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}))

    result, error = await exec_tool(
        CallSpec(id="research", func_name="ask", args=[], kwargs={"question": "question"}),
        job,
    )

    assert error is None
    assert result == "response:question"


@pytest.mark.asyncio
async def test_non_injectable_context_tool_observes_receiver_job(
    db0_fixture,  # pylint: disable=unused-argument
) -> None:
    """Signature compatibility does not remove the receiver's Statek context."""
    agent = Researcher(send_message=context_observing_message, role="receiver")
    job = Job(job_def=JobDef(agent=agent, metadata={"MODEL": "test-model"}))

    result, error = await exec_tool(
        CallSpec(id="context", func_name="ask", args=[], kwargs={"question": "question"}),
        job,
    )

    assert error is None
    assert result == "receiver:question"


@pytest.mark.asyncio
async def test_exec_step_calls_inherited_dynamic_and_internal_tools(
    resource_tool_job: Job, resource_agents: dict[str, Agent],
) -> None:
    """Named and internal donor tools execute but internal tools stay out of the prompt."""
    donor = resource_agents["preferences_assistant"]

    def dynamic_preference(value: str, **kwargs) -> str:  # pylint: disable=unused-argument
        """Return a dynamic preference result.

        Args:
            value: Value to return with the active agent role.
        """
        return f"dynamic:{get_current_agent().role}:{value}"

    donor.context["dynamic_preference"] = dynamic_preference
    donor.append_tool("dynamic_preference")

    await exec_step(
        "dynamic_result = dynamic_preference('ok')\ninternal_result = _preference_internal('ok')",
        resource_tool_job,
    )

    assert resource_tool_job.py_env.local_state["dynamic_result"] == "dynamic:receiver:ok"
    assert resource_tool_job.py_env.local_state["internal_result"] == "internal:ok"
    prompt = resource_tool_job.system_prompt()
    assert "dynamic_preference" in prompt
    assert "_preference_internal" not in prompt


@pytest.mark.asyncio
async def test_receiver_tool_name_takes_precedence_over_donor_tool(
    resource_tool_job: Job, resource_agents: dict[str, Agent],
) -> None:
    """Existing first-name-wins semantics preserve the receiver's same-named action."""
    donor = resource_agents["preferences_assistant"]

    def donor_receiver_action(value: str, **kwargs) -> str:  # pylint: disable=unused-argument
        return f"donor:{value}"

    donor_receiver_action.__name__ = "receiver_action"
    donor.context["receiver_action"] = donor_receiver_action
    donor.append_tool("receiver_action")

    tools = resource_tool_job.get_resource_tools()
    assert [tool_fn.__name__ for tool_fn in tools].count("receiver_action") == 1
    await exec_step("collision_result = receiver_action('ok')", resource_tool_job)
    assert resource_tool_job.py_env.local_state["collision_result"] == "receiver:ok"


def test_historical_requests_keep_resources_from_their_original_difficulty(
    resource_tool_job: Job,
) -> None:
    """Later escalation must not add tools to an earlier request preview."""
    low_difficulty = resource_tool_job.get_current_difficulty()
    low_request = resource_tool_job.get_next_request()
    resource_tool_job.append_chat_log(
        low_request,
        LLM_Response(
            step_data=LLM_StepData(text="low response", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ),
        request_difficulty=low_difficulty,
    )
    resource_tool_job.panic()
    medium_difficulty = resource_tool_job.get_current_difficulty()
    medium_request = resource_tool_job.get_next_request()
    resource_tool_job.append_chat_log(
        medium_request,
        LLM_Response(
            step_data=LLM_StepData(text="medium response", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ),
        request_difficulty=medium_difficulty,
    )
    resource_tool_job.panic()

    low_tools = [
        tool_fn.__name__ for tool_fn in resource_tool_job.get_request_data(0)["available_tools"]
    ]
    medium_tools = [
        tool_fn.__name__ for tool_fn in resource_tool_job.get_request_data(1)["available_tools"]
    ]

    assert "preference_action" in low_tools
    assert "schedule_action" not in low_tools
    assert "schedule_action" in medium_tools


def test_resource_wrappers_pass_active_agent_roles(resource_tool_job: Job) -> None:
    """Example and document wrappers pass receiver-first active roles to their helpers."""
    expected_roles = ["receiver", "preferences_assistant"]
    with (
        patch("statek.agents.agent.get_current_agent", return_value=resource_tool_job.agent),
        patch("statek.agents.agent.get_current_job", return_value=resource_tool_job),
        patch("statek.agents.list_of_examples.list_of_examples") as list_examples,
        patch("statek.agents.list_of_examples.show_example") as show_example,
        patch("statek.agents.list_of_documents.list_of_documents") as list_documents,
        patch("statek.agents.list_of_documents.show_document") as show_document,
    ):
        list_examples_wrapper(start_index=2, limit=3)
        show_example_wrapper(example_id=4)
        list_documents_wrapper(topic="Guide", start_index=5, limit=6)
        show_document_wrapper(key=7, topic="Guide", start_from=8, limit=9)

    list_examples.assert_called_once_with(expected_roles, 2, 3)
    show_example.assert_called_once_with(expected_roles, 4)
    list_documents.assert_called_once_with(
        expected_roles, list_documents.call_args.args[1], topic="Guide", start_index=5, limit=6,
    )
    show_document.assert_called_once_with(
        expected_roles, show_document.call_args.args[1],
        key=7, topic="Guide", start_from=8, limit=9,
    )
