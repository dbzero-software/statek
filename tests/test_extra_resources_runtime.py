"""Job-local capability donors resolve from configuration, not conversations."""
# pylint: disable=protected-access,no-member,redefined-outer-name

from collections.abc import Callable

import dbzero as db0
import pytest

from statek.agents.agent import Agent, SupervisedAgent
from statek.executors.job import Job, JobDef
from statek.extra_resources import parse_extra_resources


@pytest.fixture
def resource_agents(agent_factory: Callable[..., Agent]) -> dict[str, Agent]:
    """Configure donors in a different order from their activation order."""
    return {
        role: agent_factory(role=role)
        for role in ("receiver", "sounding_board", "schedule_assistant", "preferences_assistant")
    }


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
