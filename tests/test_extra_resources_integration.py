"""Prompt resource storage and definition reuse across job creation paths."""
# pylint: disable=protected-access,no-member

from pathlib import Path

import dbzero as db0
import pytest

from statek.agents.dialog_agent import DialogAgent
from statek.executors.job import Job, JobDef, job_def_identity_tag_for_job_def
from statek.executors.utils import (
    AgentLoopDef, find_existing_job_def, run_agentic_fleet, run_agentic_loop,
)
from statek.extra_resources import parse_extra_resources
from statek.prompt_config import PromptDef, parse_prompt_file, update_prompt_config
from statek.task import create_new_job
from tests.conftest import DB0_DIR


def _send_message(body: str, media: str | None = None) -> None:  # pylint: disable=unused-argument
    """Accept dialog messages without external effects."""


@pytest.mark.parametrize("declaration", ["H:helpdesk", "(H:helpdesk)"])
def test_extra_resources_prompt_to_job_def(
    supervised_agent, tmp_path: Path, declaration: str,
) -> None:
    """Raw declarations remain on the agent, not in job execution metadata."""
    prompt = tmp_path / "test.md"
    prompt.write_text(
        f"# MODEL: test-model\n# TEMPERATURE: 0.3\n# EXTRA_RESOURCES: {declaration}\n"
        "# System Prompt\nYou are an assistant.\n", encoding="utf-8",
    )
    prompt_def = parse_prompt_file(prompt)
    update_prompt_config({"test": prompt_def}, agents=[supervised_agent])
    definition = supervised_agent.create_job_def(user_question="question")
    assert definition.extra_resources == (None, None, ["helpdesk"])
    assert "EXTRA_RESOURCES" not in definition.metadata
    assert definition.metadata["TEMPERATURE"] == "0.3"
    assert supervised_agent._metadata["EXTRA_RESOURCES"] == declaration
    assert definition.job_params == {"user_question": "question"}
    assert "EXTRA_RESOURCES" not in Job(job_def=definition)._build_request_data()["metadata"]


def test_extra_resources_direct_constructor_and_snapshot(supervised_agent) -> None:
    """Direct constructors isolate resource metadata without altering the source."""
    metadata = {"MODEL": "test-model", "EXTRA_RESOURCES": "LM:a,H:a,b"}
    first = JobDef(agent=supervised_agent, metadata=metadata)
    second = JobDef(agent=supervised_agent, metadata=metadata)
    assert first.extra_resources == (["a"], None, ["b"])
    assert "EXTRA_RESOURCES" not in first.metadata
    assert metadata["EXTRA_RESOURCES"] == "LM:a,H:a,b"
    first.extra_resources[0].append("changed")
    assert second.extra_resources == (["a"], None, ["b"])


def test_extra_resources_internal_factory_parameter(supervised_agent) -> None:
    """Already parsed configuration is transported separately from job_params."""
    resources = parse_extra_resources("M:a,H:b")
    definition = supervised_agent.create_job_def(extra_resources=resources, question="hello")
    assert definition.extra_resources == resources
    assert definition.job_params == {"question": "hello"}
    resources[1].append("changed")
    assert definition.extra_resources == (None, ["a"], ["b"])


def test_extra_resources_absent_preserves_metadata_sharing(supervised_agent) -> None:
    definition = supervised_agent.create_job_def()
    assert definition.extra_resources is None
    assert definition.metadata is supervised_agent._metadata


def test_extra_resources_malformed_metadata_fails_before_job_creation(supervised_agent) -> None:
    supervised_agent.update_metadata({"MODEL": "test-model", "EXTRA_RESOURCES": "Q:a"})
    with pytest.raises(ValueError, match="EXTRA_RESOURCES"):
        create_new_job(supervised_agent)
    assert len(db0.find(Job, db0.as_tag(supervised_agent))) == 0


def test_extra_resources_reuse_and_prompt_changes(supervised_agent) -> None:
    supervised_agent.update_metadata({"MODEL": "test-model", "EXTRA_RESOURCES": "LM:a,H:b"})
    first = create_new_job(supervised_agent)
    supervised_agent.update_metadata({"MODEL": "test-model", "EXTRA_RESOURCES": "(L:a),(H:b,b)"})
    assert create_new_job(supervised_agent).job_def is first.job_def
    update_prompt_config({"test": PromptDef(
        system=None, metadata={"MODEL": "test-model", "EXTRA_RESOURCES": "M:a,H:b"},
    )}, agents=[supervised_agent])
    second = create_new_job(supervised_agent)
    assert second.job_def is not first.job_def
    assert first.job_def.extra_resources == (["a"], None, ["b"])
    update_prompt_config({"test": PromptDef(
        system=None, metadata={"MODEL": "test-model"},
    )}, agents=[supervised_agent])
    third = create_new_job(supervised_agent)
    assert third.job_def.extra_resources is None
    assert third.job_def is not first.job_def
    assert len(db0.find(JobDef, db0.as_tag(supervised_agent))) == 3


def test_extra_resources_lookup_filters_and_hash_collision(supervised_agent, monkeypatch) -> None:
    """Exact comparison protects resource identity even with identical short hashes."""
    monkeypatch.setattr("statek.executors.job._job_def_identity_hash", lambda *args: "same")
    first = supervised_agent.create_job_def(extra_resources=parse_extra_resources("L:a"))
    second = supervised_agent.create_job_def(extra_resources=parse_extra_resources("M:a"))
    assert job_def_identity_tag_for_job_def(first) == job_def_identity_tag_for_job_def(second)
    assert find_existing_job_def(
        supervised_agent, None, extra_resources=(None, ["a"], None),
    ) is second
    assert find_existing_job_def(supervised_agent, None, extra_resources=None) is None
    assert find_existing_job_def(supervised_agent, None) in (first, second)
    assert find_existing_job_def(
        supervised_agent, None, model="test-model", job_params=None, locale=None,
        chat_style=None,
    ) in (first, second)
    assert find_existing_job_def(
        supervised_agent, None, model="test-model", job_params=None, locale=None,
        chat_style=None, extra_resources=(None, ["a"], None),
    ) is second
    assert find_existing_job_def(
        supervised_agent, None, model="test-model", job_params=None, locale=None,
        chat_style=None, extra_resources=(["missing"], None, None),
    ) is None


def test_extra_resources_dialog_factory(db0_fixture) -> None:  # pylint: disable=unused-argument
    agent = DialogAgent(
        role="dialog", send_message=_send_message,
        _metadata={"MODEL": "test-model", "EXTRA_RESOURCES": "H:a"},
    )
    first = create_new_job(agent)
    assert first.job_def.extra_resources == (None, None, ["a"])
    assert first.job_def.job_params is None
    assert create_new_job(agent).job_def is first.job_def


def test_extra_resources_persistence_and_ordered_identity(supervised_agent) -> None:
    definition = supervised_agent.create_job_def(extra_resources=parse_extra_resources("L:a,b,H:c"))
    identifier = db0.uuid(definition)
    agent_identifier = db0.uuid(supervised_agent)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    restored = db0.fetch(identifier)
    agent = db0.fetch(agent_identifier)
    assert restored.extra_resources == (["a", "b"], None, ["c"])
    assert find_existing_job_def(agent, None, extra_resources=(["a", "b"], None, ["c"])) is restored
    assert find_existing_job_def(agent, None, extra_resources=(["b", "a"], None, ["c"])) is None


def test_extra_resources_empty_and_redundant_internal_values(supervised_agent) -> None:
    empty = supervised_agent.create_job_def(extra_resources=([], None, []))
    assert empty.extra_resources is None
    assert find_existing_job_def(
        supervised_agent, None, extra_resources=(None, None, None),
    ) is empty
    definition = supervised_agent.create_job_def(
        extra_resources=(["a", "a"], ["a", "b"], ["b"]),
    )
    assert definition.extra_resources == (["a"], ["b"], None)
    assert find_existing_job_def(
        supervised_agent, None, extra_resources=(["a"], ["a", "b"], None),
    ) is definition


@pytest.mark.asyncio
@pytest.mark.parametrize("fleet", [False, True])
async def test_extra_resources_loop_and_fleet(supervised_agent, fleet: bool) -> None:
    supervised_agent.update_metadata({"MODEL": "test-model", "EXTRA_RESOURCES": "H:a"})
    for _ in range(2):
        if fleet:
            await run_agentic_fleet(
                [AgentLoopDef(supervised_agent, None, lambda: 0)],
                queue_prefixes=[db0.get_current_prefix().name], auto_terminate=True,
            )
        else:
            await run_agentic_loop(
                supervised_agent, None, lambda: 0,
                queue_prefixes=[db0.get_current_prefix().name], auto_terminate=True,
            )
    definitions = tuple(db0.find(JobDef, db0.as_tag(supervised_agent)))
    assert len(definitions) == 1
    assert definitions[0].extra_resources == (None, None, ["a"])
    assert "EXTRA_RESOURCES" not in definitions[0].metadata
    assert supervised_agent._metadata["EXTRA_RESOURCES"] == "H:a"
