"""Compaction thresholds are parsed, persisted, reloaded, and matched by value."""
# pylint: disable=no-member,protected-access

import hashlib

import dbzero as db0
import pytest

from statek.executors.job import JobDef, job_def_identity_tag_for_job_def
from statek.executors.utils import find_existing_job_def
from statek.prompt_config import (
    PromptDef, make_system_prompt, parse_prompt_file, update_prompt_config,
)
from statek.task import create_new_job
from tests.conftest import DB0_DIR


@pytest.mark.parametrize("value,expected", [(0.5, 0.5), (1, 1.0), (None, None)])
def test_constructor_threshold(agent, value, expected):
    definition = JobDef(agent=agent, compaction_threshold=value)
    assert definition.compaction_threshold == expected


def test_default_threshold(agent):
    assert JobDef(agent=agent).compaction_threshold == 0.65


@pytest.mark.parametrize("value,expected", [("0.5", 0.5), (" 1 ", 1.0),
                                           ("None", None), (" nOnE ", None)])
def test_metadata_threshold_takes_precedence(agent, value, expected):
    definition = JobDef(agent=agent, compaction_threshold=0.8,
                        metadata={"MODEL": "test-model", "COMPACTION_THRESHOLD": value})
    assert definition.compaction_threshold == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.1", "nan", "inf", "disabled", "", True])
def test_invalid_metadata_threshold(agent, value):
    with pytest.raises(ValueError, match="COMPACTION_THRESHOLD"):
        JobDef(agent=agent, metadata={"MODEL": "test-model", "COMPACTION_THRESHOLD": value})


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_constructor_threshold(agent, value):
    with pytest.raises(ValueError, match="COMPACTION_THRESHOLD"):
        JobDef(agent=agent, compaction_threshold=value)


def test_markdown_threshold_reaches_new_definition(supervised_agent, tmp_path):
    prompt_path = tmp_path / "configured.md"
    prompt_path.write_text(
        "# MODEL: test-model\n# COMPACTION_THRESHOLD: 0.5\n# System Prompt\nConfigured.",
        encoding="utf-8",
    )
    prompt = parse_prompt_file(prompt_path)
    update_prompt_config({supervised_agent.role: prompt}, agents=[supervised_agent])
    assert supervised_agent.create_job_def().compaction_threshold == 0.5


@pytest.mark.parametrize("value,expected", [("0.5", 0.5), ("None", None), (None, 0.65)])
def test_reload_updates_existing_definitions_and_identity(supervised_agent, value, expected):
    definition = JobDef(agent=supervised_agent, compaction_threshold=0.8)
    old_tag = job_def_identity_tag_for_job_def(definition)
    metadata = {"MODEL": "test-model"}
    if value is not None:
        metadata["COMPACTION_THRESHOLD"] = value
    update_prompt_config({supervised_agent.role: PromptDef(None, metadata)},
                         agents=[supervised_agent])
    assert definition.compaction_threshold == expected
    new_tag = job_def_identity_tag_for_job_def(definition)
    assert definition not in list(db0.find(JobDef, old_tag))
    assert definition in list(db0.find(JobDef, new_tag))
    # Reapplying the same configuration is harmless.
    update_prompt_config({supervised_agent.role: PromptDef(None, metadata)},
                         agents=[supervised_agent])
    assert job_def_identity_tag_for_job_def(definition) == new_tag
    assert definition in list(db0.find(JobDef, new_tag))


def test_invalid_reload_does_not_partially_update_agent(supervised_agent):
    definition = JobDef(agent=supervised_agent)
    old_prompt = supervised_agent._system_prompt
    old_metadata = dict(supervised_agent._metadata)
    old_tag = job_def_identity_tag_for_job_def(definition)
    prompt = PromptDef(make_system_prompt("Changed"),
                       {"MODEL": "changed", "COMPACTION_THRESHOLD": "bad"})
    with pytest.raises(ValueError, match="COMPACTION_THRESHOLD"):
        update_prompt_config({supervised_agent.role: prompt}, agents=[supervised_agent])
    assert supervised_agent._system_prompt == old_prompt
    assert dict(supervised_agent._metadata) == old_metadata
    assert definition.compaction_threshold == 0.65
    assert job_def_identity_tag_for_job_def(definition) == old_tag
    assert definition in list(db0.find(JobDef, old_tag))


def test_reuse_requires_the_same_threshold(supervised_agent):
    first = create_new_job(supervised_agent)
    supervised_agent.update_metadata({"MODEL": "test-model", "COMPACTION_THRESHOLD": "0.5"})
    second = create_new_job(supervised_agent)
    assert first.job_def is not second.job_def
    assert second.job_def.compaction_threshold == 0.5
    assert create_new_job(supervised_agent).job_def is second.job_def
    supervised_agent.update_metadata({"MODEL": "test-model", "COMPACTION_THRESHOLD": "None"})
    disabled = create_new_job(supervised_agent)
    assert disabled.job_def is not second.job_def
    assert disabled.job_def.compaction_threshold is None


def test_find_filter_and_setter(agent):
    definition = JobDef(agent=agent, compaction_threshold=0.5)
    assert find_existing_job_def(agent, None, compaction_threshold=0.5) is definition
    assert find_existing_job_def(agent, None, compaction_threshold=None) is None
    old_tag = job_def_identity_tag_for_job_def(definition)
    definition.set_compaction_threshold(None)
    assert definition not in list(db0.find(JobDef, old_tag))
    assert find_existing_job_def(agent, None, compaction_threshold=None) is definition
    with pytest.raises(ValueError, match="COMPACTION_THRESHOLD"):
        definition.set_compaction_threshold(0)
    assert definition.compaction_threshold is None


def test_default_keeps_legacy_identity_and_missing_field_is_reusable(agent):
    definition = JobDef(agent=agent)
    # The pre-threshold identity payload for this otherwise default definition.
    payload = (None, None, "test-model", None, None, None, (), None, ((), (), ()))
    old_tag = "STATEK_JOBDEF:H:" + hashlib.sha1(str(payload).encode("utf-8")).hexdigest()[:4]
    assert job_def_identity_tag_for_job_def(definition) == old_tag
    del definition.compaction_threshold
    assert find_existing_job_def(agent, None, compaction_threshold=0.65) is definition


@pytest.mark.parametrize("value", [0.5, None])
def test_threshold_survives_reopen(agent, value):
    definition = JobDef(agent=agent, compaction_threshold=value)
    identifier = db0.uuid(definition)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    restored = db0.fetch(identifier)
    assert restored.compaction_threshold == value
