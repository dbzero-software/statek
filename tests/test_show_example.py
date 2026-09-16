"""Tests for show_example tool, examples_style in StatekSettings, and Agent.get_examples."""

# pylint: disable=redefined-outer-name

import os
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import StatekContextJob, run_with_statek_job
from statek.agents.agent import Agent
from statek.agents.list_of_examples import (
    get_example_difficulty,
    list_of_examples,
    show_example,
)
from statek.executors.example import load_examples
from statek.executors.job import Job, JobDef, TaskDifficulty
from statek.extra_resources import parse_extra_resources
from statek.prompt_config import make_system_prompt
from statek.settings import StatekSettings, ChatStyle, get_statek_settings


EXAMPLE_MD = """\
# seq_id: 0
# name: Dispatching to new thread
```python
print(chat_history(10))
```
[]
# ----------
```python
start_new_thread()
```
"""


@pytest.fixture(autouse=True)
def clear_caches():
    get_statek_settings.cache_clear()
    load_examples.cache_clear()
    yield
    get_statek_settings.cache_clear()
    load_examples.cache_clear()


@pytest.fixture
def examples_dir(temp_dir):
    """Create a temp examples dir with one example for agent 'myagent'."""
    agent_dir = os.path.join(temp_dir, "myagent")
    os.makedirs(agent_dir)
    with open(os.path.join(agent_dir, "example-001.md"), "w", encoding="utf-8") as f:
        f.write(EXAMPLE_MD)
    return temp_dir


def _settings(**kwargs) -> StatekSettings:
    """Build a StatekSettings with no env-var side-effects."""
    return StatekSettings(**kwargs)


def _show_example_with_job(job, **kwargs):
    return run_with_statek_job(job, lambda: show_example(**kwargs))


# --- StatekSettings.examples_style ---

def test_examples_style_defaults_to_none():
    settings = _settings()
    assert settings.examples_style is None


def test_examples_style_can_be_set_directly():
    settings = _settings(examples_style=ChatStyle.MARKDOWN)  # pylint: disable=no-member
    assert settings.examples_style == ChatStyle.MARKDOWN  # pylint: disable=no-member


def test_examples_style_read_from_env_var():
    os.environ['STATEK_EXAMPLES_STYLE'] = 'MARKDOWN'
    try:
        settings = _settings()
        assert settings.examples_style == ChatStyle.MARKDOWN  # pylint: disable=no-member
    finally:
        del os.environ['STATEK_EXAMPLES_STYLE']


# --- show_example: style selection ---

def test_show_example_uses_chat_style_when_examples_style_not_set(capsys, examples_dir):
    """Without examples_style, show_example uses chat_style (MARKDOWN here)."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.MARKDOWN,  # pylint: disable=no-member
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    assert "```python" in out


def test_show_example_uses_examples_style_over_chat_style(capsys, examples_dir):
    """When examples_style is set it takes priority over chat_style."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.MARKDOWN,    # pylint: disable=no-member
        examples_style=ChatStyle.CONSOLE, # pylint: disable=no-member
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    assert "```python" not in out   # CONSOLE style has no fences
    assert "> []" in out            # console lines are prefixed with "> "


# --- show_example: no boxing → prefix/suffix comments ---

def test_show_example_without_boxing_prints_prefix_and_suffix(capsys, examples_dir):
    """Without xml_box_example, output is wrapped in # --- EXAMPLE / END OF EXAMPLE comments."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "# --- EXAMPLE: Dispatching to new thread ---"
    assert lines[-1] == "# --- END OF EXAMPLE ---"


def test_show_example_prefix_contains_example_name(capsys, examples_dir):
    """The prefix comment includes the example name from metadata."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    assert "Dispatching to new thread" in out.splitlines()[0]


# --- show_example: with boxing → XML tags, no prefix/suffix ---

def test_show_example_with_boxing_wraps_in_xml_tag(capsys, examples_dir):
    """With xml_box_example set, output is wrapped in XML tags instead of comments."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
        xml_box_example="EXAMPLE",
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "<EXAMPLE>"
    assert lines[-1] == "</EXAMPLE>"


def test_show_example_with_boxing_has_no_comment_prefix_or_suffix(capsys, examples_dir):
    """With xml_box_example set, no # --- EXAMPLE --- comment lines are added."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
        xml_box_example="EXAMPLE",
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    assert "# --- EXAMPLE:" not in out
    assert "# --- END OF EXAMPLE ---" not in out


# --- show_example: default example_id (None) ---

def test_show_example_none_id_uses_default_example_id_from_local_context(capsys, examples_dir):
    """When example_id is None, show_example reads default_example_id from local context."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        default_example_id = 0  # noqa: F841  # pylint: disable=unused-variable
        _show_example_with_job(job, agent_name="myagent", example_id=None)

    out = capsys.readouterr().out
    assert "Dispatching to new thread" in out


def test_show_example_none_id_no_default_prints_not_found(capsys, examples_dir):
    """When example_id is None and no default_example_id in context, prints 'Example not found'."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=None)

    out = capsys.readouterr().out
    assert "Example not found" in out


def test_show_example_none_id_string_default_is_converted_to_int(capsys, examples_dir):
    """When default_example_id is a string, it is converted to int and used."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    job = StatekContextJob()
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        default_example_id = "0"  # noqa: F841  # pylint: disable=unused-variable
        _show_example_with_job(job, agent_name="myagent", example_id=None)

    out = capsys.readouterr().out
    assert "Dispatching to new thread" in out


def test_show_example_none_id_non_numeric_default_prints_not_found(capsys, examples_dir):
    """When default_example_id is a non-numeric string, prints 'Example not found'."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        default_example_id = "abc"  # noqa: F841  # pylint: disable=unused-variable
        show_example(agent_name="myagent", example_id=None)

    out = capsys.readouterr().out
    assert "Example not found" in out


def test_show_example_none_id_invalid_default_prints_not_found(capsys, examples_dir):
    """When default_example_id is out of range, prints 'Example not found'."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        default_example_id = 999  # noqa: F841  # pylint: disable=unused-variable
        show_example(agent_name="myagent", example_id=None)

    out = capsys.readouterr().out
    assert "not found" in out


def test_show_example_sets_last_example_id(capsys, examples_dir):
    """Successful show_example stores the selected example ID in persistent context."""
    job = StatekContextJob()
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=0)

    capsys.readouterr()
    assert job.py_env.local_state["_PERM_CTX"]["last_example_id"] == 0


def test_show_example_default_id_sets_last_example_id(capsys, examples_dir):
    """When default_example_id is used, the resolved ID is stored as last_example_id."""
    job = StatekContextJob()
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        default_example_id = "0"  # noqa: F841  # pylint: disable=unused-variable
        _show_example_with_job(job, agent_name="myagent", example_id=None)

    capsys.readouterr()
    assert job.py_env.local_state["_PERM_CTX"]["last_example_id"] == 0


def test_show_example_syncs_last_example_id_to_current_job(
    capsys, examples_dir, job_factory
):
    """show_example mirrors last_example_id into the current job's PyEnv context."""
    job = job_factory()
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        run_with_statek_job(job, lambda: show_example(agent_name="myagent", example_id=0))

    capsys.readouterr()
    assert job.py_env.local_state["_PERM_CTX"]["last_example_id"] == 0


def test_show_example_missing_id_does_not_overwrite_last_example_id(capsys, examples_dir):
    """Failed lookups do not replace the previous successful example trace."""
    job = StatekContextJob()
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 7}
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        _show_example_with_job(job, agent_name="myagent", example_id=99)

    capsys.readouterr()
    assert job.py_env.local_state["_PERM_CTX"]["last_example_id"] == 7


# --- Agent.get_examples ---

EXAMPLE_MD_2 = """\
# seq_id: 1
# name: Second example
```python
print("hello")
```
"""


@pytest.fixture
def examples_dir_multi(temp_dir):
    """Create a temp examples dir with two examples for agent 'myagent'."""
    agent_dir = os.path.join(temp_dir, "myagent")
    os.makedirs(agent_dir, exist_ok=True)
    with open(os.path.join(agent_dir, "example-001.md"), "w", encoding="utf-8") as f:
        f.write(EXAMPLE_MD)
    with open(os.path.join(agent_dir, "example-002.md"), "w", encoding="utf-8") as f:
        f.write(EXAMPLE_MD_2)
    return temp_dir


def _make_agent(role):
    """Create a minimal mock Agent with the given role for testing get_examples."""
    agent = MagicMock()
    agent.role = role
    agent.get_examples = Agent.get_examples.__get__(agent, Agent)
    return agent


def test_get_examples_returns_names_in_order(examples_dir_multi):
    """get_examples returns example names ordered by seq_id (index = example ID)."""
    settings = _settings(examples_dir=examples_dir_multi)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        agent = _make_agent("myagent")
        result = agent.get_examples()

    assert result == ["Dispatching to new thread", "Second example"]


def test_get_examples_returns_empty_list_when_no_examples_dir():
    """get_examples returns [] when examples_dir is not configured."""
    settings = _settings(examples_dir=None)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        agent = _make_agent("myagent")
        result = agent.get_examples()

    assert result == []


def test_get_examples_returns_empty_list_when_agent_dir_missing(temp_dir):
    """get_examples returns [] when the agent's subdirectory does not exist."""
    settings = _settings(examples_dir=temp_dir)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        agent = _make_agent("nonexistent_agent")
        result = agent.get_examples()

    assert result == []


def test_get_examples_single_example(examples_dir):
    """get_examples with a single example returns a one-element list."""
    settings = _settings(examples_dir=examples_dir)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        agent = _make_agent("myagent")
        result = agent.get_examples()

    assert result == ["Dispatching to new thread"]


@pytest.mark.usefixtures("db0_fixture")
def test_get_example_difficulty_reads_example_metadata(temp_dir):
    """Example difficulty is parsed from example metadata by example ID."""
    agent_dir = os.path.join(temp_dir, "myagent")
    os.makedirs(agent_dir)
    with open(os.path.join(agent_dir, "example-001.md"), "w", encoding="utf-8") as f:
        f.write("# seq_id: 0\n# name: Easy\n# difficulty: low\n```python\nx = 1\n```\n")

    settings = _settings(examples_dir=temp_dir)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        assert get_example_difficulty("myagent", 0) == TaskDifficulty.low  # pylint: disable=no-member


def test_get_example_difficulty_returns_none_when_missing(examples_dir):
    """Missing difficulty metadata is not an error."""
    settings = _settings(examples_dir=examples_dir)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        assert get_example_difficulty("myagent", 0) is None


@pytest.fixture
def expanded_examples_dir(temp_dir):
    """Create receiver and donor examples with overlapping local indexes."""
    examples = {
        "receiver": [("Receiver zero", "low"), ("Receiver one", "medium")],
        "preferences_assistant": [("Preference zero", "high")],
        "schedule_assistant": [("Schedule zero", "medium")],
    }
    for role, role_examples in examples.items():
        role_dir = os.path.join(temp_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        for index, (name, difficulty) in enumerate(role_examples):
            content = (
                f"# seq_id: {index}\n# name: {name}\n# difficulty: {difficulty}\n"
                f"```python\nprint({name!r})\n```\n"
            )
            path = os.path.join(role_dir, f"example-{index}.md")
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
    return temp_dir


@pytest.mark.usefixtures("db0_fixture")
def test_list_of_examples_combines_roles_with_stable_public_ids(
    capsys, expanded_examples_dir,
):
    """Combined examples keep receiver-first IDs and append newly active donors."""
    settings = _settings(examples_dir=expanded_examples_dir)
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        list_of_examples(["receiver", "preferences_assistant"], start_index=1, limit=2)
    first = capsys.readouterr().out

    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        list_of_examples(
            ["receiver", "preferences_assistant", "schedule_assistant"],
            start_index=0,
            limit=10,
        )
    expanded = capsys.readouterr().out

    assert "1: Receiver one" in first
    assert "2: Preference zero" in first
    assert "0: Receiver zero" in expanded
    assert "1: Receiver one" in expanded
    assert "2: Preference zero" in expanded
    assert "3: Schedule zero" in expanded


@pytest.mark.usefixtures("db0_fixture")
def test_show_example_records_donor_source_and_raises_job_difficulty(
    capsys, expanded_examples_dir,
):
    """A donor example stores its local source and drives difficulty from that donor."""
    receiver = Agent(
        role="receiver", _system_prompt=make_system_prompt("Receiver"),
        _tools=[], _metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": "L"},
    )
    donor = Agent(
        role="preferences_assistant", _system_prompt=make_system_prompt("Preferences"),
        _tools=[], _metadata={"MODEL": "test-model"},
    )
    job = Job(job_def=JobDef(
        agent=receiver,
        metadata={"MODEL": "test-model", "DEFAULT_DIFFICULTY": "L"},
        extra_resources=parse_extra_resources("L:preferences_assistant"),
    ))
    settings = _settings(
        examples_dir=expanded_examples_dir,
        chat_style=ChatStyle.CONSOLE,  # pylint: disable=no-member
    )

    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        run_with_statek_job(
            job,
            lambda: show_example(["receiver", "preferences_assistant"], example_id=2),
        )

    capsys.readouterr()
    assert job.py_env.local_state["_PERM_CTX"]["last_example_id"] == 2
    assert job.py_env.local_state["_PERM_CTX"]["last_example_source"] == {
        "agent_name": "preferences_assistant", "example_id": 0,
    }
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        assert job.get_current_difficulty() == TaskDifficulty.high  # pylint: disable=no-member
    assert donor in job.get_resource_agents()
