"""Tests for show_example tool and examples_style in StatekSettings."""

# pylint: disable=redefined-outer-name

import os
from unittest.mock import patch

import pytest

from statek.agents.list_of_examples import show_example
from statek.executors.example import load_examples
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
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    assert "```python" in out


def test_show_example_uses_examples_style_over_chat_style(capsys, examples_dir):
    """When examples_style is set it takes priority over chat_style."""
    settings = _settings(
        examples_dir=examples_dir,
        chat_style=ChatStyle.MARKDOWN,    # pylint: disable=no-member
        examples_style=ChatStyle.CONSOLE, # pylint: disable=no-member
    )
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=0)

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
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=0)

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
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=0)

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
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=0)

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
    with patch("statek.agents.list_of_examples.get_statek_settings", return_value=settings):
        show_example(agent_name="myagent", example_id=0)

    out = capsys.readouterr().out
    assert "# --- EXAMPLE:" not in out
    assert "# --- END OF EXAMPLE ---" not in out
