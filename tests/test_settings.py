"""Tests for statek.settings module."""

# pylint: disable=no-member

import os
import tempfile
import json
from pathlib import Path
import pytest
from statek.provider_config import load_provider_config
from statek.multi_source_settings import SettingValuesSource
from statek.settings import StatekSettings, get_provider_settings
from statek.docstring import ACL_Item, Statek_ACL
from statek.prompt_config import format_system_prompt
from statek.task_difficulty import TaskDifficulty


def test_statek_settings_parses_environment_variables():
    """Test that StatekSettings correctly parses provider-prefixed environment variables."""
    # Set up environment variables
    os.environ['OPENAI_API_URL'] = 'https://api.openai.com/v1'
    os.environ['OPENAI_API_KEY'] = 'test-key-123'

    os.environ['OPENROUTER_API_URL'] = 'https://openrouter.com/api/v1'
    os.environ['OPENROUTER_API_KEY'] = 'test-router-key'

    try:
        # Initialize settings
        settings = StatekSettings(default_llm_api_provider='OPENAI')

        # Verify both providers were parsed
        assert len(settings.llm_api_settings) == 2
        assert 'OPENAI' in settings.llm_api_settings
        assert 'OPENROUTER' in settings.llm_api_settings

        # Verify OPENAI settings
        openai_settings = settings.llm_api_settings['OPENAI']
        assert openai_settings.api_url == 'https://api.openai.com/v1'
        assert openai_settings.api_key == 'test-key-123'

        # Verify OPENROUTER settings
        router_settings = settings.llm_api_settings['OPENROUTER']
        assert router_settings.api_url == 'https://openrouter.com/api/v1'
        assert router_settings.api_key == 'test-router-key'

        # Verify default provider
        assert settings.default_llm_api_provider == 'OPENAI'

        # Verify get_provider_settings works
        default_settings = settings.get_provider_settings()
        assert default_settings is not None
        assert default_settings.api_url == 'https://api.openai.com/v1'

        # Verify get_provider_settings with specific provider
        router_settings_via_getter = settings.get_provider_settings('OPENROUTER')
        assert router_settings_via_getter is not None
        assert router_settings_via_getter.api_url == 'https://openrouter.com/api/v1'

    finally:
        # Clean up environment variables
        del os.environ['OPENAI_API_URL']
        del os.environ['OPENAI_API_KEY']
        del os.environ['OPENROUTER_API_URL']
        del os.environ['OPENROUTER_API_KEY']


def test_statek_provider_config_path_supports_explicit_and_environment_values(monkeypatch):
    """Provider configuration paths can be supplied directly or through Statek's environment."""
    monkeypatch.setenv("STATEK_PROVIDER_CONFIG", "/tmp/from-environment.json")

    assert StatekSettings().statek_provider_config == "/tmp/from-environment.json"
    assert (
        StatekSettings(statek_provider_config="/tmp/explicit.json").statek_provider_config
        == "/tmp/explicit.json"
    )


def test_statek_provider_config_path_supports_configured_value_sources():
    """Provider configuration paths participate in Statek's ordered value sources."""
    class ProviderConfigSource(SettingValuesSource):
        """Provide one configured provider-configuration path for the test."""

        def get_value(self, field_name):
            if field_name == "statek_provider_config":
                return "/tmp/from-source.json"
            return None

    settings = StatekSettings(sources=[ProviderConfigSource()])

    assert settings.statek_provider_config == "/tmp/from-source.json"


def test_load_provider_config_reads_a_json_object(tmp_path):
    """The provider configuration loader returns parsed JSON object content."""
    config_path = tmp_path / "provider-config.json"
    config_path.write_text(json.dumps({"openrouter": {"timeout": 10}}), encoding="utf-8")

    assert load_provider_config(str(config_path)) == {"openrouter": {"timeout": 10}}


def test_load_provider_config_rejects_a_non_object_json_document(tmp_path):
    """Provider configuration files must have an object at their top level."""
    config_path = tmp_path / "provider-config.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_provider_config(str(config_path))


def test_get_provider_settings_cached_function():
    """Test the module-level cached get_provider_settings function."""
    # Set up environment variables
    os.environ['GOOGLE_API_URL'] = 'https://api.google.com/v1'
    os.environ['GOOGLE_API_KEY'] = 'google-key-456'

    try:
        # Clear the cache before testing
        get_provider_settings.cache_clear()

        # Call the cached function
        settings = get_provider_settings('GOOGLE')

        # Verify settings were retrieved correctly
        assert settings is not None
        assert settings.api_url == 'https://api.google.com/v1'
        assert settings.api_key == 'google-key-456'

        # Call again to verify caching works (should return same object from cache)
        cached_settings = get_provider_settings('GOOGLE')
        assert cached_settings is settings

    finally:
        # Clean up
        del os.environ['GOOGLE_API_URL']
        del os.environ['GOOGLE_API_KEY']
        get_provider_settings.cache_clear()


def test_get_prompt_def_from_settings_instance():
    """Test that StatekSettings.get_prompt_def retrieves prompt definitions correctly."""
    # Create a temporary directory with prompt files
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir)

        # Create a test prompt file
        test_prompt = prompt_dir / "test_agent.md"
        test_prompt.write_text("""# MODEL: gpt-4o
# LLM_TOOLS_SCOPE: SYSTEM
# System Prompt
You are a test agent. You help with testing.
""")

        # Create another prompt file
        another_prompt = prompt_dir / "another_agent.md"
        another_prompt.write_text("""# MODEL: openai/gpt-4o
# CHAT_STYLE: CONSOLE
# System Prompt
You are another agent.
""")

        # Initialize settings with the prompt directory
        settings = StatekSettings(prompt_files_dir=str(prompt_dir))

        # Test retrieving existing prompt definition
        prompt_def = settings.get_prompt_def("test_agent")
        assert prompt_def is not None
        system_prompt = format_system_prompt(prompt_def.system, TaskDifficulty.medium).lower()
        assert "test agent" in system_prompt
        assert "testing" in system_prompt
        assert prompt_def.metadata.get("MODEL") == "gpt-4o"
        assert prompt_def.metadata.get("LLM_TOOLS_SCOPE") == "SYSTEM"

        # Test retrieving another prompt definition
        another_def = settings.get_prompt_def("another_agent")
        assert another_def is not None
        another_system_prompt = format_system_prompt(
            another_def.system,
            TaskDifficulty.medium,
        ).lower()
        assert "another agent" in another_system_prompt
        assert another_def.metadata.get("MODEL") == "openai/gpt-4o"
        assert another_def.metadata.get("CHAT_STYLE") == "CONSOLE"

        # Test retrieving non-existent prompt definition
        missing_def = settings.get_prompt_def("nonexistent")
        assert missing_def is None


def test_get_xml_box_tags_returns_empty_when_not_configured():
    """Test that get_xml_box_tags returns an empty dict when no tags are configured."""
    settings = StatekSettings()
    assert not settings.get_xml_box_tags()


def test_get_xml_box_tags_returns_configured_tags():
    """Test that get_xml_box_tags returns only configured XML box tags."""
    settings = StatekSettings(xml_box_console="console_output", xml_box_example="code_block")
    tags = settings.get_xml_box_tags()
    assert tags == {"console": "console_output", "example": "code_block"}


def test_get_xml_box_tags_partial_configuration():
    """Test that get_xml_box_tags excludes unconfigured tags."""
    settings = StatekSettings(xml_box_console="output_box")
    tags = settings.get_xml_box_tags()
    assert tags == {"console": "output_box"}
    assert "example" not in tags

    settings2 = StatekSettings(xml_box_example="example_box")
    tags2 = settings2.get_xml_box_tags()
    assert tags2 == {"example": "example_box"}
    assert "console" not in tags2


def test_get_xml_box_tags_from_environment_variables():
    """Test that xml_box fields are read from STATEK_XML_BOX_* environment variables."""
    os.environ['STATEK_XML_BOX_CONSOLE'] = 'env_console_tag'
    os.environ['STATEK_XML_BOX_EXAMPLE'] = 'env_example_tag'
    try:
        settings = StatekSettings()
    finally:
        del os.environ['STATEK_XML_BOX_CONSOLE']
        del os.environ['STATEK_XML_BOX_EXAMPLE']

    assert settings.xml_box_console == 'env_console_tag'
    assert settings.xml_box_example == 'env_example_tag'
    tags = settings.get_xml_box_tags()
    assert tags == {"console": "env_console_tag", "example": "env_example_tag"}


def test_default_acl_returns_deny_statek_acl_by_default():
    """Test that default_acl property returns a DENY Statek_ACL by default."""
    settings = StatekSettings()
    acl = settings.default_acl
    assert isinstance(acl, Statek_ACL)
    assert acl.acl == [ACL_Item(access=False, name="", is_prefix=True, scope=[])]


def test_default_acl_returns_grant_statek_acl_when_configured():
    """Test that default_acl property returns a GRANT Statek_ACL when set to GRANT."""
    settings = StatekSettings(default_acl_str="GRANT")
    acl = settings.default_acl
    assert isinstance(acl, Statek_ACL)
    assert acl.acl == [ACL_Item(access=True, name="", is_prefix=True, scope=[])]


def test_default_acl_loaded_from_environment_variable():
    """Test that STATEK_DEFAULT_ACL env var configures the default_acl property."""
    os.environ['STATEK_DEFAULT_ACL'] = 'GRANT'
    try:
        settings = StatekSettings()
    finally:
        del os.environ['STATEK_DEFAULT_ACL']
    acl = settings.default_acl
    assert isinstance(acl, Statek_ACL)
    assert acl.acl == [ACL_Item(access=True, name="", is_prefix=True, scope=[])]


def test_statek_default_difficulty_defaults_to_medium_short_label():
    """The settings default difficulty uses the short medium label by default."""
    settings = StatekSettings()
    assert settings.statek_default_difficulty == "M"


def test_statek_default_difficulty_loaded_from_environment_variable():
    """STATEK_DEFAULT_DIFFICULTY configures the settings default difficulty."""
    os.environ['STATEK_DEFAULT_DIFFICULTY'] = 'H'
    try:
        settings = StatekSettings()
    finally:
        del os.environ['STATEK_DEFAULT_DIFFICULTY']

    assert settings.statek_default_difficulty == "H"


def test_statek_model_info_dir_defaults_to_none():
    settings = StatekSettings()
    assert settings.statek_model_info_dir is None


def test_statek_model_info_dir_from_environment_variable():
    os.environ['STATEK_MODEL_INFO_DIR'] = '/some/path'
    try:
        settings = StatekSettings()
    finally:
        del os.environ['STATEK_MODEL_INFO_DIR']
    assert settings.statek_model_info_dir == '/some/path'
