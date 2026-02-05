"""Tests for statek.settings module."""

import os
import tempfile
from pathlib import Path
from statek.settings import StatekSettings, get_provider_settings


def test_statek_settings_parses_environment_variables():
    """Test that StatekSettings correctly parses provider-prefixed environment variables."""
    # Set up environment variables
    os.environ['OPENAI_API_URL'] = 'https://api.openai.com/v1'
    os.environ['OPENAI_API_KEY'] = 'test-key-123'
    os.environ['OPENAI_DEFAULT_MODEL'] = 'gpt-4'

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
        assert openai_settings.default_model == 'gpt-4'

        # Verify OPENROUTER settings
        router_settings = settings.llm_api_settings['OPENROUTER']
        assert router_settings.api_url == 'https://openrouter.com/api/v1'
        assert router_settings.api_key == 'test-router-key'
        assert router_settings.default_model is None

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
        del os.environ['OPENAI_DEFAULT_MODEL']
        del os.environ['OPENROUTER_API_URL']
        del os.environ['OPENROUTER_API_KEY']


def test_get_provider_settings_cached_function():
    """Test the module-level cached get_provider_settings function."""
    # Set up environment variables
    os.environ['GOOGLE_API_URL'] = 'https://api.google.com/v1'
    os.environ['GOOGLE_API_KEY'] = 'google-key-456'
    os.environ['GOOGLE_DEFAULT_MODEL'] = 'gemini-pro'

    try:
        # Clear the cache before testing
        get_provider_settings.cache_clear()

        # Call the cached function
        settings = get_provider_settings('GOOGLE')

        # Verify settings were retrieved correctly
        assert settings is not None
        assert settings.api_url == 'https://api.google.com/v1'
        assert settings.api_key == 'google-key-456'
        assert settings.default_model == 'gemini-pro'

        # Call again to verify caching works (should return same object from cache)
        cached_settings = get_provider_settings('GOOGLE')
        assert cached_settings is settings

    finally:
        # Clean up
        del os.environ['GOOGLE_API_URL']
        del os.environ['GOOGLE_API_KEY']
        del os.environ['GOOGLE_DEFAULT_MODEL']
        get_provider_settings.cache_clear()


def test_get_prompt_def_from_settings_instance():
    """Test that StatekSettings.get_prompt_def retrieves prompt definitions correctly."""
    # Create a temporary directory with prompt files
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir)

        # Create a test prompt file
        test_prompt = prompt_dir / "test_agent.md"
        test_prompt.write_text("""# System
You are a test agent. You help with testing.

# Template
Test question: {question}
""")

        # Create another prompt file
        another_prompt = prompt_dir / "another_agent.md"
        another_prompt.write_text("""# System
You are another agent.

---

# Template
Input: {input}
Output: {output}
""")

        # Initialize settings with the prompt directory
        settings = StatekSettings(prompt_files_dir=str(prompt_dir))

        # Test retrieving existing prompt definition
        prompt_def = settings.get_prompt_def("test_agent")
        assert prompt_def is not None
        assert "test agent" in prompt_def.system.lower()
        assert "testing" in prompt_def.system.lower()
        assert "Test question:" in prompt_def.template
        assert "{question}" in prompt_def.template

        # Test retrieving another prompt definition
        another_def = settings.get_prompt_def("another_agent")
        assert another_def is not None
        assert "another agent" in another_def.system.lower()
        assert "Input:" in another_def.template
        assert "{output}" in another_def.template

        # Test retrieving non-existent prompt definition
        missing_def = settings.get_prompt_def("nonexistent")
        assert missing_def is None
