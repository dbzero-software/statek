"""Tests for provider-specific model payload configuration."""

# pylint: disable=no-member,redefined-outer-name

import pytest
import dbzero as db0

from statek.provider_config import ProviderConfig


@pytest.fixture
def provider_config(db0_fixture):
    """Create a hierarchical reasoning configuration for payload lookup tests."""
    del db0_fixture
    return ProviderConfig({
        "openrouter": {
            "reasoning_level": [
                {
                    "range": {"from": 1, "to": 100},
                    "payload": {"reasoning": {"effort": "low"}},
                },
            ],
            "openai": {
                "reasoning_level": [
                    {
                        "range": {"from": 26, "to": 75},
                        "payload": {"reasoning": {"effort": "medium"}},
                    },
                ],
                "gpt-5.4": {
                    "reasoning_level": [
                        {
                            "range": {"from": 76},
                            "payload": {"reasoning": {"effort": "high"}},
                        },
                    ],
                },
            },
        },
    })


def test_find_payload_returns_provider_level_mapping(provider_config):
    """Provider-level mappings apply when no more-specific mapping matches."""
    assert provider_config.find_payload("openrouter", reasoning_level=10) == {
        "reasoning": {"effort": "low"},
    }


def test_provider_config_is_a_durable_dbzero_memo(provider_config):
    """Provider configuration snapshots are persisted as dbzero memo objects."""
    assert db0.get_type(provider_config) is ProviderConfig


def test_find_payload_prefers_the_deepest_matching_mapping(provider_config):
    """A matching model mapping takes precedence over family and provider mappings."""
    assert provider_config.find_payload(
        "openrouter", "openai", "gpt-5.4", reasoning_level=90
    ) == {"reasoning": {"effort": "high"}}


def test_find_payload_falls_back_when_a_specific_range_does_not_match(provider_config):
    """A nonmatching model range falls back to the next matching parent mapping."""
    assert provider_config.find_payload(
        "openrouter", "openai", "gpt-5.4", reasoning_level=50
    ) == {"reasoning": {"effort": "medium"}}


def test_find_payload_falls_back_to_provider_when_family_range_does_not_match(provider_config):
    """Lookup continues to a broader configuration when no family range matches."""
    assert provider_config.find_payload(
        "openrouter", "openai", "gpt-5.4", reasoning_level=20
    ) == {"reasoning": {"effort": "low"}}


def test_find_payload_accepts_inclusive_bounds_and_numeric_strings(provider_config):
    """Range endpoints are inclusive and model parameters can supply string levels."""
    assert provider_config.find_payload(
        "openrouter", "openai", "gpt-5.4", reasoning_level="75"
    ) == {"reasoning": {"effort": "medium"}}
    assert provider_config.find_payload(
        "openrouter", "openai", "gpt-5.4", reasoning_level="76"
    ) == {"reasoning": {"effort": "high"}}


def test_find_payload_ignores_missing_model_path_components(provider_config):
    """Missing optional path components do not prevent provider-level lookup."""
    assert provider_config.find_payload(
        "openrouter", None, None, reasoning_level=10
    ) == {"reasoning": {"effort": "low"}}


def test_find_payload_returns_a_defensive_copy(provider_config):
    """Request formatting cannot mutate the durable provider configuration."""
    payload = provider_config.find_payload("openrouter", reasoning_level=10)
    payload["reasoning"]["effort"] = "changed"

    assert provider_config.find_payload("openrouter", reasoning_level=10) == {
        "reasoning": {"effort": "low"},
    }


@pytest.mark.parametrize("reasoning_level", [-1, 101, "invalid"])
def test_find_payload_rejects_invalid_reasoning_levels(provider_config, reasoning_level):
    """Reasoning levels must be integer values in the Statek 0–100 range."""
    with pytest.raises(ValueError, match="reasoning_level"):
        provider_config.find_payload("openrouter", reasoning_level=reasoning_level)


def test_find_payload_rejects_unknown_query_parameters(provider_config):
    """Only the currently designed reasoning-level query is accepted."""
    with pytest.raises(ValueError, match="Unsupported provider configuration query"):
        provider_config.find_payload("openrouter", type="pro")
