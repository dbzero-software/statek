"""Tests for provider-specific model payload configuration."""

# pylint: disable=no-member,redefined-outer-name

import pytest
import dbzero as db0

from statek.provider_config import (
    ProviderConfig,
    _provider_config_identity_tag,
    resolve_provider_config,
)


@pytest.fixture
def provider_config(db0_fixture):
    """Create a hierarchical reasoning configuration for payload lookup tests."""
    del db0_fixture
    return ProviderConfig({
        "openrouter": {
            "reasoning": {
                "ignore_parameters": ["temperature"],
                "reasoning_level": [
                    {
                        "range": {"from": 1, "to": 100},
                        "payload": {"reasoning": {"effort": "low"}},
                    },
                ],
            },
            "openai": {
                "reasoning": {
                    "ignore_parameters": ["temperature", "top_p"],
                    "reasoning_level": [
                        {
                            "range": {"from": 26, "to": 75},
                            "payload": {"reasoning": {"effort": "medium"}},
                        },
                    ],
                },
                "gpt-5.4": {
                    "reasoning": {
                        "ignore_parameters": ["temperature", "logprobs"],
                        "reasoning_level": [
                            {
                                "range": {"from": 76},
                                "payload": {"reasoning": {"effort": "high"}},
                            },
                        ],
                    },
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


def test_find_payload_resolves_provider_model_path_when_family_is_redundant(db0_fixture):
    """Provider/model mappings remain reachable from a family/model selection."""
    del db0_fixture
    config = ProviderConfig({
        "openai": {
            "gpt-5": {
                "reasoning": {
                    "reasoning_level": [{
                        "range": {"from": 1},
                        "payload": {"reasoning_effort": "high"},
                    }],
                },
            },
        },
    })

    assert config.find_payload(
        "openai", "openai", "gpt-5", reasoning_level=50,
    ) == {"reasoning_effort": "high"}


def test_find_payload_returns_a_defensive_copy(provider_config):
    """Request formatting cannot mutate the durable provider configuration."""
    payload = provider_config.find_payload("openrouter", reasoning_level=10)
    payload["reasoning"]["effort"] = "changed"

    assert provider_config.find_payload("openrouter", reasoning_level=10) == {
        "reasoning": {"effort": "low"},
    }


def test_find_ignored_parameters_prefers_the_deepest_explicit_mapping(provider_config):
    """The most-specific explicit ignore list overrides broader reasoning policy."""
    assert provider_config.find_ignored_parameters(
        "openrouter", "openai", "gpt-5.4",
    ) == ["temperature", "logprobs"]


def test_find_ignored_parameters_falls_back_when_model_omits_the_list(provider_config):
    """A nested reasoning mapping inherits the closest explicit parent ignore list."""
    assert provider_config.find_ignored_parameters(
        "openrouter", "openai", "gpt-5.3",
    ) == ["temperature", "top_p"]


def test_find_ignored_parameters_allows_an_explicit_empty_override(db0_fixture):
    """An empty list disables a broader ignore policy for one model mapping."""
    del db0_fixture
    config = ProviderConfig({
        "openrouter": {
            "reasoning": {"ignore_parameters": ["temperature"]},
            "openai": {
                "gpt-5.4": {
                    "reasoning": {"ignore_parameters": []},
                },
            },
        },
    })

    assert config.find_ignored_parameters("openrouter", "openai", "gpt-5.4") == []


def test_find_ignored_parameters_returns_a_defensive_copy(provider_config):
    """Request formatting cannot mutate the durable ignored-parameter list."""
    parameters = provider_config.find_ignored_parameters("openrouter", "openai")
    parameters.append("logprobs")

    assert provider_config.find_ignored_parameters("openrouter", "openai") == [
        "temperature", "top_p",
    ]


@pytest.mark.parametrize(
    "reasoning_config",
    [
        [],
        {"ignore_parameters": "temperature"},
        {"ignore_parameters": ["temperature", 1]},
    ],
)
def test_reasoning_configuration_rejects_invalid_ignore_parameters(db0_fixture, reasoning_config):
    """The canonical reasoning object requires a list of parameter names."""
    del db0_fixture
    config = ProviderConfig({"openrouter": {"reasoning": reasoning_config}})

    with pytest.raises(ValueError, match="reasoning"):
        config.find_ignored_parameters("openrouter")


def test_flat_reasoning_level_configuration_is_not_supported(db0_fixture):
    """Only the canonical nested reasoning configuration resolves mappings."""
    del db0_fixture
    config = ProviderConfig({
        "openrouter": {
            "reasoning_level": [{
                "range": {"from": 1},
                "payload": {"reasoning": {"effort": "high"}},
            }],
        },
    })

    assert config.find_payload("openrouter", reasoning_level=1) is None


@pytest.mark.parametrize("reasoning_level", [-1, 101, "invalid"])
def test_find_payload_rejects_invalid_reasoning_levels(provider_config, reasoning_level):
    """Reasoning levels must be integer values in the Statek 0–100 range."""
    with pytest.raises(ValueError, match="reasoning_level"):
        provider_config.find_payload("openrouter", reasoning_level=reasoning_level)


def test_find_payload_rejects_unknown_query_parameters(provider_config):
    """Only the currently designed reasoning-level query is accepted."""
    with pytest.raises(ValueError, match="Unsupported provider configuration query"):
        provider_config.find_payload("openrouter", type="pro")


def test_resolve_provider_config_reuses_equal_content_despite_mapping_order(db0_fixture):
    """Equivalent configuration mappings share one durable snapshot."""
    del db0_fixture
    first = resolve_provider_config({"openrouter": {"timeout": 10, "retries": 2}})
    second = resolve_provider_config({"openrouter": {"retries": 2, "timeout": 10}})

    assert first is second


def test_resolve_provider_config_creates_distinct_snapshots_for_different_content(db0_fixture):
    """Different configuration content must not reuse an existing snapshot."""
    del db0_fixture
    first = resolve_provider_config({"openrouter": {"timeout": 10}})
    second = resolve_provider_config({"openrouter": {"timeout": 20}})

    assert first is not second


def test_resolve_provider_config_checks_content_after_identity_tag_collision(
    db0_fixture,
    monkeypatch,
):
    """A colliding identity tag only widens candidates; content remains authoritative."""
    del db0_fixture
    monkeypatch.setattr(
        "statek.provider_config._provider_config_identity_hash",
        lambda config: "collision",
    )

    first = resolve_provider_config({"openrouter": {"timeout": 10}})
    second = resolve_provider_config({"openrouter": {"timeout": 20}})

    assert first is not second


def test_resolve_provider_config_reuses_and_tags_matching_untagged_snapshot(
    db0_fixture,
):
    """An equal untagged snapshot is reused rather than duplicated."""
    del db0_fixture
    config = {"openrouter": {"timeout": 10}}
    untagged_snapshot = ProviderConfig(config)
    identity_tag = _provider_config_identity_tag(config)

    resolved = resolve_provider_config(config)

    assert resolved is untagged_snapshot
    assert untagged_snapshot in db0.find(ProviderConfig, identity_tag)


def test_resolve_provider_config_snapshots_the_source_mapping(db0_fixture):
    """Mutating the caller's mapping cannot change the durable configuration snapshot."""
    del db0_fixture
    config = {"openrouter": {"reasoning": {"effort": "low"}}}

    resolved = resolve_provider_config(config)
    config["openrouter"]["reasoning"]["effort"] = "high"

    assert resolved.provider_config == {
        "openrouter": {"reasoning": {"effort": "low"}},
    }
