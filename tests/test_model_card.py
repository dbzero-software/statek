"""Tests for persistent model cards."""

# pylint: disable=no-member,unused-argument

import dbzero as db0
import pytest

from statek.model_card import ModelCard, get_model_card, set_model_card


def test_get_creates_default_card_with_normalized_tags(db0_fixture):
    card = get_model_card("Open AI", "gpt 4", model_family="GPT Family")

    assert card.max_context_length == 0
    assert card.token_to_bytes == 0.0
    assert card.total_parameter_count is None
    assert card.active_parameter_count is None
    assert card in list(db0.find(ModelCard, "OPENAI", "GPT4", "GPTFAMILY"))
    assert get_model_card("openai", "gpt4", "gptfamily") is card


def test_get_no_create_and_constructor_values(db0_fixture):
    assert get_model_card("openai", "gpt-4", no_create=True) is None

    card = get_model_card(
        "openai", "gpt-4", max_context_length=128000, token_to_bytes=2.5
    )
    assert card.max_context_length == 128000
    assert card.token_to_bytes == 2.5
    assert get_model_card("openai", "gpt-4", max_context_length=256000) is card


def test_set_initializes_default_fields_and_is_idempotent(db0_fixture):
    card = get_model_card("openai", "gpt-4")

    result = set_model_card(
        "openai", "gpt-4", max_context_length=128000,
        token_to_bytes=2.5, total_parameter_count=100,
    )
    assert result is card
    assert result.max_context_length == 128000
    assert result.token_to_bytes == 2.5
    assert result.total_parameter_count == 100
    assert result.active_parameter_count is None
    assert set_model_card("openai", "gpt-4", max_context_length=128000) is card
    assert list(db0.find(ModelCard, "OPENAI", "GPT-4")) == [card]


def test_set_creates_card_if_missing(db0_fixture):
    card = set_model_card(
        "openrouter", "llama", model_family="meta", active_parameter_count=10
    )

    assert card.max_context_length == 0
    assert card.active_parameter_count == 10
    assert card in list(db0.find(ModelCard, "OPENROUTER", "LLAMA", "META"))


def test_set_rejects_changed_initialized_value_atomically(db0_fixture):
    card = set_model_card("openai", "gpt-4", max_context_length=128000)

    with pytest.raises(ValueError, match="max_context_length"):
        set_model_card(
            "openai", "gpt-4", token_to_bytes=2.5, max_context_length=256000
        )

    assert card.max_context_length == 128000
    assert card.token_to_bytes == 0.0
    assert list(db0.find(ModelCard, "OPENAI", "GPT-4")) == [card]


def test_set_rejects_changed_optional_parameter_count(db0_fixture):
    card = set_model_card("openai", "gpt-4", total_parameter_count=100)

    with pytest.raises(ValueError, match="total_parameter_count"):
        set_model_card("openai", "gpt-4", total_parameter_count=200)

    assert card.total_parameter_count == 100
