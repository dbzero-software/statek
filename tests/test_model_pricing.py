"""Tests for ModelPricing, get_model_pricing, and set_model_pricing."""

# pylint: disable=no-member,unused-argument

from decimal import Decimal

import dbzero as db0

from statek.model_pricing import ModelPricing, get_model_pricing, set_model_pricing


def test_is_valid(db0_fixture):
    assert ModelPricing().is_valid is False
    assert ModelPricing(input_price_per_M=Decimal("1")).is_valid is False
    assert ModelPricing(
        input_price_per_M=Decimal("1"), output_price_per_M=Decimal("2")
    ).is_valid is True


def test_get_creates_placeholder_with_normalized_tags_and_usage(db0_fixture):
    pricing = get_model_pricing("Open AI", "gpt-4 turbo")
    assert pricing is not None
    assert pricing.is_valid is False
    assert pricing in list(db0.find(ModelPricing, "OPENAI", "GPT-4TURBO", "USAGE"))


def test_get_returns_same_object_and_no_create(db0_fixture):
    p1 = get_model_pricing("openai", "gpt-4")
    assert get_model_pricing("openai", "gpt-4") is p1
    assert get_model_pricing("openai", "gpt-4", no_create=True) is p1
    assert get_model_pricing("anthropic", "claude", no_create=True) is None


def test_set_initializes_empty_and_returns_identical(db0_fixture):
    placeholder = get_model_pricing("openai", "gpt-4")
    result = set_model_pricing("openai", "gpt-4", Decimal("5"), Decimal("15"))
    assert result is placeholder
    assert result.input_price_per_M == Decimal("5")
    assert set_model_pricing("openai", "gpt-4", Decimal("5"), Decimal("15")) is result


def test_set_creates_new_on_different_values_and_tags_correctly(db0_fixture):
    p1 = set_model_pricing(
        "openrouter", "llama", Decimal("1"), Decimal("2"), model_family="meta"
    )
    p2 = set_model_pricing(
        "openrouter", "llama", Decimal("3"), Decimal("6"), model_family="meta"
    )
    assert p1 is not p2
    assert p2 in list(db0.find(ModelPricing, "OPENROUTER", "LLAMA", "META", "USAGE"))
