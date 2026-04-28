"""Tests for LLM_Usage class and its total_cost resolution logic."""

# pylint: disable=no-member,unused-argument

from decimal import Decimal

import pytest

from statek.model_pricing import ModelPricing, set_model_pricing
from statek.executors.llm_usage import LLM_Usage


def test_total_cost_returns_reported_when_pricing_invalid(db0_fixture):
    usage = LLM_Usage(pricing=ModelPricing())
    usage.total_reported_cost = 1.5
    assert usage.total_cost == 1.5


def test_total_cost_returns_none_when_no_pricing_and_no_reported(db0_fixture):
    usage = LLM_Usage(pricing=ModelPricing())
    assert usage.total_cost is None


def test_total_cost_calculated_from_tokens_when_pricing_valid(db0_fixture):
    pricing = set_model_pricing("openai", "gpt-4", Decimal("10"), Decimal("30"))
    usage = LLM_Usage(pricing=pricing)
    usage.total_input_tokens = 500_000
    usage.total_output_tokens = 100_000
    # cost = 500_000/1M * 10 + 100_000/1M * 30 = 5.0 + 3.0 = 8.0
    assert usage.total_cost == pytest.approx(8.0)


def test_total_cost_uses_cached_price_when_set(db0_fixture):
    pricing = set_model_pricing(
        "anthropic", "claude-3",
        Decimal("10"), Decimal("30"),
        input_price_per_cached_M=Decimal("2"),
    )
    usage = LLM_Usage(pricing=pricing)
    usage.total_input_tokens = 1_000_000
    usage.total_cached_tokens = 500_000   # included in total_input_tokens
    usage.total_output_tokens = 0
    # non_cached = 500_000 → 500_000/1M * 10 = 5.0
    # cached    = 500_000 → 500_000/1M *  2 = 1.0
    assert usage.total_cost == pytest.approx(6.0)


def test_total_cost_falls_back_to_reported_when_no_tokens(db0_fixture):
    pricing = set_model_pricing("openai", "gpt-4", Decimal("10"), Decimal("30"))
    usage = LLM_Usage(pricing=pricing)
    usage.total_reported_cost = 0.99
    # tokens are 0 → cannot calculate → fall back
    assert usage.total_cost == 0.99
