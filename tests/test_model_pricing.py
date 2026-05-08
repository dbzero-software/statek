"""Tests for ModelPricing, get_model_pricing, and set_model_pricing."""

# pylint: disable=no-member,unused-argument,redefined-outer-name

from decimal import Decimal

import dbzero as db0
import pytest

from statek.model_pricing import (
    ModelPricing, get_model_pricing, set_model_pricing, init_model_pricing
)


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


# --- init_model_pricing tests ---

@pytest.fixture
def pricing_dir(tmp_path):
    """Provide a temporary directory for pricing files."""
    return tmp_path


def test_init_loads_csv_file(db0_fixture, pricing_dir):
    csv_content = (
        "PROVIDER,MODEL_FAMILY,MODEL,"
        "INPUT_PRICE_PER_M,INPUT_PRICE_PER_CACHED_M,OUTPUT_PRICE_PER_M\n"
        "openrouter,openai,gpt-5.4-mini,0.75,0.075,4.50\n"
    )
    (pricing_dir / "models.csv").write_text(csv_content)
    init_model_pricing(str(pricing_dir))
    pricing = get_model_pricing("openrouter", "gpt-5.4-mini", model_family="openai", no_create=True)
    assert pricing is not None
    assert pricing.input_price_per_M == Decimal("0.75")
    assert pricing.input_price_per_cached_M == Decimal("0.075")
    assert pricing.output_price_per_M == Decimal("4.50")


def test_init_loads_csv_without_model_family(db0_fixture, pricing_dir):
    csv_content = (
        "PROVIDER,MODEL_FAMILY,MODEL,"
        "INPUT_PRICE_PER_M,INPUT_PRICE_PER_CACHED_M,OUTPUT_PRICE_PER_M\n"
        "anthropic,,claude-3-opus,15.00,,75.00\n"
    )
    (pricing_dir / "models.csv").write_text(csv_content)
    init_model_pricing(str(pricing_dir))
    pricing = get_model_pricing("anthropic", "claude-3-opus", no_create=True)
    assert pricing is not None
    assert pricing.input_price_per_M == Decimal("15.00")
    assert pricing.input_price_per_cached_M is None
    assert pricing.output_price_per_M == Decimal("75.00")


def test_init_scans_subdirectories(db0_fixture, pricing_dir):
    subdir = pricing_dir / "subdir"
    subdir.mkdir()
    csv_content = (
        "PROVIDER,MODEL_FAMILY,MODEL,"
        "INPUT_PRICE_PER_M,INPUT_PRICE_PER_CACHED_M,OUTPUT_PRICE_PER_M\n"
        "openai,,gpt-4,10.00,1.00,30.00\n"
    )
    (subdir / "models.csv").write_text(csv_content)
    init_model_pricing(str(pricing_dir))
    pricing = get_model_pricing("openai", "gpt-4", no_create=True)
    assert pricing is not None
    assert pricing.input_price_per_M == Decimal("10.00")


def test_init_skips_malformed_rows(db0_fixture, pricing_dir):
    csv_content = (
        "PROVIDER,MODEL_FAMILY,MODEL,"
        "INPUT_PRICE_PER_M,INPUT_PRICE_PER_CACHED_M,OUTPUT_PRICE_PER_M\n"
        "openai,,gpt-4,not-a-number,,30.00\n"
        "openai,,gpt-4o,1.00,,3.00\n"
    )
    (pricing_dir / "models.csv").write_text(csv_content)
    init_model_pricing(str(pricing_dir))
    assert get_model_pricing("openai", "gpt-4", no_create=True) is None
    assert get_model_pricing("openai", "gpt-4o", no_create=True) is not None


def test_init_skips_non_csv_txt_files(db0_fixture, pricing_dir):
    (pricing_dir / "notes.json").write_text('{"provider": "openai"}')
    (pricing_dir / "readme.md").write_text("# pricing")
    init_model_pricing(str(pricing_dir))
    assert not list(db0.find(ModelPricing, "USAGE"))


def test_init_loads_multiple_rows(db0_fixture, pricing_dir):
    csv_content = (
        "PROVIDER,MODEL_FAMILY,MODEL,"
        "INPUT_PRICE_PER_M,INPUT_PRICE_PER_CACHED_M,OUTPUT_PRICE_PER_M\n"
        "openai,,gpt-4,10.00,,30.00\n"
        "openai,,gpt-3.5-turbo,0.50,,1.50\n"
    )
    (pricing_dir / "models.csv").write_text(csv_content)
    init_model_pricing(str(pricing_dir))
    assert get_model_pricing("openai", "gpt-4", no_create=True) is not None
    assert get_model_pricing("openai", "gpt-3.5-turbo", no_create=True) is not None
