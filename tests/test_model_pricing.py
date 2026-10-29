"""Tests for ModelPricing, get_model_pricing, and set_model_pricing."""

# pylint: disable=no-member,unused-argument,redefined-outer-name

from decimal import Decimal

import dbzero as db0
import pytest
import statek

from statek.model_pricing import (
    ModelPricing, get_model_pricing, set_model_pricing, init_model_pricing
)
from statek.model_card import ModelCard, get_model_card, init_model_cards
from statek.settings import StatekSettings


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


def test_init_loads_standalone_card_and_pricing_files(db0_fixture, pricing_dir):
    (pricing_dir / "prices.csv").write_text(
        "PROVIDER,MODEL_FAMILY,MODEL,INPUT_PRICE_PER_M,OUTPUT_PRICE_PER_M\n"
        "openrouter,openai,gpt-mini,0.75,4.50\n"
    )
    (pricing_dir / "cards.csv").write_text(
        "PROVIDER,MODEL_FAMILY,MODEL,MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES,"
        "TOTAL_PARAMETER_COUNT,ACTIVE_PARAMETER_COUNT\n"
        "openrouter,openai,gpt-mini,400000,4.0,,\n"
    )

    init_model_pricing(str(pricing_dir))
    assert get_model_card("openrouter", "gpt-mini", "openai", no_create=True) is None
    init_model_cards(str(pricing_dir))

    pricing = get_model_pricing("openrouter", "gpt-mini", "openai", no_create=True)
    card = get_model_card("openrouter", "gpt-mini", "openai", no_create=True)
    assert pricing.input_price_per_M == Decimal("0.75")
    assert pricing.input_price_per_cached_M is None
    assert card.max_context_length == 400000
    assert card.token_to_bytes == 4.0
    assert card.total_parameter_count is None


def test_init_loads_combined_file_with_optional_counts(db0_fixture, pricing_dir):
    (pricing_dir / "combined.csv").write_text(
        "PROVIDER, MODEL_FAMILY, MODEL, INPUT_PRICE_PER_M, "
        "INPUT_PRICE_PER_CACHED_M, OUTPUT_PRICE_PER_M, MAX_CONTEXT_LENGTH, "
        "TOKEN_TO_BYTES, TOTAL_PARAMETER_COUNT, ACTIVE_PARAMETER_COUNT\n"
        "openrouter, deepseek, deepseek-v3.2, 0.252, , 0.378, "
        "163840, 4.0, 685000000000, 37000000000\n"
    )

    init_model_cards(str(pricing_dir))
    assert get_model_pricing("openrouter", "deepseek-v3.2", "deepseek", no_create=True) is None
    init_model_pricing(str(pricing_dir))

    pricing = get_model_pricing("openrouter", "deepseek-v3.2", "deepseek", no_create=True)
    card = get_model_card("openrouter", "deepseek-v3.2", "deepseek", no_create=True)
    assert pricing.output_price_per_M == Decimal("0.378")
    assert card.max_context_length == 163840
    assert card.total_parameter_count == 685000000000
    assert card.active_parameter_count == 37000000000


def test_init_skips_incomplete_card_without_blocking_pricing(db0_fixture, pricing_dir):
    (pricing_dir / "combined.csv").write_text(
        "PROVIDER,MODEL,INPUT_PRICE_PER_M,OUTPUT_PRICE_PER_M,"
        "MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES\n"
        "openai,gpt-4o,2.50,10.00,128000,\n"
        "openai,gpt-4o-mini,0.15,0.60,128000,4.0\n"
    )

    init_model_pricing(str(pricing_dir))
    init_model_cards(str(pricing_dir))

    assert get_model_pricing("openai", "gpt-4o", no_create=True) is not None
    assert get_model_card("openai", "gpt-4o", no_create=True) is None
    assert get_model_card("openai", "gpt-4o-mini", no_create=True).token_to_bytes == 4.0


def test_init_skips_invalid_pricing_without_blocking_card(db0_fixture, pricing_dir):
    (pricing_dir / "combined.csv").write_text(
        "PROVIDER,MODEL,INPUT_PRICE_PER_M,OUTPUT_PRICE_PER_M,"
        "MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES\n"
        "openai,gpt-4o,invalid,10.00,128000,4.0\n"
    )

    init_model_pricing(str(pricing_dir))
    init_model_cards(str(pricing_dir))

    assert get_model_pricing("openai", "gpt-4o", no_create=True) is None
    assert get_model_card("openai", "gpt-4o", no_create=True).max_context_length == 128000


def test_init_blank_optional_counts_preserve_initialized_values(db0_fixture, pricing_dir):
    (pricing_dir / "cards.csv").write_text(
        "PROVIDER,MODEL,MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES,"
        "TOTAL_PARAMETER_COUNT,ACTIVE_PARAMETER_COUNT\n"
        "openai,gpt-4o,128000,4.0,,\n"
    )
    from statek.model_card import set_model_card  # pylint: disable=import-outside-toplevel
    card = set_model_card("openai", "gpt-4o", total_parameter_count=100)

    init_model_cards(str(pricing_dir))
    init_model_cards(str(pricing_dir))

    assert card.total_parameter_count == 100
    assert card.max_context_length == 128000


def test_init_model_cards_scans_subdirectories_and_txt(db0_fixture, pricing_dir):
    nested = pricing_dir / "nested"
    nested.mkdir()
    (nested / "cards.txt").write_text(
        "PROVIDER,MODEL,MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES\n"
        "openai,gpt-4o,128000,4.0\n"
    )
    (nested / "ignored.json").write_text(
        '{"PROVIDER":"openai","MODEL":"other"}'
    )

    init_model_cards(str(pricing_dir))

    assert get_model_card("openai", "gpt-4o", no_create=True).max_context_length == 128000
    assert get_model_card("openai", "other", no_create=True) is None
    assert not list(db0.find(ModelPricing))


def test_statek_init_loads_pricing_and_cards(db0_fixture, pricing_dir):
    (pricing_dir / "combined.csv").write_text(
        "PROVIDER,MODEL,INPUT_PRICE_PER_M,OUTPUT_PRICE_PER_M,"
        "MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES\n"
        "openai,gpt-4o,2.50,10.00,128000,4.0\n"
    )

    statek.init(StatekSettings(prompt_defs={}, statek_model_info_dir=str(pricing_dir)))

    pricing = get_model_pricing("openai", "gpt-4o", no_create=True)
    assert pricing.input_price_per_M == Decimal("2.50")
    assert get_model_card("openai", "gpt-4o", no_create=True).max_context_length == 128000
    assert len(list(db0.find(ModelPricing))) == 1
    assert len(list(db0.find(ModelCard))) == 1
