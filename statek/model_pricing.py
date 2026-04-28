"""ModelPricing — provider/model pricing details for LLM cost tracking."""

# pylint: disable=no-member

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import dbzero as db0


def _normalize_tag(tag: str) -> str:
    return "".join(tag.split()).upper()


def _pricing_tags(
    provider: str, model: str, model_family: Optional[str] = None
) -> list:
    tags = [_normalize_tag(provider), _normalize_tag(model)]
    if model_family is not None:
        tags.append(_normalize_tag(model_family))
    return tags


def _create_pricing(
    provider: str,
    model: str,
    model_family: Optional[str] = None,
    input_price_per_M: Optional[Decimal] = None,
    output_price_per_M: Optional[Decimal] = None,
    input_price_per_cached_M: Optional[Decimal] = None,
) -> "ModelPricing":
    pricing = ModelPricing(
        input_price_per_M=input_price_per_M,
        output_price_per_M=output_price_per_M,
        input_price_per_cached_M=input_price_per_cached_M,
    )
    db0.tags(pricing).add(_pricing_tags(provider, model, model_family) + ["USAGE"])
    return pricing


@db0.memo
@dataclass
class ModelPricing:
    """price per 1M of non-cached input tokens"""
    input_price_per_M: Optional[Decimal] = None
    """price per 1M cached input tokens"""
    input_price_per_cached_M: Optional[Decimal] = None
    output_price_per_M: Optional[Decimal] = None

    @property
    def is_valid(self) -> bool:
        """If at least input_price_per_M and output_price_per_M is defined"""
        return self.input_price_per_M is not None and self.output_price_per_M is not None


def get_model_pricing(
    provider: str,
    model: str,
    model_family: Optional[str] = None,
    no_create: bool = False,
) -> Optional[ModelPricing]:
    tags = _pricing_tags(provider, model, model_family)
    print(f"Looking for pricing with tags: {tags}")
    model_iter = db0.find(ModelPricing, 'OPENROUTER')
    for existing in model_iter:
        return existing
    # if existing is not None:
    #     return existing
    if no_create:
        return None
    return _create_pricing(provider, model, model_family)


def set_model_pricing(
    provider: str,
    model: str,
    input_price_per_M: Decimal,
    output_price_per_M: Decimal,
    input_price_per_cached_M: Optional[Decimal] = None,
    model_family: Optional[str] = None,
) -> ModelPricing:
    existing = get_model_pricing(provider, model, model_family, no_create=True)

    if existing is not None:
        if not existing.is_valid:
            existing.input_price_per_M = input_price_per_M
            existing.output_price_per_M = output_price_per_M
            existing.input_price_per_cached_M = input_price_per_cached_M
            return existing
        if (
            existing.input_price_per_M == input_price_per_M
            and existing.output_price_per_M == output_price_per_M
            and existing.input_price_per_cached_M == input_price_per_cached_M
        ):
            return existing

    return _create_pricing(
        provider, model, model_family,
        input_price_per_M=input_price_per_M,
        output_price_per_M=output_price_per_M,
        input_price_per_cached_M=input_price_per_cached_M,
    )
