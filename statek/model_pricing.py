# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ModelPricing — provider/model pricing details for LLM cost tracking."""

# pylint: disable=no-member

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

import dbzero as db0

from ._model_tags import model_tags
from ._model_info import iter_model_info_rows


def _pricing_tags(
    provider: str, model: str, model_family: Optional[str] = None
) -> list:
    return model_tags(provider, model, model_family)


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
    model_iter = db0.find(ModelPricing, *tags)
    existing = next(iter(model_iter), None)
    if existing is not None:
        return existing
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


def _parse_optional_decimal(value: Optional[str]) -> Optional[Decimal]:
    stripped = (value or "").strip()
    if not stripped:
        return None
    return Decimal(stripped)


def _load_pricing_row(row: dict, provider: str, model: str,
                      model_family: Optional[str]) -> None:
    if "INPUT_PRICE_PER_M" not in row or "OUTPUT_PRICE_PER_M" not in row:
        return
    try:
        input_price = _parse_optional_decimal(row["INPUT_PRICE_PER_M"])
        output_price = _parse_optional_decimal(row["OUTPUT_PRICE_PER_M"])
        cached_price = _parse_optional_decimal(row.get("INPUT_PRICE_PER_CACHED_M"))
    except InvalidOperation:
        return
    if input_price is not None and output_price is not None:
        set_model_pricing(
            provider, model,
            input_price_per_M=input_price,
            output_price_per_M=output_price,
            input_price_per_cached_M=cached_price,
            model_family=model_family,
        )


def init_model_pricing(model_info_dir: str) -> None:
    """Load pricing from CSV or TXT files in a directory tree."""
    for provider, model, model_family, row in iter_model_info_rows(model_info_dir):
        _load_pricing_row(row, provider, model, model_family)
