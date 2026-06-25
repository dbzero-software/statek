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

import csv
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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


def _parse_optional_decimal(value: str) -> Optional[Decimal]:
    stripped = value.strip()
    if not stripped:
        return None
    return Decimal(stripped)


def _load_pricing_csv(file_path: str) -> None:
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                provider = row["PROVIDER"].strip()
                model = row["MODEL"].strip()
                model_family = row["MODEL_FAMILY"].strip() or None
                input_price = _parse_optional_decimal(row["INPUT_PRICE_PER_M"])
                cached_price = _parse_optional_decimal(row["INPUT_PRICE_PER_CACHED_M"])
                output_price = _parse_optional_decimal(row["OUTPUT_PRICE_PER_M"])
                if input_price is None or output_price is None:
                    continue
                set_model_pricing(
                    provider, model,
                    input_price_per_M=input_price,
                    output_price_per_M=output_price,
                    input_price_per_cached_M=cached_price,
                    model_family=model_family,
                )
            except (KeyError, InvalidOperation):
                continue


def init_model_pricing(model_info_dir: str) -> None:
    for root, _dirs, files in os.walk(model_info_dir):
        for name in files:
            if name.endswith(".csv") or name.endswith(".txt"):
                _load_pricing_csv(os.path.join(root, name))
