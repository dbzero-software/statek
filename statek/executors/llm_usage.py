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

"""LLM_Usage — per-job LLM usage metrics with cost resolution."""

# pylint: disable=no-member

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import dbzero as db0

from statek.model_pricing import ModelPricing, get_model_pricing


@db0.memo(no_default_tags=True)
@dataclass
class LLM_Usage:
    """Usage-related pricing object (might be empty)"""
    pricing: ModelPricing
    """Total context bytes used by this job so far"""
    context_bytes: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_input_tokens: int = 0
    """This number is already included in total_input_tokens"""
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    """Total cost as reported by the LLM API provider (where available)"""
    total_reported_cost: Optional[float] = None

    @property
    def total_cost(self) -> Optional[float]:
        if (self.pricing.is_valid
                and self.total_input_tokens + self.total_output_tokens > 0):
            pricing = self.pricing
            non_cached = self.total_input_tokens - self.total_cached_tokens
            cached_price = (
                pricing.input_price_per_cached_M
                if pricing.input_price_per_cached_M is not None
                else pricing.input_price_per_M
            )
            return float(
                (Decimal(non_cached) * pricing.input_price_per_M
                 + Decimal(self.total_cached_tokens) * cached_price
                 + Decimal(self.total_output_tokens) * pricing.output_price_per_M)
                / Decimal("1000000")
            )
        return self.total_reported_cost

    @staticmethod
    def for_model(
        provider: str, model: str, model_family: Optional[str] = None
    ) -> "LLM_Usage":
        pricing = get_model_pricing(provider, model, model_family)
        return LLM_Usage(pricing=pricing)
