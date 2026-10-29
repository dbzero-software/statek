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

"""Persistent model properties, initialized once per field."""

# pylint: disable=no-member

from dataclasses import dataclass, fields
import math
from typing import Optional

import dbzero as db0

from ._model_tags import model_tags
from ._model_info import iter_model_info_rows


@db0.memo
@dataclass
class ModelCard:
    """Inherent model properties used by the framework."""

    max_context_length: int = 0
    token_to_bytes: float = 0.0
    total_parameter_count: Optional[int] = None
    active_parameter_count: Optional[int] = None


def get_model_card(
    provider: str,
    model: str,
    model_family: Optional[str] = None,
    no_create: bool = False,
    **kwargs,
) -> Optional[ModelCard]:
    """Find a tagged card, creating one when allowed and absent."""
    tags = model_tags(provider, model, model_family)
    existing = next(iter(db0.find(ModelCard, *tags)), None)
    if existing is not None or no_create:
        return existing
    card = ModelCard(**kwargs)
    db0.tags(card).add(tags)
    return card


def set_model_card(
    provider: str,
    model: str,
    model_family: Optional[str] = None,
    **kwargs,
) -> ModelCard:
    """Set only fields that still have their default value or None."""
    existing = get_model_card(provider, model, model_family, no_create=True)
    if existing is None:
        card = get_model_card(provider, model, model_family, **kwargs)
        assert card is not None
        return card

    valid_fields = {field.name for field in fields(ModelCard)}
    for name, value in kwargs.items():
        if name not in valid_fields:
            raise TypeError(f"Unexpected ModelCard field: {name}")
        current = getattr(existing, name)
        if current is not None and current != 0 and current != value:
            raise ValueError(f"ModelCard.{name} is already initialized")

    for name, value in kwargs.items():
        if getattr(existing, name) != value:
            setattr(existing, name, value)
    return existing


def _load_card_row(row: dict, provider: str, model: str,
                   model_family: Optional[str]) -> None:
    if "MAX_CONTEXT_LENGTH" not in row or "TOKEN_TO_BYTES" not in row:
        return
    try:
        max_context_length = int(row["MAX_CONTEXT_LENGTH"])
        token_to_bytes = float(row["TOKEN_TO_BYTES"])
        total_count = row.get("TOTAL_PARAMETER_COUNT")
        active_count = row.get("ACTIVE_PARAMETER_COUNT")
        total_count = int(total_count) if total_count and total_count.strip() else None
        active_count = int(active_count) if active_count and active_count.strip() else None
    except (ValueError, TypeError):
        return
    if max_context_length <= 0 or not math.isfinite(token_to_bytes) or token_to_bytes <= 0:
        return
    if any(count is not None and count <= 0 for count in (total_count, active_count)):
        return
    card_values = {
        "max_context_length": max_context_length,
        "token_to_bytes": token_to_bytes,
    }
    if total_count is not None:
        card_values["total_parameter_count"] = total_count
    if active_count is not None:
        card_values["active_parameter_count"] = active_count
    set_model_card(provider, model, model_family=model_family, **card_values)


def init_model_cards(model_info_dir: str) -> None:
    """Load model cards from CSV or TXT files in a directory tree."""
    for provider, model, model_family, row in iter_model_info_rows(model_info_dir):
        _load_card_row(row, provider, model, model_family)
