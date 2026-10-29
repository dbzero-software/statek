"""Shared tag normalization for model metadata."""

from typing import Optional


def model_tags(provider: str, model: str, model_family: Optional[str] = None) -> list:
    """Build normalized provider, model, and optional family tags."""
    tags = ["".join(value.split()).upper() for value in (provider, model)]
    if model_family is not None:
        tags.append("".join(model_family.split()).upper())
    return tags
