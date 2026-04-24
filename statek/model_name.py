"""Helpers for parsing and routing STATEK model selection strings."""

from collections import namedtuple
from typing import Optional, Union


ModelName = namedtuple("ModelName", ["provider", "model_family", "model"])

_PROVIDERS_REQUIRING_MODEL_FAMILY = {"OPENROUTER"}


def parse_model_name(input: str) -> ModelName:  # pylint: disable=redefined-builtin
    """Parse ``provider/model_family/model`` strings into normalized components."""
    if input is None:
        raise ValueError("model name must be provided")

    value = str(input).strip()
    if not value:
        raise ValueError("model name must not be empty")

    parts = [(part.strip() or None) for part in value.split("/", 2)]
    if len(parts) == 1:
        return ModelName(None, None, parts[0])
    if len(parts) == 2:
        return ModelName(None, parts[0], parts[1])
    return ModelName(parts[0], parts[1], parts[2])


def ensure_model_name(
    value: Union[str, ModelName],
    *,
    model_family: Optional[str] = None,
) -> ModelName:
    """Return a ``ModelName``, applying a legacy model-family fallback when needed."""
    model_name = value if isinstance(value, ModelName) else parse_model_name(value)
    if model_name.model_family is not None or model_family is None:
        return model_name
    return ModelName(model_name.provider, model_family, model_name.model)


def select_model_provider(
    value: Union[str, ModelName],
    *,
    default_provider: Optional[str] = None,
    model_family: Optional[str] = None,
) -> Optional[str]:
    """Return the effective provider, preferring the model string over defaults."""
    if value is None:
        return default_provider
    model_name = ensure_model_name(value, model_family=model_family)
    return model_name.provider or default_provider


def format_model_for_provider(
    value: Union[str, ModelName],
    provider: Optional[str],
    *,
    model_family: Optional[str] = None,
) -> str:
    """Return the provider-specific model identifier to send upstream."""
    model_name = ensure_model_name(value, model_family=model_family)
    if not model_name.model:
        raise ValueError("model name must include a model component")

    provider_key = provider.upper() if provider else None
    if provider_key in _PROVIDERS_REQUIRING_MODEL_FAMILY and model_name.model_family:
        return f"{model_name.model_family}/{model_name.model}"
    return model_name.model
