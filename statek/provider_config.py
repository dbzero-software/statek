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

"""Durable provider-specific mappings for Statek model parameters."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import dbzero as db0


_PROVIDER_CONFIG_HASH_TAG_PREFIX = "STATEK_PROVIDER_CONFIG:H:"


@db0.memo
@dataclass
class ProviderConfig:
    """Durable snapshot of provider-specific reasoning payload and conflict mappings."""

    provider_config: Dict[str, Any]

    def __post_init__(self) -> None:
        """Snapshot a plain mapping so callers cannot mutate this durable configuration."""
        if not _is_mapping(self.provider_config):
            raise ValueError("provider_config must be an object")
        self.provider_config = _plain_json_value(self.provider_config)

    def find_payload(self, *args: Optional[str], **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Find the deepest matching provider payload for a model-parameter query.

        ``args`` identify the configured provider, optional model family, and
        model. The only current query parameter is ``reasoning_level``.
        """
        if set(kwargs) != {"reasoning_level"}:
            raise ValueError("Unsupported provider configuration query")
        reasoning_level = _reasoning_level(kwargs["reasoning_level"])

        for path in _candidate_paths(args):
            node = _find_path_node(self.provider_config, path)
            if node is None:
                continue
            reasoning = _reasoning_config(node)
            if reasoning is None:
                continue
            payload = _matching_payload(reasoning, reasoning_level)
            if payload is not None:
                return payload
        return None

    def find_ignored_parameters(self, *args: Optional[str]) -> Optional[List[str]]:
        """Return the deepest explicit reasoning parameter ignore list for a model path."""
        for path in _candidate_paths(args):
            node = _find_path_node(self.provider_config, path)
            if node is None:
                continue
            reasoning = _reasoning_config(node)
            if reasoning is None or "ignore_parameters" not in reasoning:
                continue
            return _ignored_parameters(reasoning["ignore_parameters"])
        return None


def resolve_provider_config(provider_config: Mapping[str, Any]) -> ProviderConfig:
    """Reuse or persist a durable provider-configuration snapshot by exact content."""
    if not _is_mapping(provider_config):
        raise ValueError("provider_config must be a mapping")
    snapshot = _plain_json_value(provider_config)
    identity_tag = _provider_config_identity_tag(snapshot)

    for existing in db0.find(ProviderConfig, identity_tag):  # pylint: disable=no-member
        if _has_same_content(existing, snapshot):
            return existing

    for existing in db0.find(ProviderConfig):  # pylint: disable=no-member
        if _has_same_content(existing, snapshot):
            db0.tags(existing).add(identity_tag)  # pylint: disable=no-member
            return existing

    resolved_config = ProviderConfig(snapshot)
    db0.tags(resolved_config).add(identity_tag)  # pylint: disable=no-member
    return resolved_config


def load_provider_config(path: str) -> Dict[str, Any]:
    """Load a provider configuration JSON object from a UTF-8 file."""
    with open(path, encoding="utf-8") as config_file:
        provider_config = json.load(config_file)
    if not _is_mapping(provider_config):
        raise ValueError("Provider configuration JSON must contain a top-level mapping")
    return _plain_json_value(provider_config)


def resolve_settings_provider_config(settings: Any) -> Optional[ProviderConfig]:
    """Resolve the provider configuration selected by a settings snapshot."""
    config_path = getattr(settings, "statek_provider_config", None)
    if config_path is None:
        return None
    return resolve_provider_config(load_provider_config(config_path))


def provider_config_identity(provider_config: Optional[ProviderConfig]) -> Optional[str]:
    """Return content identity for a durable provider configuration snapshot."""
    if provider_config is None:
        return None
    return _provider_config_identity_tag(_plain_json_value(provider_config.provider_config))


def provider_configs_match(
    first: Optional[ProviderConfig],
    second: Optional[ProviderConfig],
) -> bool:
    """Return whether two optional provider snapshots have equal full content."""
    if first is None or second is None:
        return first is second
    return _has_same_content(first, _plain_json_value(second.provider_config))


def _matching_payload(config: Mapping[str, Any], reasoning_level: int) -> Optional[Dict[str, Any]]:
    """Return the first payload matching a reasoning level at one config hierarchy level."""
    definitions = config.get("reasoning_level")
    if definitions is None:
        return None
    if not _is_sequence(definitions):
        raise ValueError("Provider configuration reasoning_level must be a sequence")

    for definition in definitions:
        if not _is_mapping(definition):
            raise ValueError("Provider configuration reasoning-level entry must be a mapping")
        range_config = definition.get("range")
        payload = definition.get("payload")
        if not _is_mapping(range_config) or not _is_mapping(payload):
            raise ValueError(
                "Provider configuration reasoning-level entry requires range and payload objects"
            )

        lower_bound = _reasoning_level(range_config.get("from", 0))
        upper_bound = _reasoning_level(range_config.get("to", 100))
        if lower_bound > upper_bound:
            raise ValueError("Provider configuration reasoning-level range from must not exceed to")
        if lower_bound <= reasoning_level <= upper_bound:
            return deepcopy(_plain_json_value(payload))
    return None


def _reasoning_config(config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the canonical nested reasoning configuration at one hierarchy node."""
    if "reasoning" not in config:
        return None
    reasoning = config["reasoning"]
    if not _is_mapping(reasoning):
        raise ValueError("Provider configuration reasoning must be a mapping")
    return reasoning


def _ignored_parameters(parameters: Any) -> List[str]:
    """Validate and copy a reasoning parameter ignore list."""
    error_message = (
        "Provider configuration reasoning ignore_parameters must be a sequence of strings"
    )
    if not _is_sequence(parameters):
        raise ValueError(error_message)
    copied_parameters = list(parameters)
    if any(not isinstance(parameter, str) for parameter in copied_parameters):
        raise ValueError(error_message)
    return copied_parameters


def _is_mapping(value: Any) -> bool:
    """Return whether a regular or dbzero persistent value exposes mapping items."""
    return isinstance(value, Mapping) or callable(getattr(value, "items", None))


def _is_sequence(value: Any) -> bool:
    """Return whether a regular or dbzero persistent value is a JSON array."""
    return (
        not isinstance(value, (str, bytes))
        and not _is_mapping(value)
        and callable(getattr(value, "__iter__", None))
    )


def _plain_json_value(value: Any) -> Any:
    """Copy a JSON-like regular or dbzero persistent value into plain Python objects."""
    if _is_mapping(value):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if _is_sequence(value):
        return [_plain_json_value(item) for item in value]
    return value


def _provider_config_identity_tag(config: Mapping[str, Any]) -> str:
    """Return the namespaced dbzero lookup tag for configuration content."""
    return f"{_PROVIDER_CONFIG_HASH_TAG_PREFIX}{_provider_config_identity_hash(config)}"


def _provider_config_identity_hash(config: Mapping[str, Any]) -> str:
    """Return the cryptographic identity digest for canonical configuration content."""
    return hashlib.sha256(_canonical_provider_config(config).encode("utf-8")).hexdigest()


def _canonical_provider_config(config: Mapping[str, Any]) -> str:
    """Serialize JSON-compatible configuration content in a deterministic form."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _has_same_content(provider_config: ProviderConfig, config: Mapping[str, Any]) -> bool:
    """Return whether a tagged durable configuration has the requested canonical content."""
    return (
        _canonical_provider_config(_plain_json_value(provider_config.provider_config))
        == _canonical_provider_config(config)
    )


def _reasoning_level(value: Any) -> int:
    """Return a validated Statek reasoning level from an integer or numeric string."""
    if isinstance(value, bool):
        raise ValueError("reasoning_level must be an integer from 0 through 100")
    if isinstance(value, str):
        if not value.strip() or not value.strip().lstrip("+-").isdigit():
            raise ValueError("reasoning_level must be an integer from 0 through 100")
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError("reasoning_level must be an integer from 0 through 100")
    return value


def _find_child(config: Mapping[str, Any], path_part: str) -> Optional[Mapping[str, Any]]:
    """Return a case-insensitive matching nested provider configuration mapping."""
    if path_part in config and _is_mapping(config[path_part]):
        return config[path_part]
    normalized_path_part = path_part.casefold()
    for key, value in config.items():
        if isinstance(key, str) and key.casefold() == normalized_path_part:
            if not _is_mapping(value):
                raise ValueError(f"Provider configuration for {path_part!r} must be a mapping")
            return value
    return None


def _candidate_paths(args: Tuple[Optional[str], ...]) -> List[Tuple[str, ...]]:
    """Return provider-config paths ordered from most to least specific."""
    path = tuple(str(part) for part in args if part is not None)
    if not path:
        return [path]
    candidates = [path]
    if len(path) == 3:
        candidates.extend([
            (path[0], path[2]),
            path[:2],
            path[:1],
        ])
    elif len(path) == 2:
        candidates.append(path[:1])
    return list(dict.fromkeys(candidates))


def _find_path_node(
    root: Mapping[str, Any],
    path: Tuple[str, ...],
) -> Optional[Mapping[str, Any]]:
    """Return the exact configuration node for one candidate path."""
    current_node = root
    for path_part in path:
        current_node = _find_child(current_node, path_part)
        if current_node is None:
            return None
    return current_node
