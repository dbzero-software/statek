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

"""Parse incremental extra resource declarations for difficulty levels L/M/H."""

import re
from collections.abc import Mapping
from typing import TypeAlias


ExtraResources: TypeAlias = tuple[list[str] | None, list[str] | None, list[str] | None]

_RESOURCE_NAME = r"[^\s(),:{}\[\]'\"\\]+"
_RESOURCE_GROUP = re.compile(
    rf"(?P<levels>[LMH]+)\s*:\s*"
    rf"(?P<names>{_RESOURCE_NAME}(?:\s*,\s*(?!{_RESOURCE_NAME}\s*:){_RESOURCE_NAME})*)",
    re.IGNORECASE,
)


def parse_extra_resources(input: str) -> ExtraResources:  # pylint: disable=redefined-builtin
    """Parse prompt metadata into incremental resource lists for L, M and H.

    Round parentheses around individual groups are optional, for example
    ``(LM:assistant,researcher),(H:helpdesk)`` or
    ``LM:assistant,researcher,H:helpdesk``. Selectors are case-insensitive;
    resource names are case-sensitive unquoted tokens. Empty input returns
    three None values.

    Each resource is stored only at its lowest declared difficulty, since it
    is inherited at higher levels. Order within that lowest-level bucket is
    retained. Parsing neither resolves agents nor executes expressions.

    Raises:
        ValueError: If the declaration has invalid selectors, names or grouping.
    """
    remaining = input.strip()
    buckets: list[list[str]] = [[], [], []]
    while remaining:
        parenthesized = remaining.startswith("(")
        if parenthesized:
            remaining = remaining[1:].lstrip()
        match = _RESOURCE_GROUP.match(remaining)
        if match is None:
            raise ValueError(f"Invalid EXTRA_RESOURCES declaration near {remaining!r}")

        level = min("LMH".index(label) for label in match["levels"].upper())
        buckets[level].extend(name.strip() for name in match["names"].split(","))
        remaining = remaining[match.end():].lstrip()
        if parenthesized:
            if not remaining.startswith(")"):
                raise ValueError("Invalid EXTRA_RESOURCES: expected closing ')' after group")
            remaining = remaining[1:].lstrip()
        if remaining:
            if not remaining.startswith(","):
                raise ValueError(f"Invalid EXTRA_RESOURCES separator near {remaining!r}")
            remaining = remaining[1:].lstrip()
            if not remaining:
                raise ValueError("Invalid EXTRA_RESOURCES: trailing comma")

    return normalize_extra_resources((buckets[0], buckets[1], buckets[2]))


def normalize_extra_resources(resources: ExtraResources | None) -> ExtraResources:
    """Copy resource lists and retain each name only at its lowest level."""
    seen: set[str] = set()
    result: list[list[str] | None] = []
    for bucket in resources or (None, None, None):
        additions = []
        for name in bucket or ():
            if name not in seen:
                seen.add(name)
                additions.append(name)
        result.append(additions or None)
    return result[0], result[1], result[2]


def extra_resources_from_metadata(metadata: Mapping[str, str] | None) -> ExtraResources:
    """Prepare prompt resources once for both definition lookup and creation."""
    declaration = metadata.get("EXTRA_RESOURCES") if metadata else None
    return parse_extra_resources(declaration) if declaration is not None else (None, None, None)


def extra_resources_identity(resources: ExtraResources | None) -> tuple[tuple[str, ...], ...]:
    """Return the same ordered value identity for Python and persisted lists."""
    return tuple(tuple(bucket or ()) for bucket in normalize_extra_resources(resources))
