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

"""Multi-source Pydantic settings support."""

from abc import ABC, abstractmethod
from typing import Any, Iterable

from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingValuesSource(ABC):
    """
    Base interface for external settings sources.
    """

    def __init__(self, source_fields: Iterable[str] | None = None) -> None:
        self.source_fields = set(source_fields) if source_fields is not None else None

    def _should_check_field(self, field_name: str) -> bool:
        if self.source_fields is None:
            return True

        return field_name in self.source_fields

    @abstractmethod
    def get_value(self, field_name: str) -> Any | None:
        """
        Return value for a field, or None when value is not available.
        """
        raise NotImplementedError


class MultiSourceBaseSettings(BaseSettings):
    """
    BaseSettings with support for ordered external sources.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(
        self,
        *,
        sources: Iterable[SettingValuesSource] | None = None,
        **values: Any,
    ) -> None:
        source_data: dict[str, Any] = {}
        sources = list(sources or [])
        # Pydantic blocks normal assignment for private runtime attributes before init.
        object.__setattr__(self, "_settings_sources", sources)

        for field_name in self.__class__.model_fields.keys():
            if field_name in values:
                continue

            value = self.get_value_from_sources(field_name)
            if value is not None:
                source_data[field_name] = value

        super().__init__(**{**source_data, **values})
        # Pydantic init resets private attrs, so keep the configured sources for later lookups.
        object.__setattr__(self, "_settings_sources", sources)

    def get_value_from_sources(self, field_name: str) -> Any | None:
        """
        Return the first available value for a field from configured sources.
        """
        for source in self._settings_sources:
            value = source.get_value(field_name)

            if value is not None:
                return value

        return None
