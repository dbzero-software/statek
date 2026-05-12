"""Multi-source Pydantic settings support."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable

from botocore.exceptions import BotoCoreError, ClientError
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


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


class AwsSecretsManagerSource(SettingValuesSource):
    """
    Configuration source that loads JSON settings from AWS Secrets Manager.
    """

    def __init__(
        self,
        secret_id: str,
        source_fields: Iterable[str] | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        super().__init__(source_fields=source_fields)
        self.secret_id = secret_id
        self._client = client
        self._values: dict[str, Any] | None = None

    def _get_client(self):
        if self._client is None:
            import boto3  # pylint: disable=import-outside-toplevel

            self._client = boto3.client("secretsmanager")

        return self._client

    def _load_values(self) -> dict[str, Any]:
        if self._values is not None:
            return self._values

        self._values = {}

        try:
            response = self._get_client().get_secret_value(SecretId=self.secret_id)
        except (BotoCoreError, ClientError) as error:
            log.error("Failed to load AWS Secrets Manager secret %s: %s", self.secret_id, error)
            return self._values

        secret_string = response.get("SecretString")
        if not secret_string:
            log.error("AWS Secrets Manager secret %s does not contain SecretString", self.secret_id)
            return self._values

        try:
            values = json.loads(secret_string)
        except json.JSONDecodeError as error:
            log.error("AWS Secrets Manager secret %s is not valid JSON: %s", self.secret_id, error)
            return self._values

        if not isinstance(values, dict):
            log.error("AWS Secrets Manager secret %s JSON must be an object", self.secret_id)
            return self._values

        if self.source_fields is None:
            self._values = values
        else:
            self._values = {
                field_name: values[field_name]
                for field_name in self.source_fields
                if field_name in values
            }

        return self._values

    def get_value(self, field_name: str) -> Any | None:
        """
        Load a field value from AWS Secrets Manager.
        """
        if not self._should_check_field(field_name):
            return None

        return self._load_values().get(field_name)


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
