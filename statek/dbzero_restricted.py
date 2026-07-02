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

"""dbzero restricted-mode integration helpers for Statek."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

import dbzero as db0

from statek.settings import StatekSettings, get_statek_settings


class DbzeroRestrictedModeError(RuntimeError):
    """Raised when Statek restricted mode is not backed by dbzero restricted mode."""


_DBZERO_RESTRICTED_CONTEXT: ContextVar[bool] = ContextVar(
    "statek_dbzero_restricted",
    default=False,
)


@contextmanager
def llm_dbzero_restricted_context():
    """Temporarily enable dbzero restricted mode for LLM-authored code."""
    token = _DBZERO_RESTRICTED_CONTEXT.set(True)
    try:
        yield
    finally:
        _DBZERO_RESTRICTED_CONTEXT.reset(token)


@contextmanager
def as_unrestricted() -> Iterator[None]:
    """Temporarily disable dbzero restricted mode within the current context."""
    token = _DBZERO_RESTRICTED_CONTEXT.set(False)
    try:
        yield
    finally:
        _DBZERO_RESTRICTED_CONTEXT.reset(token)


def _statek_restricted(settings: Optional[StatekSettings] = None) -> bool:
    resolved = settings or get_statek_settings()
    mode = getattr(resolved, "python_sandbox_mode", "restricted")
    return not (isinstance(mode, str) and mode.lower() == "off")


def _dbzero_not_initialized(exc: RuntimeError) -> bool:
    return "dbzero not initialized" in str(exc).lower()


def _dbzero_config() -> Optional[dict[str, Any]]:
    try:
        return db0.get_config()  # pylint: disable=no-member
    except RuntimeError as exc:
        if _dbzero_not_initialized(exc):
            return None
        raise
    except AttributeError as exc:
        raise DbzeroRestrictedModeError(
            "Statek restricted mode requires a dbzero version that exposes get_config()."
        ) from exc


def configure_dbzero_restricted_context(settings: Optional[StatekSettings] = None) -> None:
    """Configure dbzero to use Statek's dynamic restricted-mode context."""
    if not _statek_restricted(settings):
        return

    config = _dbzero_config()
    if config is None:
        raise DbzeroRestrictedModeError(
            "Statek restricted mode requires dbzero to be initialized before statek.init()."
        )
    if config.get("restricted", False):
        return

    try:
        db0.set_restricted(  # pylint: disable=no-member
            restricted_context=_DBZERO_RESTRICTED_CONTEXT
        )
    except AttributeError as exc:
        raise DbzeroRestrictedModeError(
            "Statek restricted mode requires dbzero >= 0.4.2 with set_restricted()."
        ) from exc


def dbzero_restricted_enabled() -> Optional[bool]:
    """Return dbzero global restricted mode, or None when dbzero is not initialized."""
    config = _dbzero_config()
    if config is None:
        return None
    return bool(config.get("restricted", False))


def validate_dbzero_restricted(settings: Optional[StatekSettings] = None) -> None:
    """Validate/configure dbzero restricted mode when Statek is restricted."""
    configure_dbzero_restricted_context(settings)


def validate_current_prefix_restricted(settings: Optional[StatekSettings] = None) -> None:
    """Validate the current dbzero prefix when Statek uses static dbzero restriction."""
    if not _statek_restricted(settings):
        return
    if dbzero_restricted_enabled() is False:
        return
    try:
        stats = db0.get_prefix_stats()  # pylint: disable=no-member
    except RuntimeError as exc:
        if _dbzero_not_initialized(exc):
            return
        raise
    except AttributeError as exc:
        raise DbzeroRestrictedModeError(
            "Statek restricted mode requires a dbzero version that exposes "
            "get_prefix_stats()."
        ) from exc

    if not stats.get("restricted", False):
        prefix_name = stats.get("name", "<unknown>")
        raise DbzeroRestrictedModeError(
            f"Statek restricted mode requires dbzero prefix {prefix_name!r} "
            "to be opened with restricted=True."
        )


def open_prefix(
    prefix_name: str,
    open_mode: str = "rw",
    *,
    restricted: Optional[bool] = None,
    settings: Optional[StatekSettings] = None,
    **kwargs: Any,
) -> None:
    """Open a dbzero prefix with Statek's restricted-mode contract enforced."""
    statek_restricted = _statek_restricted(settings)
    if statek_restricted:
        if restricted is False:
            raise DbzeroRestrictedModeError(
                "Statek restricted mode cannot open dbzero prefixes with restricted=False."
            )
    elif restricted is not None:
        kwargs["restricted"] = restricted

    try:
        db0.open(prefix_name, open_mode, **kwargs)  # pylint: disable=no-member
    except TypeError as exc:
        if statek_restricted and "restricted" in str(exc):
            raise DbzeroRestrictedModeError(
                "Statek restricted mode requires a dbzero version that supports "
                "open(..., restricted=True)."
            ) from exc
        raise
    validate_current_prefix_restricted(settings)
