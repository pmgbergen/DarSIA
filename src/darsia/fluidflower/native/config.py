"""Explicit parameter documents for camera and radiometric processing.

Numerical functions receive mappings directly. The context accessor exists
only for retained entry points whose historical signature had no parameters.
It never supplies method or camera values of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from .io import read_bound_json

_configured: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "native_parameter_document", default=None
)


def load_parameters(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Load a content-bound parameter object, preserving JSON numeric types."""
    return read_bound_json(path, expected_sha256)


def configure_parameters(configuration: Mapping[str, Any]) -> Token:
    """Bind an explicit document to this context for retained entry points."""
    if not isinstance(configuration, Mapping):
        raise TypeError("Native configuration must be a mapping")
    return _configured.set(configuration)


def reset_parameters(token: Token) -> None:
    """Restore the preceding parameter context after a compatibility call."""
    _configured.reset(token)


@lru_cache(maxsize=8)
def _environment_document(path: str, digest: str) -> dict[str, Any]:
    return load_parameters(path, digest)


def require_configuration(
    *, configuration: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Return an explicit context or content-bound environment document.

    Missing configuration is an error. An environment document is verified
    once per path/digest pair, rather than reread by numerical functions.
    """
    document = configuration if configuration is not None else _configured.get()
    if document is None:
        path = os.environ.get("DARSIA_NATIVE_CONFIGURATION")
        digest = os.environ.get("DARSIA_NATIVE_CONFIGURATION_SHA256")
        if not path or not digest:
            raise ValueError("An explicit native parameter document is required")
        document = _environment_document(path, digest)
    return document


def require_parameters(
    section: str, *, configuration: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Select a required mapping section, without supplying method defaults."""
    document = require_configuration(configuration=configuration)
    value = document.get(section)
    if not isinstance(value, Mapping):
        raise ValueError("Native parameter section is absent: " + section)
    return value
