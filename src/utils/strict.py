"""
Strict utilities for fail-fast execution.

This module provides helpers that enforce the repo's STRICT FAIL-FAST policy:
- No silent fallbacks
- No "best effort" execution
- No "graceful degradation"

If something is missing, ambiguous, or inconsistent, these helpers raise
immediately with a clear, actionable error message.

Usage:
    from utils.strict import require, require_attr, require_key, require_env

See docs/STRICTNESS.md for the full policy.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence, TypeVar, overload

__all__ = [
    # Core assertions
    "require",
    "require_not_none",
    "require_attr",
    "require_key",
    "require_one_of",
    "require_env",
    "require_positive",
    "require_in_range",
    # Exceptions
    "StrictError",
    "ConfigError",
    "LayerResolutionError",
    "ShapeError",
    "InvariantError",
    "MissingKeyError",
    "MissingAttrError",
    "MissingEnvError",
]

T = TypeVar("T")


# =============================================================================
# Custom Exceptions
# =============================================================================


class StrictError(Exception):
    """Base class for all strict-mode exceptions."""

    pass


class ConfigError(StrictError):
    """Raised when configuration is invalid, missing, or inconsistent."""

    def __init__(self, message: str, *, config_name: str | None = None):
        self.config_name = config_name
        if config_name:
            message = f"[{config_name}] {message}"
        super().__init__(message)


class LayerResolutionError(StrictError):
    """Raised when a target layer cannot be resolved in a model."""

    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        requested_layer: str | None = None,
        available_layers: Sequence[str] | None = None,
    ):
        self.model_name = model_name
        self.requested_layer = requested_layer
        self.available_layers = list(available_layers) if available_layers else None

        parts = [message]
        if model_name:
            parts.append(f"Model: {model_name}")
        if requested_layer:
            parts.append(f"Requested layer: {requested_layer}")
        if available_layers:
            parts.append(f"Available layers: {list(available_layers)}")
        parts.append("Fix: Add an explicit mapping in MODEL_LAYER_MAP or use a supported layer.")

        super().__init__("\n".join(parts))


class ShapeError(StrictError):
    """Raised when tensor shapes are invalid, ambiguous, or inconsistent."""

    def __init__(
        self,
        message: str,
        *,
        actual_shape: tuple | None = None,
        expected_shape: tuple | str | None = None,
        tensor_name: str | None = None,
    ):
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape
        self.tensor_name = tensor_name

        parts = [message]
        if tensor_name:
            parts.append(f"Tensor: {tensor_name}")
        if actual_shape is not None:
            parts.append(f"Actual shape: {actual_shape}")
        if expected_shape is not None:
            parts.append(f"Expected shape: {expected_shape}")

        super().__init__("\n".join(parts))


class InvariantError(StrictError):
    """Raised when a program invariant is violated."""

    pass


class MissingKeyError(StrictError, KeyError):
    """Raised when a required key is missing from a mapping."""

    def __init__(self, key: str, mapping_name: str | None = None):
        self.key = key
        self.mapping_name = mapping_name
        msg = f"Missing required key: '{key}'"
        if mapping_name:
            msg = f"{msg} in {mapping_name}"
        super().__init__(msg)


class MissingAttrError(StrictError, AttributeError):
    """Raised when a required attribute is missing from an object."""

    def __init__(self, attr: str, obj_type: str | None = None):
        self.attr = attr
        self.obj_type = obj_type
        msg = f"Missing required attribute: '{attr}'"
        if obj_type:
            msg = f"{msg} on {obj_type}"
        super().__init__(msg)


class MissingEnvError(StrictError, OSError):
    """Raised when a required environment variable is missing."""

    def __init__(self, var_name: str, hint: str | None = None):
        self.var_name = var_name
        msg = f"Missing required environment variable: {var_name}"
        if hint:
            msg = f"{msg}\nHint: {hint}"
        super().__init__(msg)


# =============================================================================
# Core Assertion Functions
# =============================================================================


def require(condition: bool, message: str) -> None:
    """Assert a condition is true, raising InvariantError if not.

    Args:
        condition: The condition to check.
        message: Error message if condition is False.

    Raises:
        InvariantError: If condition is False.

    Example:
        require(len(tokens) == num_patches + 1, f"Expected {num_patches+1} tokens, got {len(tokens)}")
    """
    if not condition:
        raise InvariantError(message)


def require_not_none(value: T | None, name: str) -> T:
    """Assert value is not None, returning it if valid.

    Args:
        value: The value to check.
        name: Name of the value for error message.

    Returns:
        The value, guaranteed non-None.

    Raises:
        InvariantError: If value is None.

    Example:
        model = require_not_none(config.model, "config.model")
    """
    if value is None:
        raise InvariantError(f"{name} must not be None")
    return value


def require_attr(obj: Any, attr: str, *, context: str | None = None) -> Any:
    """Get a required attribute from an object, raising if missing.

    Args:
        obj: The object to get the attribute from.
        attr: The attribute name.
        context: Optional context for error message.

    Returns:
        The attribute value.

    Raises:
        MissingAttrError: If the attribute doesn't exist.

    Example:
        embed_dim = require_attr(model, "embed_dim", context="ViT model")
    """
    if not hasattr(obj, attr):
        obj_type = type(obj).__name__
        if context:
            obj_type = f"{obj_type} ({context})"
        raise MissingAttrError(attr, obj_type)
    return getattr(obj, attr)


def require_key(mapping: Mapping[str, T], key: str, *, context: str | None = None) -> T:
    """Get a required key from a mapping, raising if missing.

    Args:
        mapping: The mapping to get the key from.
        key: The key to retrieve.
        context: Optional context for error message.

    Returns:
        The value for the key.

    Raises:
        MissingKeyError: If the key doesn't exist.

    Example:
        lr = require_key(config, "learning_rate", context="training config")
    """
    if key not in mapping:
        raise MissingKeyError(key, context)
    return mapping[key]


def require_one_of(value: T, allowed: Sequence[T], name: str) -> T:
    """Assert value is one of the allowed values.

    Args:
        value: The value to check.
        allowed: Sequence of allowed values.
        name: Name of the value for error message.

    Returns:
        The value, guaranteed to be in allowed.

    Raises:
        InvariantError: If value is not in allowed.

    Example:
        variant = require_one_of(cam_variant, ["gradcam", "layercam"], "cam_variant")
    """
    if value not in allowed:
        raise InvariantError(f"{name} must be one of {list(allowed)}, got: {value!r}")
    return value


def require_env(var_name: str, *, hint: str | None = None) -> str:
    """Get a required environment variable, raising if missing.

    Args:
        var_name: The environment variable name.
        hint: Optional hint for how to set the variable.

    Returns:
        The environment variable value.

    Raises:
        MissingEnvError: If the variable is not set or empty.

    Example:
        data_root = require_env("CFT_DATA_ROOT", hint="Set to your data directory")
    """
    value = os.environ.get(var_name)
    if not value:
        raise MissingEnvError(var_name, hint)
    return value


def require_positive(value: int | float, name: str) -> int | float:
    """Assert value is positive (> 0).

    Args:
        value: The value to check.
        name: Name of the value for error message.

    Returns:
        The value, guaranteed positive.

    Raises:
        InvariantError: If value <= 0.
    """
    if value <= 0:
        raise InvariantError(f"{name} must be positive, got: {value}")
    return value


def require_in_range(
    value: int | float,
    lo: int | float,
    hi: int | float,
    name: str,
    *,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> int | float:
    """Assert value is within a range.

    Args:
        value: The value to check.
        lo: Lower bound.
        hi: Upper bound.
        name: Name of the value for error message.
        lo_inclusive: Whether lower bound is inclusive (default True).
        hi_inclusive: Whether upper bound is inclusive (default True).

    Returns:
        The value, guaranteed in range.

    Raises:
        InvariantError: If value is outside range.
    """
    lo_ok = value >= lo if lo_inclusive else value > lo
    hi_ok = value <= hi if hi_inclusive else value < hi

    if not (lo_ok and hi_ok):
        lo_bracket = "[" if lo_inclusive else "("
        hi_bracket = "]" if hi_inclusive else ")"
        raise InvariantError(f"{name} must be in {lo_bracket}{lo}, {hi}{hi_bracket}, got: {value}")

    return value


# =============================================================================
# Optional Helpers (explicit optional handling)
# =============================================================================


def get_optional_env(var_name: str) -> str | None:
    """Get an optional environment variable.

    Unlike require_env, this returns None if the variable is not set.
    Use this ONLY for truly optional debug/override variables.

    Args:
        var_name: The environment variable name.

    Returns:
        The value or None if not set.
    """
    value = os.environ.get(var_name)
    return value if value else None
