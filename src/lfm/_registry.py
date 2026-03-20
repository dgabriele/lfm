"""Global component registry for the LFM framework.

Provides a decorator-based registration system and factory function for creating
component instances by category and name. This enables a fully pluggable architecture
where any component (quantizer, phonology module, loss, etc.) can be swapped via
configuration alone.
"""

from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, dict[str, type]] = {}


def register(category: str, name: str | None = None):
    """Class decorator that registers a class under the given category.

    If ``name`` is not provided, the class name is lower-cased and used instead.

    Usage::

        @register("quantizer", "vqvae")
        class VQVAEQuantizer(LFMModule):
            ...

    Args:
        category: Registry category (e.g. ``"quantizer"``, ``"loss"``).
        name: Optional explicit name. Defaults to ``cls.__name__.lower()``.

    Returns:
        A class decorator that registers and returns the class unchanged.
    """

    def decorator(cls: type) -> type:
        registry_name = name if name is not None else cls.__name__.lower()
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        if registry_name in _REGISTRY[category]:
            existing = _REGISTRY[category][registry_name]
            raise ValueError(
                f"Duplicate registration: {category}/{registry_name} is already "
                f"registered to {existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[category][registry_name] = cls
        return cls

    return decorator


def create(category: str, name: str, config: Any = None, **kwargs: Any) -> Any:
    """Factory function that instantiates a registered component.

    Looks up the class registered under ``category/name`` and calls it with the
    provided ``config`` and any extra keyword arguments.

    Args:
        category: Registry category (e.g. ``"quantizer"``).
        name: Registered name within that category.
        config: Configuration object passed as the first positional argument.
        **kwargs: Additional keyword arguments forwarded to the constructor.

    Returns:
        An instance of the registered class.

    Raises:
        KeyError: If the category or name is not found in the registry.
    """
    if category not in _REGISTRY:
        raise KeyError(
            f"Unknown registry category: {category!r}. "
            f"Available categories: {sorted(_REGISTRY.keys())}"
        )
    if name not in _REGISTRY[category]:
        raise KeyError(
            f"Unknown name {name!r} in category {category!r}. "
            f"Available: {sorted(_REGISTRY[category].keys())}"
        )
    cls = _REGISTRY[category][name]
    if config is not None:
        return cls(config, **kwargs)
    return cls(**kwargs)


def list_registered(category: str) -> list[str]:
    """Return sorted list of names registered under the given category.

    Args:
        category: Registry category to query.

    Returns:
        Sorted list of registered component names. Returns an empty list if the
        category does not exist.
    """
    if category not in _REGISTRY:
        return []
    return sorted(_REGISTRY[category].keys())
