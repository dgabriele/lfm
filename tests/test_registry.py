"""Tests for the component registry system."""

from __future__ import annotations

from lfm._registry import create, list_registered, register


def test_register_and_list():
    """Registering a class makes it visible via list_registered."""

    @register("test_category", "test_component")
    class _TestComponent:
        def __init__(self, config):
            self.config = config

    names = list_registered("test_category")
    assert "test_component" in names


def test_create():
    """create() instantiates the correct class with config."""

    @register("test_create_cat", "my_thing")
    class _MyThing:
        def __init__(self, config, extra=None):
            self.config = config
            self.extra = extra

    instance = create("test_create_cat", "my_thing", config="hello", extra=42)
    assert instance.config == "hello"
    assert instance.extra == 42


def test_list_empty_category():
    """Listing an unregistered category returns an empty list."""
    assert list_registered("nonexistent_category_xyz") == []


def test_register_uses_class_name_when_no_name():
    """When name is omitted, the lowercased class name is used as the key."""

    @register("test_auto_name")
    class AutoNamed:
        pass

    assert "autonamed" in list_registered("test_auto_name")
