"""Unit tests for the generic Registry."""

from __future__ import annotations

import pytest

from mbrila import Registry


class _Foo:
    pass


class _Bar(_Foo):
    pass


def test_register_and_get() -> None:
    reg: Registry[_Foo] = Registry("widget")
    reg.register("foo", _Foo)
    reg.register("bar", _Bar)
    assert reg.get("foo") is _Foo
    assert reg.get("BAR") is _Bar  # case-insensitive
    assert "foo" in reg
    assert len(reg) == 2
    assert sorted(reg.names()) == ["bar", "foo"]


def test_unknown_key_raises() -> None:
    reg: Registry[_Foo] = Registry("widget")
    reg.register("foo", _Foo)
    with pytest.raises(KeyError, match="Registered"):
        reg.get("missing")


def test_double_register_raises_unless_replace() -> None:
    reg: Registry[_Foo] = Registry("widget")
    reg.register("foo", _Foo)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("foo", _Bar)
    reg.register("foo", _Bar, replace=True)
    assert reg.get("foo") is _Bar


def test_unregister() -> None:
    reg: Registry[_Foo] = Registry("widget")
    reg.register("foo", _Foo)
    reg.unregister("foo")
    assert "foo" not in reg
    # Idempotent.
    reg.unregister("foo")


def test_decorator_usage() -> None:
    reg: Registry[_Foo] = Registry("widget")
    decorated = reg.register("foo", _Foo)
    assert decorated is _Foo
