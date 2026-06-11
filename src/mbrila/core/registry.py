"""Registries for kernels, inference engines, and models.

The registries support an ergonomic ``mbrila.api.fit(method="dlag", ...)``
facade and let third-party packages contribute methods via entry points
without touching the core library.

A registry is a lightweight wrapper around a string-keyed dict with two
behaviours we want to keep consistent across categories:

- Re-registering an existing key raises (forces a deliberate ``unregister``).
- Lookup raises a clear error listing the available keys.

Type ``T`` is bound to a class (typically a base type from :mod:`mbrila.core`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


class Registry[T]:
    """A name → class registry."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str, cls: type[T], *, replace: bool = False) -> type[T]:
        """Register ``cls`` under ``name``.

        Returns ``cls`` so this can be used as a decorator.
        """
        key = name.lower()
        if not replace and key in self._items:
            raise ValueError(
                f"{self._kind} '{name}' is already registered "
                f"(class={self._items[key].__qualname__}); "
                "pass replace=True to override."
            )
        self._items[key] = cls
        return cls

    def unregister(self, name: str) -> None:
        self._items.pop(name.lower(), None)

    def get(self, name: str) -> type[T]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(f"unknown {self._kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[key]

    def names(self) -> Iterable[str]:
        return tuple(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
