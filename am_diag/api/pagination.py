"""Cursor-pagination wrapper for resource list endpoints."""

from __future__ import annotations

from typing import Generic, TypeVar


T = TypeVar("T")


class Page(Generic[T]):
    """One page of results plus the cursor for the next page.

    Attributes:
        items: The records on this page.
        next_cursor: Opaque cursor for the next page, or ``None`` when the page
            is the last one.
    """

    __slots__ = ("items", "next_cursor")

    def __init__(self, items: list[T], next_cursor: str | None) -> None:
        """Store the page items and the next-page cursor."""
        self.items = items
        self.next_cursor = next_cursor

    def __iter__(self):
        """Iterate over the items on this page."""
        return iter(self.items)

    def __len__(self) -> int:
        """Return the number of items on this page."""
        return len(self.items)

    def __repr__(self) -> str:
        """Return a debug representation of the page."""
        return f"Page(items={len(self.items)}, next_cursor={self.next_cursor!r})"
