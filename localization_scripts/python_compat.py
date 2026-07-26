"""Small runtime helpers for the Windows Python 3.9 environment."""

from collections.abc import Iterable, Iterator
from itertools import zip_longest
from typing import Any


def strict_zip(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """Iterate equally sized iterables, raising when their lengths differ."""
    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values
