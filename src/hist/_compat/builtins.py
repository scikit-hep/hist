from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    import builtins
    import itertools
    from collections.abc import Iterator
    from typing import Any

    def zip(*iterables: Any, strict: bool = False) -> Iterator[tuple[Any, ...]]:
        if strict:
            marker = object()
            for each in itertools.zip_longest(*iterables, fillvalue=marker):
                for val in each:
                    if val is marker:
                        raise ValueError("zip() arguments are not the same length")
                yield each
        else:
            yield from builtins.zip(*iterables)

else:
    from builtins import zip  # noqa: UP029

__all__ = ["zip"]


def __dir__() -> list[str]:
    return __all__
