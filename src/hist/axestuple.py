from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

from boost_histogram.axis import ArrayTuple, AxesTuple

if sys.version_info < (3, 10):
    import builtins
    import itertools

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


__all__ = ("NamedAxesTuple", "AxesTuple", "ArrayTuple")


def __dir__() -> tuple[str, ...]:
    return __all__


class NamedAxesTuple(AxesTuple):
    __slots__ = ()

    def _get_index_by_name(self, name: int | str | None) -> int | None:
        if not isinstance(name, str):
            return name

        for i, ax in enumerate(self):
            if ax.name == name:
                return i
        raise KeyError(f"{name} not found in axes")

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            item = slice(
                self._get_index_by_name(item.start),
                self._get_index_by_name(item.stop),
                self._get_index_by_name(item.step),
            )
        else:
            item = self._get_index_by_name(item)

        return super().__getitem__(item)

    @property
    def name(self) -> tuple[str]:
        """
        The names of the axes. May be empty strings.
        """
        return tuple(ax.name for ax in self)  # type: ignore[return-value]

    @name.setter
    def name(self, values: Iterable[str]) -> None:
        # strict = True from Python 3.10
        for ax, val in zip(self, values, strict=True):
            ax._ax.metadata["name"] = val

        disallowed_names = {"weight", "sample", "threads"}
        for ax in self:
            if ax.name in disallowed_names:
                disallowed_warning = (
                    f"{ax.name} is a protected keyword and cannot be used as axis name"
                )
                warnings.warn(disallowed_warning)

        valid_names = [ax.name for ax in self if ax.name]
        if len(valid_names) != len(set(valid_names)):
            raise KeyError(
                f"{self.__class__.__name__} instance cannot contain axes with duplicated names"
            )

    @property
    def label(self) -> tuple[str]:
        """
        The labels of the axes. Defaults to name if label not given, or Axis N
        if neither was given.
        """
        return tuple(ax.label for ax in self)  # type: ignore[return-value]
