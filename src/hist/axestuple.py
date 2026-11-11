from __future__ import annotations

__lazy_modules__ = {"warnings"}

import warnings
from typing import TYPE_CHECKING, Any

from boost_histogram.axis import ArrayTuple, AxesTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ("ArrayTuple", "AxesTuple", "NamedAxesTuple")


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
        msg = f"{name} not found in axes"
        raise KeyError(msg)

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
        return tuple(ax.name for ax in self)

    @name.setter
    def name(self, values: Iterable[str]) -> None:
        axes_and_names = tuple(zip(self, values, strict=True))

        disallowed_names = {"weight", "sample", "threads"}
        for _, name in axes_and_names:
            if name in disallowed_names:
                disallowed_warning = (
                    f"{name} is a protected keyword and cannot be used as axis name"
                )
                warnings.warn(disallowed_warning, stacklevel=2)

        valid_names = [name for _, name in axes_and_names if name]
        if len(valid_names) != len(set(valid_names)):
            msg = f"{self.__class__.__name__} instance cannot contain axes with duplicated names"
            raise KeyError(msg)

        for ax, name in axes_and_names:
            ax._raw_metadata["name"] = name

    @property
    def label(self) -> tuple[str]:
        """
        The labels of the axes. Defaults to name if label not given, or Axis N
        if neither was given.
        """
        return tuple(ax.label for ax in self)
