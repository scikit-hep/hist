# -*- coding: utf-8 -*-
from boost_histogram.axis import AxesTuple, ArrayTuple

from typing import Any, Union, Tuple

__all__ = ("NamedAxesTuple", "AxesTuple", "ArrayTuple")


class NamedAxesTuple(AxesTuple):
    __slots__ = ()

    def _get_index_by_name(self, name: Union[int, str, None]) -> Union[int, None]:
        if isinstance(name, str):
            for i, ax in enumerate(self):
                if ax.name == name:
                    return i
            raise KeyError(f"{name} not found in axes")
        else:
            return name

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
    def name(self) -> Tuple[str]:
        """
        The names of the axes. May be empty strings.
        """
        return tuple(ax.name for ax in self)  # type: ignore

    @property
    def label(self) -> Tuple[str]:
        """
        The labels of the axes. Defaults to name if label not given, or Axis N
        if neither was given.
        """
        return tuple(ax.label for ax in self)  # type: ignore
