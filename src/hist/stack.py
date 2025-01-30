from __future__ import annotations

import copy
import typing
from collections.abc import Iterator
from typing import Any

import histoprint
import numpy as np

from ._compat.typing import Self
from .axestuple import NamedAxesTuple
from .basehist import BaseHist

if typing.TYPE_CHECKING:
    import matplotlib as mpl


__all__ = ("Stack",)


class Stack:
    def __init__(
        self,
        *args: BaseHist,
    ) -> None:
        """
        Initialize Stack of histograms.
        """

        self._stack = list(args)

        if not args:
            raise ValueError("There should be histograms in the Stack")

        if not all(isinstance(a, BaseHist) for a in args):
            raise ValueError("There should be only histograms in Stack")

        first_axes = args[0].axes
        for a in args[1:]:
            if first_axes != a.axes:
                raise ValueError("The Histogram axes don't match")

    @classmethod
    def from_iter(cls, iterable: typing.Iterable[BaseHist]) -> Self:
        """
        Create a Stack from an iterable of histograms.
        """
        return cls(*iterable)

    @classmethod
    def from_dict(cls, d: typing.Mapping[str, BaseHist]) -> Self:
        """
        Create a Stack from a dictionary of histograms. The keys of the
        dictionary are used as names.
        """

        new_dict = {k: copy.copy(h) for k, h in d.items()}
        for k, h in new_dict.items():
            h.name = k

        return cls(*new_dict.values())

    def _get_index(self, name: str | int) -> int:
        "Returns the index associated with a name. Passes through ints"
        if not isinstance(name, str):
            return name

        for n, h in enumerate(self._stack):
            if h.name == name:
                return n

        raise IndexError(f"Name not found: {name}")

    @typing.overload
    def __getitem__(self, val: int | str) -> BaseHist: ...

    @typing.overload
    def __getitem__(self, val: slice) -> Self: ...

    def __getitem__(self, val: int | slice | str) -> BaseHist | Self:
        if isinstance(val, str):
            val = self._get_index(val)
        if isinstance(val, slice):
            my_slice = slice(
                self._get_index(val.start), self._get_index(val.stop), val.step
            )
            return self.__class__(*self._stack.__getitem__(my_slice))

        return self._stack.__getitem__(val)

    def __setitem__(self, key: int | str, value: BaseHist) -> None:
        """
        Set a histogram in the Stack. Checks the axes of the histogram, they must match.
        """
        if not isinstance(value, BaseHist):
            raise ValueError("The value should be a histogram")
        if isinstance(key, str):
            key = self._get_index(key)
        if not value.axes == self._stack[key].axes:
            raise ValueError("The histogram axes don't match")

        self._stack[key] = value

    def __iter__(self) -> Iterator[BaseHist]:
        return iter(self._stack)

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        names = ", ".join(repr(getattr(h, "name", f"H{i}")) for i, h in enumerate(self))
        h = repr(self[0]) if len(self) else "Empty stack"
        return f"{self.__class__.__name__}<({names}) of {h}>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Stack):
            return False
        return self._stack == other._stack

    @property
    def axes(self) -> NamedAxesTuple:
        return self._stack[0].axes

    def plot(self, *, ax: mpl.axes.Axes | None = None, **kwargs: Any) -> Any:
        """
        Plot method for Stack object.
        """

        import hist.plot

        return hist.plot.plot_stack(self, ax=ax, **kwargs)

    def show(self, **kwargs: object) -> Any:
        """
        Pretty print the stacked histograms to the console.
        """
        if "labels" not in kwargs and all(h.name is not None for h in self):
            kwargs["labels"] = [h.name for h in self]

        return histoprint.print_hist(list(self), stack=True, **kwargs)

    def __mul__(self, other: float) -> Self:
        """
        Multiply the Stack by a scalar.
        """
        return self.__class__(*(h * other for h in self))

    def __imul__(self, other: float) -> Self:
        """
        Multiply each histogram in the Stack by a scalar.
        """
        for h in self:
            h *= other  # noqa: PLW2901
        return self

    def __rmul__(self, other: float) -> Self:
        """
        Multiply the Stack by a scalar.
        """
        return self.__mul__(other)

    def __add__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Add a scalar or array to the Stack.
        """
        return self.__class__(*(h + other for h in self))

    def __iadd__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Add a scalar or array to the Stack.
        """
        for h in self:
            h += other  # noqa: PLW2901
        return self

    def __radd__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Add a scalar or array to the Stack.
        """
        return self.__add__(other)

    def __sub__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Subtract a scalar or array to the Stack.
        """
        return self.__class__(*(h - other for h in self))

    def __isub__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Subtract a scalar or array to the Stack.
        """
        for h in self:
            h -= other  # noqa: PLW2901
        return self

    def project(self, *args: int | str) -> Self:
        """
        Project the Stack onto a new axes.
        """
        return self.__class__(*(h.project(*args) for h in self))  # type: ignore[arg-type]


def __dir__() -> tuple[str, ...]:
    return __all__
