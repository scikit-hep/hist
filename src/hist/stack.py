from __future__ import annotations

import copy
import sys
import typing
from typing import Any, Iterator, TypeVar

import histoprint
import numpy as np

from .axestuple import NamedAxesTuple
from .basehist import BaseHist

try:
    import matplotlib
except ModuleNotFoundError:
    print(
        "Hist requires mplhep to plot, either install hist[plot] or mplhep",
        file=sys.stderr,
    )
    raise

__all__ = ("Stack",)

T = TypeVar("T", bound="Stack")


class Stack:
    def __init__(
        self,
        *args: BaseHist,
    ) -> None:
        """
        Initialize Stack of histograms.
        """

        self._stack = list(args)

        if len(args) == 0:
            raise ValueError("There should be histograms in the Stack")

        if not all(isinstance(a, BaseHist) for a in args):
            raise ValueError("There should be only histograms in Stack")

        first_axes = args[0].axes
        for a in args[1:]:
            if first_axes != a.axes:
                raise ValueError("The Histogram axes don't match")

    @classmethod
    def from_iter(cls: type[T], iterable: typing.Iterable[BaseHist]) -> T:
        """
        Create a Stack from an iterable of histograms.
        """
        return cls(*iterable)

    @classmethod
    def from_dict(cls: type[T], d: typing.Mapping[str, BaseHist]) -> T:
        """
        Create a Stack from a dictionary of histograms. The keys of the
        dictionary are used as names.
        """

        new_dict = {k: copy.copy(h) for k, h in d.items()}
        for k, h in new_dict.items():
            h.name = k

        return cls(*new_dict.values())

    @typing.overload
    def __getitem__(self, val: int) -> BaseHist:
        ...

    @typing.overload
    def __getitem__(self: T, val: slice) -> T:
        ...

    def __getitem__(self: T, val: int | slice) -> BaseHist | T:
        if isinstance(val, slice):
            return self.__class__(*self._stack.__getitem__(val))

        return self._stack.__getitem__(val)

    def __setitem__(self: T, key: int, value: BaseHist) -> None:
        """
        Set a histogram in the Stack. Checks the axes of the histogram, they must match.
        """
        if not isinstance(value, BaseHist):
            raise ValueError("The value should be a histogram")
        if not value.axes == self._stack[key].axes:
            raise ValueError("The histogram axes don't match")

        self._stack[key] = value

    def __iter__(self) -> Iterator[BaseHist]:
        return iter(self._stack)

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        str_stack = ", ".join(repr(h) for h in self)
        return f"{self.__class__.__name__}({str_stack})"

    @property
    def axes(self) -> NamedAxesTuple:
        return self._stack[0].axes

    def plot(self, *, ax: matplotlib.axes.Axes | None = None, **kwargs: Any) -> Any:
        """
        Plot method for Stack object.
        """

        import hist.plot

        return hist.plot.plot_stack(self, ax=ax, **kwargs)

    def show(self, **kwargs: Any) -> Any:
        """
        Pretty print the stacked histograms to the console.
        """
        if "label" not in kwargs:
            if all(h.name is not None for h in self):
                kwargs["label"] = [h.name for h in self]

        return histoprint.print_hist(list(self), stack=True, **kwargs)

    def __mul__(self: T, other: float) -> T:
        """
        Multiply the Stack by a scalar.
        """
        return self.__class__(*(h * other for h in self))

    def __imul__(self: T, other: float) -> T:
        """
        Multiply each histogram in the Stack by a scalar.
        """
        for h in self:
            h *= other
        return self

    def __rmul__(self: T, other: float) -> T:
        """
        Multiply the Stack by a scalar.
        """
        return self.__mul__(other)

    def __add__(self: T, other: float | np.typing.NDArray[Any]) -> T:
        """
        Add a scalar or array to the Stack.
        """
        return self.__class__(*(h + other for h in self))

    def __iadd__(self: T, other: float | np.typing.NDArray[Any]) -> T:
        """
        Add a scalar or array to the Stack.
        """
        for h in self:
            h += other
        return self

    def __radd__(self: T, other: float | np.typing.NDArray[Any]) -> T:
        """
        Add a scalar or array to the Stack.
        """
        return self.__add__(other)

    def __sub__(self: T, other: float | np.typing.NDArray[Any]) -> T:
        """
        Subtract a scalar or array to the Stack.
        """
        return self.__class__(*(h - other for h in self))  # type: ignore

    def __isub__(self: T, other: float | np.typing.NDArray[Any]) -> T:
        """
        Subtract a scalar or array to the Stack.
        """
        for h in self:
            h -= other  # type: ignore
        return self

    def project(self: T, *args: int | str) -> T:
        """
        Project the Stack onto a new axes.
        """
        return self.__class__(*(h.project(*args) for h in self))  # type: ignore


def __dir__() -> tuple[str, ...]:
    return __all__
