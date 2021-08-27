from __future__ import annotations

import sys
import typing
from typing import Any, Iterator, TypeVar

import histoprint

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

        self._stack = args

        if len(args) == 0:
            raise ValueError("There should be histograms in the Stack")

        if not all(isinstance(a, BaseHist) for a in args):
            raise ValueError("There should be only histograms in Stack")

        first_axes = args[0].axes
        for a in args[1:]:
            if first_axes != a.axes:
                raise ValueError("The Histogram axes don't match")

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


def __dir__() -> tuple[str, ...]:
    return __all__
