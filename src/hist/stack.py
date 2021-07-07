import sys
import typing
from typing import Any, Iterator, List, Tuple, Union

from .basehist import BaseHist

if typing.TYPE_CHECKING:
    from mplhep.plot import Hist1DArtists


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

    def __repr__(self) -> str:
        str_stack = ", ".join(repr(h) for h in self._stack)
        return f"{self.__class__.__name__}({str_stack})"

    def __getitem__(
        self, val: Union[int, slice]
    ) -> Union[BaseHist, Tuple[BaseHist, ...]]:
        return self._stack.__getitem__(val)

    def __iter__(self) -> Iterator[BaseHist]:
        return iter(self._stack)

    def plot(self, *, overlay: None = None, **kwargs: Any) -> "List[Hist1DArtists]":
        """
        Plot method for Stack object.
        """
        if overlay is not None:
            raise NotImplementedError("Currently overlay is not supported")

        if self._stack[0].ndim != 1:
            raise NotImplementedError("Please project to 1D before calling plot")

        try:
            import mplhep.plot
        except ModuleNotFoundError:
            print(
                f"{self.__class__.__name__}.plot() requires mplhep to plot, either install hist[plot] or mplhep",
                file=sys.stderr,
            )
            raise

        return mplhep.plot.histplot(list(self._stack), **kwargs)  # type: ignore
