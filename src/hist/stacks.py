import sys
from typing import Any, Optional

from .axis import AxesMixin
from .basehist import BaseHist

try:
    from mplhep.plot import histplot
except ModuleNotFoundError:
    print(
        "Hist stack requires mplhep to plot, either install hist[plot] or mplhep",
        file=sys.stderr,
    )
    raise


class Stack:
    def __init__(
        self,
        *args: Any,
    ) -> None:
        """
        Initialize Stack for histograms and axes.
        """
        if len(args) == 0:
            raise ValueError("There should be histograms or axes in stack")

        if all([isinstance(a, BaseHist) for a in args]):
            axes_type = args[0].axes
            for a in args:
                if axes_type != a.axes:
                    raise ValueError("Histograms' axes don't match")

            self._stack = [*args]
            self._stack_len = len(args)

        elif all([isinstance(a, AxesMixin) for a in args]):
            if len({*args}) != 1:
                raise ValueError("Axes don't match")

            self._stack = [*args]
            self._stack_len = len(args)

        else:
            raise ValueError("There should be histograms or axes in Stack")

    def plot(self, *args: Any, overlay: "Optional[str]" = None, **kwargs: Any) -> "Any":
        """
        Plot method for Stack object.
        """

        if all([isinstance(a, BaseHist) for a in args]):
            if self._stack[0].ndim == 1:
                return histplot(self._stack, stack=True)
            else:
                raise NotImplementedError("Please project to 1D before calling plot")

        else:
            raise NotImplementedError("Only plot histogram stack are supported")
