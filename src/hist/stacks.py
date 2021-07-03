import sys
from typing import Any, Optional

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
        # todo: make the args support axes.

        if len(args) == 0:
            raise ValueError("There should be histograms or axes in Stack")

        axes_type = [type(ax) for ax in args[0].axes]
        for h in args:
            if axes_type != [type(ax) for ax in h.axes]:
                raise ValueError("Histograms' axes don't match")

        self._stack = [*args]
        self._stack_len = len(args)

    def plot(self, *args: Any, overlay: "Optional[str]" = None, **kwargs: Any) -> "Any":
        """
        Plot method for Stack object.
        """

        if self._stack[0].ndim == 1:
            return histplot(self._stack, stack=True)
        else:
            raise NotImplementedError("Please project to 1D before calling plot")
