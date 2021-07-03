from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from mplhep.plot import Hist1DArtists, Hist2DArtists


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
        axes_check = args[0].axes
        for a in args:
            if axes_check != a.axes:
                raise ValueError("Histograms' axes don't match")
        self._stack = [*args]
        self._stack_len = len(args)

    def plot(
        self, *args: Any, overlay: "Optional[str]" = None, **kwargs: Any
    ) -> "List[Union[Hist1DArtists, Hist2DArtists]]":
        """
        Plot method for Stack object.
        """
        _has_categorical = 0
        if (
            self._stack[0].axes.traits.ordered == 1
            and self._stack[0].axes.traits.discrete == 1
        ):
            _has_categorical = 1
        _project = _has_categorical or overlay is not None
        if self._stack[0].ndim == 1 or (self._stack[0].ndim == 2 and _project):
            return [
                self._stack[-i - 1].plot1d(*args, overlay=overlay, **kwargs)
                for i in range(self._stack_len)
            ]
        else:
            raise NotImplementedError("Please project to 1D before calling plot")
