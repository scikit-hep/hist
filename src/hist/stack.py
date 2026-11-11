from __future__ import annotations

__lazy_modules__ = {
    "copy",
    "histoprint",
    f"{__spec__.parent}.axis",
    f"{__spec__.parent}.basehist",
}

import copy
import typing
from typing import Any, Generic, TypeVar

import boost_histogram as bh
import histoprint

from .axis import IntCategory, StrCategory
from .basehist import BaseHist

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    import matplotlib as mpl
    import numpy as np

    from ._compat.typing import Self
    from .axestuple import NamedAxesTuple


__all__ = ("Stack",)


S = TypeVar("S", bound=bh.storage.Storage)


class Stack(Generic[S]):
    def __init__(
        self,
        *args: BaseHist[S],
    ) -> None:
        """
        Initialize Stack of histograms.
        """

        self._stack = list(args)

        if not args:
            msg = "There should be histograms in the Stack"
            raise ValueError(msg)

        if not all(isinstance(a, BaseHist) for a in args):
            msg = "There should be only histograms in Stack"
            raise ValueError(msg)

        first_axes = args[0].axes
        for a in args[1:]:
            if first_axes != a.axes:
                msg = "The Histogram axes don't match"
                raise ValueError(msg)

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_iter(cls, iterable: typing.Iterable[BaseHist[S]]) -> Self:
        """
        Create a Stack from an iterable of histograms.
        """
        return cls(*iterable)

    @classmethod
    def from_dict(cls, d: typing.Mapping[str, BaseHist[S]]) -> Self:
        """
        Create a Stack from a dictionary of histograms. The keys of the
        dictionary are used as names.
        """

        new_dict = {k: copy.copy(h) for k, h in d.items()}
        for k, h in new_dict.items():
            h.name = k

        return cls(*new_dict.values())

    def to_hist(self, *, name: str = "stack", label: str = "") -> BaseHist[S]:
        """Copy the stack into a histogram with a leading categorical axis.

        Histogram names become categories in stack order. Names must be unique
        and either all strings (``StrCategory``) or all ints (``IntCategory``).
        All histograms must use the same storage. The new axis name must not
        collide with an existing axis name. The result has the class of the
        first histogram. Flow bins and accumulator fields are copied; the
        result owns its data and axes independently of the stack. The new
        axis has no overflow bin, so entries that ``BaseHist.stack`` dropped
        from a category overflow bin are not restored. If any histogram has
        unknown variances, the result has unknown variances.
        """
        names: list[object] = [h.name for h in self]
        if all(isinstance(n, str) for n in names):
            stack_axis: StrCategory | IntCategory = StrCategory(
                typing.cast("list[str]", names), name=name, label=label, overflow=False
            )
        elif all(isinstance(n, int) and not isinstance(n, bool) for n in names):
            stack_axis = IntCategory(
                typing.cast("list[int]", names), name=name, label=label, overflow=False
            )
        else:
            msg = "Stack histograms must have all string or all int names"
            raise ValueError(msg)
        if len(set(names)) != len(names):
            msg = "Stack histogram names must be unique"
            raise ValueError(msg)
        if name and name in self.axes.name:
            msg = "Stack axis name conflicts with an existing axis"
            raise ValueError(msg)
        if any(h.storage_type != self[0].storage_type for h in self):
            msg = "Stack histograms must have the same storage"
            raise ValueError(msg)

        result = type(self[0])(
            stack_axis,
            *(copy.deepcopy(ax) for ax in self.axes),
            storage=self[0].storage_type(),
        )
        for i, h in enumerate(self):
            result.view(flow=True)[i, ...] = h.view(flow=True)
        result._variance_known = all(h._variance_known for h in self)
        return result

    def _get_index(self, name: str | int) -> int:
        "Returns the index associated with a name. Passes through ints"
        if not isinstance(name, str):
            return name

        for n, h in enumerate(self._stack):
            if h.name == name:
                return n

        msg = f"Name not found: {name}"
        raise IndexError(msg)

    @typing.overload
    def __getitem__(self, val: int | str) -> BaseHist[S]: ...

    @typing.overload
    def __getitem__(self, val: slice) -> Self: ...

    def __getitem__(self, val: int | slice | str) -> BaseHist[S] | Self:
        """
        Index the Stack by position or by histogram name.

        Slices follow normal Python (half-open) semantics, so a slice stop
        given as a name is *exclusive*: ``stack["a":"c"]`` yields the
        histograms from ``"a"`` up to but not including ``"c"``.
        """
        if isinstance(val, str):
            val = self._get_index(val)
        if isinstance(val, slice):
            my_slice = slice(
                self._get_index(val.start), self._get_index(val.stop), val.step
            )
            return self.__class__(*self._stack.__getitem__(my_slice))

        return self._stack.__getitem__(val)

    def __setitem__(self, key: int | str, value: BaseHist[S]) -> None:
        """
        Set a histogram in the Stack. Checks the axes of the histogram, they must match.
        """
        if not isinstance(value, BaseHist):
            msg = "The value should be a histogram"  # type: ignore[unreachable]
            raise TypeError(msg)
        if isinstance(key, str):
            key = self._get_index(key)
        if value.axes != self._stack[key].axes:
            msg = "The histogram axes don't match"
            raise ValueError(msg)

        self._stack[key] = value

    def __iter__(self) -> Iterator[BaseHist[S]]:
        return iter(self._stack)

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        names = ", ".join(repr(getattr(h, "name", f"H{i}")) for i, h in enumerate(self))
        h = repr(self[0]) if len(self) else "Empty stack"
        return f"{self.__class__.__name__}<({names}) of {h}>"

    def __eq__(self, other: object) -> bool:
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

    def __truediv__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Divide the Stack by a scalar or array.
        """
        return self.__class__(*(h / other for h in self))

    def __itruediv__(self, other: float | np.typing.NDArray[Any]) -> Self:
        """
        Divide each histogram in the Stack by a scalar or array.
        """
        for h in self:
            h /= other  # noqa: PLW2901
        return self

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
        return self.__class__(*(h.project(*args) for h in self))


def __dir__() -> tuple[str, ...]:
    return __all__
