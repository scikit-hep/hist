import functools
import operator
import typing
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import boost_histogram as bh
import histoprint
import numpy as np

import hist

from .axestuple import NamedAxesTuple
from .axis import AxisProtocol
from .quick_construct import MetaConstructor
from .storage import Storage
from .svgplots import html_hist, svg_hist_1d, svg_hist_1d_c, svg_hist_2d, svg_hist_nd
from .typing import ArrayLike, Protocol, SupportsIndex

if typing.TYPE_CHECKING:
    from builtins import ellipsis

    import matplotlib.axes
    from mplhep.plot import Hist1DArtists, Hist2DArtists

    from .plot import FitResultArtists, MainAxisArtists, RatiolikeArtists


class SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


InnerIndexing = Union[
    SupportsIndex, str, Callable[[bh.axis.Axis], int], slice, "ellipsis"
]
IndexingWithMapping = Union[InnerIndexing, Mapping[Union[int, str], InnerIndexing]]
IndexingExpr = Union[IndexingWithMapping, Tuple[IndexingWithMapping, ...]]


# Workaround for bug in mplhep
def _proc_kw_for_lw(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        f"{k[:-3]}_linestyle"
        if k.endswith("_ls")
        else "linestyle"
        if k == "ls"
        else k: v
        for k, v in kwargs.items()
    }


T = TypeVar("T", bound="BaseHist")


class BaseHist(bh.Histogram, metaclass=MetaConstructor, family=hist):
    __slots__ = ()

    def __init__(
        self,
        *args: Union[AxisProtocol, Storage, str, Tuple[int, float, float]],
        storage: Optional[Union[Storage, str]] = None,
        metadata: Any = None,
        data: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize BaseHist object. Axis params can contain the names.
        """
        self._hist: Any = None
        self.axes: NamedAxesTuple

        if args and storage is None and isinstance(args[-1], (Storage, str)):
            storage = args[-1]
            args = args[:-1]

        # Support raw Quick Construct being accidentally passed in
        args = [
            a.axes[0]  # type: ignore
            if isinstance(a, hist.quick_construct.ConstructProxy) and len(a.axes) == 1
            else a
            for a in args
        ]

        if args:
            if isinstance(storage, str):
                storage_str = storage.title()
                if storage_str == "Atomicint64":
                    storage_str = "AtomicInt64"
                elif storage_str == "Weightedmean":
                    storage_str = "WeightedMean"
                storage = getattr(bh.storage, storage_str)()
            elif isinstance(storage, type):
                msg = (
                    f"Please use '{storage.__name__}()' instead of '{storage.__name__}'"
                )
                warnings.warn(msg)
                storage = storage()
            super().__init__(*args, storage=storage, metadata=metadata)  # type: ignore

            disallowed_names = {"weight", "sample", "threads"}
            for ax in self.axes:
                if ax.name in disallowed_names:
                    disallowed_warning = f"{ax.name} is a protected keyword and cannot be used as axis name"
                    warnings.warn(disallowed_warning)

            valid_names = [ax.name for ax in self.axes if ax.name]
            if len(valid_names) != len(set(valid_names)):
                raise KeyError(
                    f"{self.__class__.__name__} instance cannot contain axes with duplicated names"
                )
            for i, ax in enumerate(self.axes):
                # label will return name if label is not set, so this is safe
                if not ax.label:
                    ax.label = f"Axis {i}"

        if data is not None:
            self[...] = data

    def _generate_axes_(self) -> NamedAxesTuple:
        """
        This is called to fill in the axes. Subclasses can override it if they need
        to change the axes tuple.
        """

        return NamedAxesTuple(self._axis(i) for i in range(self.ndim))

    def _repr_html_(self) -> str:
        if self.size == 0:
            return str(self)
        elif self.ndim == 1:
            if self.axes[0].traits.circular:
                return str(html_hist(self, svg_hist_1d_c))
            else:
                return str(html_hist(self, svg_hist_1d))
        elif self.ndim == 2:
            return str(html_hist(self, svg_hist_2d))
        elif self.ndim > 2:
            return str(html_hist(self, svg_hist_nd))

        return str(self)

    def _name_to_index(self, name: str) -> int:
        """
            Transform axis name to axis index, given axis name, return axis \
            index.
        """
        for index, axis in enumerate(self.axes):
            if name == axis.name:
                return index

        raise ValueError(f"The axis name {name} could not be found")

    @classmethod
    def from_columns(
        cls: Type[T],
        data: Mapping[str, ArrayLike],
        axes: Sequence[Union[str, AxisProtocol]],
        *,
        weight: Optional[str] = None,
        storage: hist.storage.Storage = hist.storage.Double(),  # noqa: B008
    ) -> T:
        axes_list: List[Any] = list()
        for ax in axes:
            if isinstance(ax, str):
                assert ax in data, f"{ax} must be present in data={list(data)}"
                cats = set(data[ax])  # type: ignore
                if all(isinstance(a, str) for a in cats):
                    axes_list.append(hist.axis.StrCategory(sorted(cats), name=ax))  # type: ignore
                elif all(isinstance(a, int) for a in cats):
                    axes_list.append(hist.axis.IntCategory(sorted(cats), name=ax))  # type: ignore
                else:
                    raise TypeError(
                        f"{ax} must be all int or strings if axis not given"
                    )
            elif not ax.name or ax.name not in data:
                raise TypeError("All axes must have names present in the data")
            else:
                axes_list.append(ax)

        weight_arr = data[weight] if weight else None

        self = cls(*axes_list, storage=storage)
        data_list = {x.name: data[x.name] for x in axes_list}
        self.fill(**data_list, weight=weight_arr)  # type: ignore
        return self

    def project(
        self: T, *args: Union[int, str]
    ) -> Union[T, float, bh.accumulators.Accumulator]:
        """
        Projection of axis idx.
        """
        int_args = [self._name_to_index(a) if isinstance(a, str) else a for a in args]
        return super().project(*int_args)

    def fill(
        self: T,
        *args: ArrayLike,
        weight: Optional[ArrayLike] = None,
        sample: Optional[ArrayLike] = None,
        threads: Optional[int] = None,
        **kwargs: ArrayLike,
    ) -> T:
        """
        Insert data into the histogram using names and indices, return
        a Hist object.
        """

        data_dict = {
            self._name_to_index(k) if isinstance(k, str) else k: v
            for k, v in kwargs.items()
        }

        if set(data_dict) != set(range(len(args), self.ndim)):
            raise TypeError(
                "All axes must be accounted for in fill, you may have used a disallowed name in the axes"
            )

        data = (data_dict[i] for i in range(len(args), self.ndim))

        total_data = tuple(args) + tuple(data)  # Python 2 can't unpack twice
        return super().fill(*total_data, weight=weight, sample=sample, threads=threads)

    def sort(
        self: T,
        axis: Union[int, str],
        key: Union[
            Callable[[int], SupportsLessThan], Callable[[str], SupportsLessThan], None
        ] = None,
        reverse: bool = False,
    ) -> T:
        """
        Sort a categorical axis.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sorted_cats = sorted(self.axes[axis], key=key, reverse=reverse)
            # This can only return T, not float, etc., so we ignore the extra types here
            return self[{axis: [bh.loc(x) for x in sorted_cats]}]  # type: ignore

    def _loc_shortcut(self, x: Any) -> Any:
        """
        Convert some specific indices to location.
        """

        if isinstance(x, list):
            return [self._loc_shortcut(each) for each in x]
        elif isinstance(x, slice):
            return slice(
                self._loc_shortcut(x.start),
                self._loc_shortcut(x.stop),
                self._step_shortcut(x.step),
            )
        elif isinstance(x, complex):
            if x.real % 1 != 0:
                raise ValueError("The real part should be an integer")
            else:
                return bh.loc(x.imag, int(x.real))
        elif isinstance(x, str):
            return bh.loc(x)
        else:
            return x

    def _step_shortcut(self, x: Any) -> Any:
        """
        Convert some specific indices to step.
        """

        if not isinstance(x, complex):
            return x

        if x.real != 0:
            raise ValueError("The step should not have real part")
        elif x.imag % 1 != 0:
            raise ValueError("The imaginary part should be an integer")
        else:
            return bh.rebin(int(x.imag))

    def _index_transform(
        self, index: Union[List[IndexingExpr], IndexingExpr]
    ) -> bh.IndexingExpr:
        """
        Auxiliary function for __getitem__ and __setitem__.
        """

        if isinstance(index, dict):
            new_indices = {
                (
                    self._name_to_index(k) if isinstance(k, str) else k
                ): self._loc_shortcut(v)
                for k, v in index.items()
            }
            if len(new_indices) != len(index):
                raise ValueError(
                    "Duplicate index keys, numbers and names cannot overlap"
                )
            return new_indices

        elif not isinstance(index, tuple):
            index = (index,)  # type: ignore

        return tuple(self._loc_shortcut(v) for v in index)  # type: ignore

    def __getitem__(  # type: ignore
        self: T, index: IndexingExpr
    ) -> Union[T, float, bh.accumulators.Accumulator]:
        """
        Get histogram item.
        """

        return super().__getitem__(self._index_transform(index))

    def __setitem__(  # type: ignore
        self, index: IndexingExpr, value: Union[ArrayLike, bh.accumulators.Accumulator]
    ) -> None:
        """
        Set histogram item.
        """

        return super().__setitem__(self._index_transform(index), value)

    def profile(self: T, axis: Union[int, str]) -> T:
        """
        Returns a profile (Mean/WeightedMean) histogram from a normal histogram
        with N-1 axes. The axis given is profiled over and removed from the
        final histogram.
        """

        if self.kind != bh.Kind.COUNT:
            raise TypeError("Profile requires a COUNT histogram")

        axes = list(self.axes)
        iaxis = axis if isinstance(axis, int) else self._name_to_index(axis)
        axes.pop(iaxis)

        values = self.values()
        tmp_variances = self.variances()
        variances = tmp_variances if tmp_variances is not None else values
        centers = self.axes[iaxis].centers

        count = np.sum(values, axis=iaxis)

        num = np.tensordot(values, centers, ([iaxis], [0]))
        num_err = np.sqrt(np.tensordot(variances, centers ** 2, ([iaxis], [0])))

        den = np.sum(values, axis=iaxis)
        den_err = np.sqrt(np.sum(variances, axis=iaxis))

        with np.errstate(invalid="ignore"):
            new_values = num / den
            new_variances = (num_err / den) ** 2 - (den_err * num / den ** 2) ** 2

        retval = self.__class__(*axes, storage=hist.storage.Mean())
        retval[...] = np.stack([count, new_values, count * new_variances], axis=-1)
        return retval

    def density(self) -> np.ndarray:
        """
        Density NumPy array.
        """
        total = np.sum(self.values()) * functools.reduce(operator.mul, self.axes.widths)
        dens: np.ndarray = self.values() / np.where(total > 0, total, 1)
        return dens

    def show(self, **kwargs: Any) -> Any:
        """
        Pretty print histograms to the console.
        """

        return histoprint.print_hist(self, **kwargs)

    def plot(
        self, *args: Any, overlay: "Optional[str]" = None, **kwargs: Any
    ) -> "Union[Hist1DArtists, Hist2DArtists]":
        """
        Plot method for BaseHist object.
        """
        _has_categorical = 0
        if (
            np.sum(self.axes.traits.ordered) == 1
            and np.sum(self.axes.traits.discrete) == 1
        ):
            _has_categorical = 1
        _project = _has_categorical or overlay is not None
        if self.ndim == 1 or (self.ndim == 2 and _project):
            return self.plot1d(*args, overlay=overlay, **kwargs)
        elif self.ndim == 2:
            return self.plot2d(*args, **kwargs)
        else:
            raise NotImplementedError("Please project to 1D or 2D before calling plot")

    def plot1d(
        self,
        *,
        ax: "Optional[matplotlib.axes.Axes]" = None,
        overlay: "Optional[Union[str, int]]" = None,
        **kwargs: Any,
    ) -> "Hist1DArtists":
        """
        Plot1d method for BaseHist object.
        """

        import hist.plot

        if self.ndim == 1:
            return hist.plot.histplot(self, ax=ax, **_proc_kw_for_lw(kwargs))
        if overlay is None:
            (overlay,) = (i for i, ax in enumerate(self.axes) if ax.traits.discrete)
        assert overlay is not None
        cat_ax = self.axes[overlay]
        cats = cat_ax if cat_ax.traits.discrete else np.arange(len(cat_ax.centers))
        d1hists = [self[{overlay: cat}] for cat in cats]
        return hist.plot.histplot(d1hists, ax=ax, label=cats, **_proc_kw_for_lw(kwargs))

    def plot2d(
        self,
        *,
        ax: "Optional[matplotlib.axes.Axes]" = None,
        **kwargs: Any,
    ) -> "Hist2DArtists":
        """
        Plot2d method for BaseHist object.
        """

        import hist.plot

        return hist.plot.hist2dplot(self, ax=ax, **_proc_kw_for_lw(kwargs))

    def plot2d_full(
        self,
        *,
        ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
        **kwargs: Any,
    ) -> "Tuple[Hist2DArtists, Hist1DArtists, Hist1DArtists]":
        """
        Plot2d_full method for BaseHist object.

        Pass a dict of axes to ``ax_dict``, otherwise, the current figure will be used.
        """
        # Type judgement

        import hist.plot

        return hist.plot.plot2d_full(self, ax_dict=ax_dict, **kwargs)

    def plot_ratio(
        self,
        other: Union["hist.BaseHist", Callable[[np.ndarray], np.ndarray], str],
        *,
        ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
        **kwargs: Any,
    ) -> "Tuple[MainAxisArtists, RatiolikeArtists]":
        """
        ``plot_ratio`` method for ``BaseHist`` object.

        Return a tuple of artists following a structure of
        ``(main_ax_artists, subplot_ax_artists)``
        """

        import hist.plot

        return hist.plot._plot_ratiolike(
            self, other, ax_dict=ax_dict, view="ratio", **kwargs
        )

    def plot_pull(
        self,
        func: Union[Callable[[np.ndarray], np.ndarray], str],
        *,
        ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
        **kwargs: Any,
    ) -> "Tuple[FitResultArtists, RatiolikeArtists]":
        """
        ``plot_pull`` method for ``BaseHist`` object.

        Return a tuple of artists following a structure of
        ``(main_ax_artists, subplot_ax_artists)``
        """

        import hist.plot

        return hist.plot._plot_ratiolike(
            self, func, ax_dict=ax_dict, view="pull", **kwargs
        )

    def plot_pie(
        self,
        *,
        ax: "Optional[matplotlib.axes.Axes]" = None,
        **kwargs: Any,
    ) -> Any:

        import hist.plot

        return hist.plot.plot_pie(self, ax=ax, **kwargs)

    def stack(self, axis: Union[int, str]) -> "hist.stack.Stack":
        """
        Returns a stack from a normal histogram axes.
        """
        if self.ndim < 2:
            raise RuntimeError("Cannot stack with less than two axis")
        stack_histograms: Iterator[BaseHist] = [
            self[{axis: i}] for i in range(len(self.axes[axis]))  # type: ignore
        ]
        for name, h in zip(self.axes[axis], stack_histograms):
            h.name = name  # type: ignore

        return hist.stack.Stack(*stack_histograms)
