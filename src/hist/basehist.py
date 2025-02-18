from __future__ import annotations

import fnmatch
import functools
import itertools
import operator
import typing
import warnings
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    SupportsIndex,
    Tuple,
    Union,
)

import boost_histogram as bh
import histoprint
import numpy as np

import hist

from . import interop
from ._compat.typing import ArrayLike, Self
from .axestuple import NamedAxesTuple
from .axis import AxisProtocol
from .quick_construct import MetaConstructor
from .storage import Storage
from .svgplots import html_hist, svg_hist_1d, svg_hist_1d_c, svg_hist_2d

if typing.TYPE_CHECKING:
    from builtins import ellipsis

    import matplotlib.axes
    from mplhep.plot import Hist1DArtists, Hist2DArtists

    from .plot import FitResultArtists, MainAxisArtists, RatiolikeArtists


class SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool: ...


InnerIndexing = Union[
    SupportsIndex, str, Callable[[bh.axis.Axis], int], slice, "ellipsis"
]
IndexingWithMapping = Union[InnerIndexing, Mapping[Union[int, str], InnerIndexing]]
IndexingExpr = Union[IndexingWithMapping, Tuple[IndexingWithMapping, ...]]
AxisTypes = Union[AxisProtocol, Tuple[int, float, float]]


# Workaround for bug in mplhep
def _proc_kw_for_lw(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{k[:-3]}_linestyle"
        if k.endswith("_ls")
        else "linestyle"
        if k == "ls"
        else k: v
        for k, v in kwargs.items()
    }


def process_mistaken_quick_construct(
    axes: Sequence[AxisTypes | hist.quick_construct.ConstructProxy],
) -> Generator[AxisTypes, None, None]:
    for ax in axes:
        if isinstance(ax, hist.quick_construct.ConstructProxy):
            yield from ax.axes
        else:
            yield ax


class BaseHist(bh.Histogram, metaclass=MetaConstructor, family=hist):
    __slots__ = ()

    def __init__(
        self,
        *in_args: AxisTypes | Storage | str,
        storage: Storage | str | None = None,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize BaseHist object. Axis params can contain the names.
        """
        self._hist: Any = None
        self.axes: NamedAxesTuple
        self.name = name
        self.label = label

        args: tuple[AxisTypes, ...]

        if in_args and storage is None and isinstance(in_args[-1], (Storage, str)):
            storage = in_args[-1]
            args = in_args[:-1]  # type: ignore[assignment]
        else:
            args = in_args  # type: ignore[assignment]

        # Support raw Quick Construct being accidentally passed in
        args = tuple(ax for ax in process_mistaken_quick_construct(args))

        if isinstance(storage, str):
            storage_str = storage.title()
            if storage_str == "Atomicint64":
                storage_str = "AtomicInt64"
            elif storage_str == "Weightedmean":
                storage_str = "WeightedMean"
            storage = getattr(bh.storage, storage_str)()
        elif isinstance(storage, type):
            msg = f"Please use '{storage.__name__}()' instead of '{storage.__name__}'"  # type: ignore[unreachable]
            warnings.warn(msg, stacklevel=2)
            storage = storage()

        super().__init__(*args, storage=storage, metadata=metadata)  # type: ignore[call-overload]

        disallowed_names = {"weight", "sample", "threads"}
        for ax in self.axes:
            if ax.name in disallowed_names:
                disallowed_warning = (
                    f"{ax.name} is a protected keyword and cannot be used as axis name"
                )
                warnings.warn(disallowed_warning, stacklevel=2)

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

    # Backport of storage_type from boost-histogram 1.3.2:
    if not hasattr(bh.Histogram, "storage_type"):

        @property
        def storage_type(self) -> type[bh.storage.Storage]:
            return self._storage_type

    def _generate_axes_(self) -> NamedAxesTuple:
        """
        This is called to fill in the axes. Subclasses can override it if they need
        to change the axes tuple.
        """

        return NamedAxesTuple(self._axis(i) for i in range(self.ndim))

    def _repr_html_(self) -> str | None:
        if self.size == 0:
            return None

        if self.ndim == 1 and len(self.axes[0]) <= 1000:
            return str(
                html_hist(
                    self,
                    svg_hist_1d_c if self.axes[0].traits.circular else svg_hist_1d,
                )
            )

        if self.ndim == 2 and len(self.axes[0]) <= 200 and len(self.axes[1]) <= 200:
            return str(html_hist(self, svg_hist_2d))

        return None

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
        cls,
        data: Mapping[str, ArrayLike],
        axes: Sequence[str | AxisProtocol],
        *,
        weight: str | None = None,
        storage: hist.storage.Storage = hist.storage.Double(),  # noqa: B008
    ) -> Self:
        axes_list: list[Any] = []
        for ax in axes:
            if isinstance(ax, str):
                assert ax in data, f"{ax} must be present in data={list(data)}"
                cats = set(data[ax])
                if all(isinstance(a, str) for a in cats):
                    axes_list.append(hist.axis.StrCategory(sorted(cats), name=ax))
                elif all(isinstance(a, int) for a in cats):
                    axes_list.append(hist.axis.IntCategory(sorted(cats), name=ax))
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
        self.fill(**data_list, weight=weight_arr)
        return self

    def project(self, *args: int | str) -> Self | float | bh.accumulators.Accumulator:
        """
        Projection of axis idx.
        """
        int_args = [self._name_to_index(a) if isinstance(a, str) else a for a in args]
        return super().project(*int_args)

    @property
    def T(self) -> Self:
        return self.project(*reversed(range(self.ndim)))  # type: ignore[return-value]

    def fill(
        self,
        *args: ArrayLike,
        weight: ArrayLike | None = None,
        sample: ArrayLike | None = None,
        threads: int | None = None,
        **kwargs: ArrayLike,
    ) -> Self:
        """
        Insert data into the histogram using names and indices, return
        a Hist object.
        """

        if (
            issubclass(self.storage_type, (bh.storage.Mean, bh.storage.WeightedMean))
            and sample is None
        ):
            msg = "A Mean-based storage requires a sample= argument with the values to average"
            raise TypeError(msg)

        data_dict = {
            self._name_to_index(k) if isinstance(k, str) else k: v  # type: ignore[redundant-expr]
            for k, v in kwargs.items()
        }

        if set(data_dict) != set(range(len(args), self.ndim)):
            raise TypeError(
                "All axes must be accounted for in fill, you may have used a disallowed name in the axes"
            )

        data = (data_dict[i] for i in range(len(args), self.ndim))

        return super().fill(*args, *data, weight=weight, sample=sample, threads=threads)

    def fill_flattened(
        self: Self,
        *args: Any,
        weight: Any | None = None,
        sample: Any | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ) -> Self:
        axis_names = {ax.name for ax in self.axes}

        non_user_kwargs = {}
        if weight is not None:
            non_user_kwargs["weight"] = weight
        if sample is not None:
            non_user_kwargs["sample"] = sample

        # Single arguments are either arrays for single-dimensional histograms, or
        # structures for multi-dimensional hists that must first be unpacked
        if len(args) == 1 and not kwargs:
            (arg,) = args
            destructured = interop.destructure(arg)
            # Try to unpack the array, if it's valid to do so, i.e. is the Awkward Array a record array?
            if destructured is None:
                # Can't unpack, fall back on broadcasting single array (to flatten and convert)
                broadcast = interop.broadcast_and_flatten(
                    (arg, *non_user_kwargs.values())
                )
                # Partition out non-user args
                user_args_broadcast = broadcast[:1]
                user_kwargs_broadcast = {}
                non_user_kwargs_broadcast = dict(
                    zip(non_user_kwargs.keys(), broadcast[1:])
                )
            else:
                # Result must be broadcast, so unpack and rebuild
                broadcast = interop.broadcast_and_flatten(
                    (*destructured.values(), *non_user_kwargs.values())
                )
                # Partition into user and non-user args
                user_args_broadcast = ()
                user_kwargs_broadcast = {
                    k: v
                    for k, v in zip(destructured, broadcast[: len(destructured)])
                    if k in axis_names
                }
                non_user_kwargs_broadcast = dict(
                    zip(non_user_kwargs, broadcast[len(destructured) :])
                )
        # Multiple args: broadcast and flatten!
        else:
            inputs = (*args, *kwargs.values(), *non_user_kwargs)
            broadcast = interop.broadcast_and_flatten(inputs)
            user_args_broadcast = broadcast[: len(args)]
            user_kwargs_broadcast = dict(
                zip(kwargs, broadcast[len(args) : len(args) + len(kwargs)])
            )
            non_user_kwargs_broadcast = dict(
                zip(non_user_kwargs, broadcast[len(args) + len(kwargs) :])
            )
        return self.fill(
            *user_args_broadcast,
            **user_kwargs_broadcast,
            threads=threads,
            **non_user_kwargs_broadcast,
        )

    def sort(
        self,
        axis: int | str,
        key: (
            Callable[[int], SupportsLessThan] | Callable[[str], SupportsLessThan] | None
        ) = None,
        reverse: bool = False,
    ) -> Self:
        """
        Sort a categorical axis.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sorted_cats = sorted(self.axes[axis], key=key, reverse=reverse)
            # This can only return Self, not float, etc., so we ignore the extra types here
            return self[{axis: [bh.loc(x) for x in sorted_cats]}]  # type: ignore[dict-item, return-value]

    def _convert_index_wildcards(self, x: Any, ax_id: str | int | None = None) -> Any:
        """
        Convert wildcards to available indices before passing to bh.loc
        """

        if not any(
            isinstance(x, t) for t in [str, list]
        ):  # Process only lists and strings
            return x
        _x = x if isinstance(x, list) else [x]  # Convert to list if not already
        if not all(isinstance(n, str) for n in _x):
            return x
        if any(any(special in pattern for special in ["*", "?"]) for pattern in _x):
            available = [n for n in self.axes[ax_id] if isinstance(n, str)]
            all_matches = []
            for pattern in _x:
                all_matches.append(
                    [k for k in available if fnmatch.fnmatch(k, pattern)]
                )
            matches = list(
                dict.fromkeys(list(itertools.chain.from_iterable(all_matches)))
            )
            if len(matches) == 0:
                raise ValueError(f"No matches found for {x}")
            return matches
        return x

    def _loc_shortcut(self, x: Any, ax_id: str | int | None = None) -> Any:
        """
        Convert some specific indices to location.
        """
        x = self._convert_index_wildcards(x, ax_id)
        if isinstance(x, list):
            return [self._loc_shortcut(each) for each in x]
        if isinstance(x, slice):
            return slice(
                self._loc_shortcut(x.start),
                self._loc_shortcut(x.stop),
                self._step_shortcut(x.step),
            )
        if isinstance(x, complex):
            if x.real % 1 != 0:
                raise ValueError("The real part should be an integer")
            return bh.loc(x.imag, int(x.real))
        if isinstance(x, str):
            return bh.loc(x)
        return x

    @staticmethod
    def _step_shortcut(x: Any) -> Any:
        """
        Convert some specific indices to step.
        """

        if not isinstance(x, complex):
            return x

        if x.real != 0:
            raise ValueError("The step should not have real part")
        if x.imag % 1 != 0:
            raise ValueError("The imaginary part should be an integer")
        return bh.rebin(int(x.imag))

    def _index_transform(self, index: list[IndexingExpr] | IndexingExpr) -> Any:
        """
        Auxiliary function for __getitem__ and __setitem__.
        """

        if isinstance(index, dict):
            new_indices = {
                (
                    self._name_to_index(k) if isinstance(k, str) else k
                ): self._loc_shortcut(v, k)
                for k, v in index.items()
            }
            if len(new_indices) != len(index):
                raise ValueError(
                    "Duplicate index keys, numbers and names cannot overlap"
                )
            return new_indices

        if not isinstance(index, tuple):
            index = (index,)  # type: ignore[assignment]

        return tuple(self._loc_shortcut(v, i) for i, (v) in enumerate(index))  # type: ignore[arg-type]

    def __getitem__(  # type: ignore[override]
        self, index: IndexingExpr
    ) -> Self | float | bh.accumulators.Accumulator:
        """
        Get histogram item.
        """

        return super().__getitem__(self._index_transform(index))

    def __setitem__(  # type: ignore[override]
        self, index: IndexingExpr, value: ArrayLike | bh.accumulators.Accumulator
    ) -> None:
        """
        Set histogram item.
        """

        return super().__setitem__(self._index_transform(index), value)

    def profile(self, axis: int | str) -> Self:
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
        num_err = np.sqrt(np.tensordot(variances, centers**2, ([iaxis], [0])))

        den = np.sum(values, axis=iaxis)
        den_err = np.sqrt(np.sum(variances, axis=iaxis))

        with np.errstate(invalid="ignore"):
            new_values = num / den
            new_variances = (num_err / den) ** 2 - (den_err * num / den**2) ** 2

        retval = self.__class__(*axes, storage=hist.storage.Mean())
        retval[...] = np.stack([count, new_values, count * new_variances], axis=-1)
        return retval

    def density(self) -> np.typing.NDArray[Any]:
        """
        Density NumPy array.
        """
        total = np.sum(self.values()) * functools.reduce(operator.mul, self.axes.widths)
        dens: np.typing.NDArray[Any] = self.values() / np.where(total > 0, total, 1)
        return dens

    def show(self, **kwargs: Any) -> Any:
        """
        Pretty print histograms to the console.
        """

        return histoprint.print_hist(self, **kwargs)

    def plot(
        self, *args: Any, overlay: str | None = None, **kwargs: Any
    ) -> Hist1DArtists | Hist2DArtists:
        """
        Plot method for BaseHist object.
        """
        ordered_traits = sum(t.ordered for t in self.axes.traits)
        discrete_traits = sum(t.discrete for t in self.axes.traits)
        _has_categorical = ordered_traits == discrete_traits == 1
        _project = _has_categorical or overlay is not None

        if self.ndim == 1 or (self.ndim == 2 and _project):
            return self.plot1d(*args, overlay=overlay, **kwargs)

        if self.ndim == 2:
            return self.plot2d(*args, **kwargs)

        raise NotImplementedError("Please project to 1D or 2D before calling plot")

    def plot1d(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        overlay: str | int | None = None,
        **kwargs: Any,
    ) -> Hist1DArtists:
        """
        Plot1d method for BaseHist object.
        """

        from hist import plot

        if self.ndim == 1:
            return plot.histplot(self, ax=ax, **_proc_kw_for_lw(kwargs))
        if overlay is None:
            (overlay,) = (i for i, ax in enumerate(self.axes) if ax.traits.discrete)
        assert overlay is not None
        cat_ax = self.axes[overlay]
        icats = np.arange(len(cat_ax))
        cats = cat_ax if cat_ax.traits.discrete else icats
        d1hists = [self[{overlay: cat}] for cat in icats]
        if "label" in kwargs:
            if not isinstance(kwargs["label"], str) and len(kwargs["label"]) == len(
                cats
            ):
                cats = kwargs["label"]
                kwargs.pop("label")
            elif isinstance(kwargs["label"], str):
                cats = [kwargs["label"]] * len(cats)
                kwargs.pop("label")
            else:
                raise ValueError(
                    f"label ``{kwargs['label']}`` not understood for {len(cats)} categories"
                )
        return plot.histplot(d1hists, ax=ax, label=cats, **_proc_kw_for_lw(kwargs))

    def plot2d(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> Hist2DArtists:
        """
        Plot2d method for BaseHist object.
        """

        from hist import plot

        return plot.hist2dplot(self, ax=ax, **_proc_kw_for_lw(kwargs))

    def plot2d_full(
        self,
        *,
        ax_dict: dict[str, matplotlib.axes.Axes] | None = None,
        **kwargs: Any,
    ) -> tuple[Hist2DArtists, Hist1DArtists, Hist1DArtists]:
        """
        Plot2d_full method for BaseHist object.

        Pass a dict of axes to ``ax_dict``, otherwise, the current figure will be used.
        """
        # Type judgement

        from hist import plot

        return plot.plot2d_full(self, ax_dict=ax_dict, **kwargs)

    def plot_ratio(
        self,
        other: hist.BaseHist
        | Callable[[np.typing.NDArray[Any]], np.typing.NDArray[Any]]
        | str,
        *,
        ax_dict: dict[str, matplotlib.axes.Axes] | None = None,
        **kwargs: Any,
    ) -> tuple[MainAxisArtists, RatiolikeArtists]:
        """
        ``plot_ratio`` method for ``BaseHist`` object.

        Return a tuple of artists following a structure of
        ``(main_ax_artists, subplot_ax_artists)``
        """

        from hist import plot

        return plot._plot_ratiolike(
            self, other, ax_dict=ax_dict, view="ratio", **kwargs
        )

    def plot_pull(
        self,
        func: Callable[[np.typing.NDArray[Any]], np.typing.NDArray[Any]] | str,
        *,
        ax_dict: dict[str, matplotlib.axes.Axes] | None = None,
        **kwargs: Any,
    ) -> tuple[FitResultArtists, RatiolikeArtists]:
        """
        ``plot_pull`` method for ``BaseHist`` object.

        Return a tuple of artists following a structure of
        ``(main_ax_artists, subplot_ax_artists)``
        """

        from hist import plot

        return plot._plot_ratiolike(self, func, ax_dict=ax_dict, view="pull", **kwargs)

    def plot_pie(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> Any:
        from hist import plot

        return plot.plot_pie(self, ax=ax, **kwargs)

    def stack(self, axis: int | str) -> hist.stack.Stack:
        """
        Returns a stack from a normal histogram axes.
        """
        if self.ndim < 2:
            raise RuntimeError("Cannot stack with less than two axis")
        stack_histograms: Iterator[BaseHist] = [  # type: ignore[assignment]
            self[{axis: i}] for i in range(len(self.axes[axis]))
        ]
        for name, h in zip(self.axes[axis], stack_histograms):
            h.name = name

        return hist.stack.Stack(*stack_histograms)

    def integrate(
        self,
        name: int | str,
        i_or_list: InnerIndexing | list[str | int] | None = None,
        j: InnerIndexing | None = None,
    ) -> Self | float | bh.accumulators.Accumulator:
        if isinstance(i_or_list, list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # TODO: We could teach the typing system that list always returns Self type
                selection: Self = self[{name: i_or_list}]  # type: ignore[assignment, dict-item]
                return selection[{name: slice(0, len(i_or_list), sum)}]

        return self[{name: slice(i_or_list, j, sum)}]

    def sum(self, flow: bool = False) -> float | bh.accumulators.Accumulator:
        """
        Compute the sum over the histogram bins (optionally including the flow bins).
        """
        # TODO: This method will go away once we can guarantee a modern boost-histogram (1.3.2 or better)
        if any(x == 0 for x in (self.axes.extent if flow else self.axes.size)):
            storage = self.storage_type
            if issubclass(storage, (bh.storage.AtomicInt64, bh.storage.Int64)):
                return 0
            if issubclass(storage, (bh.storage.Double, bh.storage.Unlimited)):
                return 0.0
            if issubclass(storage, bh.storage.Weight):
                return bh.accumulators.WeightedSum(0, 0)
            if issubclass(storage, bh.storage.Mean):
                return bh.accumulators.Mean(0, 0, 0)
            if issubclass(storage, bh.storage.WeightedMean):
                return bh.accumulators.WeightedMean(0, 0, 0, 0)
            raise AssertionError(f"Unsupported storage type {storage}")

        return super().sum(flow=flow)
