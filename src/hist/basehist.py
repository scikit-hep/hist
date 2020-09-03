# -*- coding: utf-8 -*-
import numpy as np
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
import warnings
from scipy.optimize import curve_fit
from uncertainties import correlated_values, unumpy
from typing import Callable, Optional, Tuple, Union, Dict, List, Any
import functools
import operator
import histoprint

import hist.utils
import hist.storage
from hist.storage import Storage

from .axis import Regular, Boolean, Variable, Integer, IntCategory, StrCategory
from .axestuple import NamedAxesTuple


class always_normal_method:
    def __get__(self, instance, owner=None):
        self.instance = instance or owner()
        return self

    def __init__(self, method):
        self.method = method
        self.instance = None

    def __call__(self, *args, **kwargs):
        return self.method(self.instance, *args, **kwargs)


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class BaseHist(bh.Histogram):
    __slots__ = ("_ax", "_storage_proxy")

    def __init__(self, *args, storage: Optional[Storage] = None, metadata=None):
        """
            Initialize BaseHist object. Axis params can contain the names.
        """
        # TODO: Make a base class type Axis for Hist
        self._ax: List[bh.axis.Axes] = []
        self._hist: Any = None
        self._storage_proxy: Optional[Storage] = None
        self.axes: NamedAxesTuple

        if len(args):
            self._hist = None
            self._ax = []
            if isinstance(storage, type):
                msg = (
                    f"Please use '{storage.__name__}()' instead of '{storage.__name__}'"
                )
                warnings.warn(msg)
                storage = storage()
            super().__init__(*args, storage=storage, metadata=metadata)
            valid_names = [ax.name for ax in self.axes if ax.name]
            if len(valid_names) != len(set(valid_names)):
                raise KeyError(
                    f"{self.__class__.__name__} instance cannot contain axes with duplicated names"
                )
            for i, ax in enumerate(self.axes):
                # label will return name if label is not set, so this is safe
                if not ax.label:
                    ax.label = f"Axis {i}"

    def _generate_axes_(self) -> NamedAxesTuple:
        """
        This is called to fill in the axes. Subclasses can override it if they need
        to change the axes tuple.
        """

        return NamedAxesTuple(self._axis(i) for i in range(self.ndim))

    @always_normal_method
    def Reg(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(Regular(*args, **kwargs))
        return self

    @always_normal_method
    def Sqrt(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(Regular(*args, transform=hist.axis.transform.sqrt, **kwargs))
        return self

    @always_normal_method
    def Log(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(Regular(*args, transform=hist.axis.transform.log, **kwargs))
        return self

    @always_normal_method
    def Pow(self, *args, power, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(
            Regular(*args, transform=hist.axis.transform.Pow(power), **kwargs)
        )
        return self

    @always_normal_method
    def Func(self, *args, forward, inverse, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(
            Regular(
                *args,
                transform=hist.axis.transform.Function(forward, inverse),
                **kwargs,
            )
        )
        return self

    @always_normal_method
    def Bool(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(Boolean(*args, **kwargs))
        return self

    @always_normal_method
    def Var(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(Variable(*args, **kwargs))
        return self

    @always_normal_method
    def Int(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(Integer(*args, **kwargs))
        return self

    @always_normal_method
    def IntCat(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(IntCategory(*args, **kwargs))
        return self

    @always_normal_method
    def StrCat(self, *args, **kwargs):
        if self._hist:
            raise RuntimeError("Cannot add an axis to an existing histogram")
        self._ax.append(StrCategory(*args, **kwargs))
        return self

    @always_normal_method
    def Double(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.Double()
        return self

    @always_normal_method
    def Int64(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.Int64()
        return self

    @always_normal_method
    def AtomicInt64(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.AtomicInt64()
        return self

    @always_normal_method
    def Weight(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.Weight()
        return self

    @always_normal_method
    def Mean(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.Mean()
        return self

    @always_normal_method
    def WeightedMean(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.WeightedMean()
        return self

    @always_normal_method
    def Unlimited(self):
        if self._hist:
            raise RuntimeError("Cannot add a storage to an existing histogram")
        elif self._storage_proxy:
            raise RuntimeError("Cannot add another storage")

        self._storage_proxy = hist.storage.Unlimited()
        return self

    def __getattribute__(self, item):

        if (
            not object.__getattribute__(self, "_hist")
            and not isinstance(
                object.__getattribute__(self, item), always_normal_method
            )
            and item not in {"_hist", "_ax", "_storage_proxy"}
        ):
            # Make histogram real here
            ax = object.__getattribute__(self, "_ax")
            storage = (
                object.__getattribute__(self, "_storage_proxy") or bh.storage.Double()
            )
            object.__getattribute__(self, "__init__")(*ax, storage=storage)

        return object.__getattribute__(self, item)

    def _name_to_index(self, name: str) -> int:
        """
            Transform axis name to axis index, given axis name, return axis \
            index.
        """
        for index, axis in enumerate(self.axes):
            if name == axis.name:
                return index

        raise ValueError("The axis names could not be found")

    def project(self, *args: Union[int, str]):
        """
        Projection of axis idx.
        """
        int_args = [self._name_to_index(a) if isinstance(a, str) else a for a in args]
        return super().project(*int_args)

    def fill(
        self, *args, weight=None, sample=None, threads: Optional[int] = None, **kwargs
    ):
        """
        Insert data into the histogram using names and indices, return
        a Hist object.
        """

        data_dict = {
            self._name_to_index(k) if isinstance(k, str) else k: v
            for k, v in kwargs.items()
        }

        if set(data_dict) != set(range(len(args), self.ndim)):
            raise TypeError("All axes must be accounted for in fill")

        data = (data_dict[i] for i in range(len(args), self.ndim))

        total_data = tuple(args) + tuple(data)  # Python 2 can't unpack twice
        return super().fill(*total_data, weight=weight, sample=sample, threads=threads)

    def _loc_shortcut(self, x):
        """
            Convert some specific indices to location.
        """

        if isinstance(x, slice):
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

    def _step_shortcut(self, x):
        """
            Convert some specific indices to step.
        """

        if isinstance(x, complex):
            if x.real != 0:
                raise ValueError("The step should not have real part")
            elif x.imag % 1 != 0:
                raise ValueError("The imaginary part should be an integer")
            else:
                return bh.rebin(int(x.imag))
        else:
            return x

    def _index_transform(self, index):
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

        elif not hasattr(index, "__iter__"):
            index = (index,)

        return tuple(self._loc_shortcut(v) for v in index)

    def __getitem__(self, index):
        """
            Get histogram item.
        """

        return super().__getitem__(self._index_transform(index))

    def __setitem__(self, index, value):
        """
            Set histogram item.
        """

        return super().__setitem__(self._index_transform(index), value)

    def density(self):
        """
        Density numpy array.
        """
        # ToDo: maybe should not be filled
        # ToDo: flow should be removed
        return self.view() / (
            self.sum() * functools.reduce(operator.mul, self.axes.widths)
        )

    def show(self, **kwargs):
        """
        Pretty print histograms to the console.
        """

        return histoprint.print_hist(self, **kwargs)

    def plot(self, *args, **kwargs) -> matplotlib.artist.Artist:
        """
        Plot method for BaseHist object.
        """
        if self.ndim == 1:
            return self.plot1d(*args, **kwargs)
        elif self.ndim == 2:
            return self.plot2d(*args, **kwargs)
        else:
            raise NotImplementedError("Please project to 1D or 2D before calling plot")

    def plot1d(
        self,
        *,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs,
    ) -> matplotlib.lines.Line2D:
        """
        Plot1d method for BaseHist object.
        """
        # Type judgement
        if self.ndim != 1:
            raise TypeError("Only 1D-histogram has plot1d")

        # Default Figure: construct the figure and axes
        if fig is not None and ax is not None:
            if fig != ax.figure:
                raise TypeError(
                    "You cannot pass both a figure and axes that are not shared!"
                )

        elif fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        elif fig is not None:
            ax = fig.add_subplot(111)

        elif ax is not None:
            fig = ax.figure

        (lines,) = ax.step(
            self.axes.edges[0][:-1],
            self.project(self.axes[0].name or 0).view(),
            **kwargs,
        )

        ax.set_xlabel(self.axes[0].label)
        ax.set_ylabel("Counts")

        return lines

    def plot2d(
        self,
        *,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs,
    ) -> matplotlib.collections.QuadMesh:
        """
        Plot2d method for BaseHist object.
        """
        # Type judgement
        if self.ndim != 2:
            raise TypeError("Only 2D-histogram has plot2d")

        # Default Figure: construct the figure and axes
        if fig is not None and ax is not None:
            if fig != ax.figure:
                raise TypeError(
                    "You cannot pass both a figure and axes that are not shared!"
                )

        elif fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        elif fig is not None:
            ax = fig.add_subplot(111)

        elif ax is not None:
            fig = ax.figure

        else:
            pass

        # Plot: plot the 2d-histogram
        X, Y = self.axes.edges
        res = ax.pcolormesh(X.T, Y.T, self.view().T, **kwargs)

        ax.set_xlabel(self.axes[0].label)
        ax.set_ylabel(self.axes[1].label)

        return res

    def plot2d_full(
        self,
        *,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax_dict: Optional[Dict[str, matplotlib.axes.Axes]] = None,
        **kwargs,
    ) -> Tuple[
        matplotlib.collections.QuadMesh,
        matplotlib.lines.Line2D,
        matplotlib.lines.Line2D,
    ]:
        """
        Plot2d_full method for BaseHist object.
        """
        # Type judgement
        if self.ndim != 2:
            raise TypeError("Only 2D-histogram has plot2d_full")

        if ax_dict is None:
            ax_dict = dict()

        if len(ax_dict) != 0 and len(ax_dict) != 3:
            raise ValueError("All axes should be all given or none at all")

        # Default Figure: construct the figure and axes
        if fig is not None and len(ax_dict):
            for kw, ax in ax_dict.items():
                if kw not in {"main_ax", "top_ax", "side_ax"}:
                    raise TypeError(f"{kw} is not supported in the ax_dict")
                if fig != ax.figure:
                    raise TypeError(
                        "You cannot pass both a figure and axes that are not shared"
                    )

            main_ax = ax_dict["main_ax"]
            top_ax = ax_dict["top_ax"]
            side_ax = ax_dict["side_ax"]

        elif fig is None and not len(ax_dict):
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(4, 4, hspace=0, wspace=0)
            main_ax = fig.add_subplot(grid[1:4, 0:3])
            top_ax = fig.add_subplot(grid[0:1, 0:3], sharex=main_ax)
            side_ax = fig.add_subplot(grid[1:4, 3:4], sharey=main_ax)

        elif fig is not None and not len(ax_dict):
            grid = fig.add_gridspec(4, 4, hspace=0, wspace=0)
            main_ax = fig.add_subplot(grid[1:4, 0:3])
            top_ax = fig.add_subplot(grid[0:1, 0:3], sharex=main_ax)
            side_ax = fig.add_subplot(grid[1:4, 3:4], sharey=main_ax)

        elif fig is None and len(ax_dict):
            main_ax = ax_dict["main_ax"]
            top_ax = ax_dict["top_ax"]
            side_ax = ax_dict["side_ax"]

        # Keyword Argument Conversion: convert the kwargs to several independent args

        # main plot keyword arguments
        main_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:4] == "main":
                main_kwargs[kw[5:]] = kwargs[kw]

        for k in main_kwargs:
            kwargs.pop("main_" + k)

        # top plot keyword arguments
        top_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:3] == "top":
                top_kwargs[kw[4:]] = kwargs[kw]

        for k in top_kwargs:
            kwargs.pop("top_" + k)

        # side plot keyword arguments
        side_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:4] == "side":
                side_kwargs[kw[5:]] = kwargs[kw]

        for k in side_kwargs:
            kwargs.pop("side_" + k)

        # judge whether some arguments left
        if len(kwargs):
            raise ValueError(f"'{list(kwargs.keys())[0]}' not needed")

        # Plot: plot the 2d-histogram

        # main plot
        X, Y = self.axes.edges
        main_res = main_ax.pcolormesh(X.T, Y.T, self.view().T, **main_kwargs)

        main_ax.set_xlabel(self.axes[0].label)
        main_ax.set_ylabel(self.axes[1].label)

        # top plot
        top_res = top_ax.step(
            self.axes[0].edges[:-1],
            self.project(self.axes[0].name or 0).view(),
            **top_kwargs,
        )

        top_ax.spines["top"].set_visible(False)
        top_ax.spines["right"].set_visible(False)
        top_ax.xaxis.set_visible(False)

        top_ax.set_ylabel("Counts")

        # side plot
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)

        side_res = side_ax.step(
            self.axes[1].edges[:-1],
            -self.project(self.axes[1].name or 1).view(),
            transform=rot + base,
            **side_kwargs,
        )

        side_ax.spines["top"].set_visible(False)
        side_ax.spines["right"].set_visible(False)
        side_ax.yaxis.set_visible(False)
        side_ax.set_xlabel("Counts")

        return main_res, top_res, side_res

    def plot_pull(
        self,
        func: Callable,
        *,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax_dict: Optional[Dict[str, matplotlib.axes.Axes]] = None,
        **kwargs,
    ) -> Tuple[
        matplotlib.collections.QuadMesh, matplotlib.lines.Line2D,
    ]:
        """
        Plot_pull method for BaseHist object.
        """

        # Type judgement
        if not callable(func):
            raise TypeError(
                f"Callable parameter func is supported for {self.__class__.__name__} in plot pull"
            )
        if self.ndim != 1:
            raise TypeError("Only 1D-histogram has pull plot")

        # Default Figure: construct the figure and axes
        if ax_dict is None:
            ax_dict = {}

        if len(ax_dict) != 0 and len(ax_dict) != 2:
            raise ValueError("All axes should be all given or none at all")

        if fig is not None and len(ax_dict):
            for kw, ax in ax_dict.items():
                if kw not in {"main_ax", "pull_ax"}:
                    raise TypeError(f"{kw} is not supported in the ax_dict")
                if fig != ax.figure:
                    raise TypeError(
                        "You cannot pass both a figure and axes that are not shared"
                    )

            main_ax = ax_dict["main_ax"]
            pull_ax = ax_dict["pull_ax"]

        elif fig is None and not len(ax_dict):
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
            main_ax = fig.add_subplot(grid[0])
            pull_ax = fig.add_subplot(grid[1], sharex=main_ax)

        elif fig is not None and not len(ax_dict):
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
            main_ax = fig.add_subplot(grid[0])
            pull_ax = fig.add_subplot(grid[1], sharex=main_ax)

        elif fig is None and len(ax_dict):
            main_ax = ax_dict["main_ax"]
            pull_ax = ax_dict["pull_ax"]

        # Computation and Fit
        yerr = np.sqrt(self.view())

        # Compute fit values: using func as fit model
        popt, pcov = curve_fit(f=func, xdata=self.axes.centers[0], ydata=self.view())
        fit = func(self.axes.centers[0], *popt)

        # Compute uncertainty
        copt = correlated_values(popt, pcov)
        y_unc = func(self.axes.centers[0], *copt)
        y_nv = unumpy.nominal_values(y_unc)
        y_sd = unumpy.std_devs(y_unc)

        # Compute pulls: containing no INF values
        with np.errstate(divide="ignore"):
            pulls = (self.view() - y_nv) / yerr

        pulls[np.isnan(pulls)] = 0
        pulls[np.isinf(pulls)] = 0

        # Keyword Argument Conversion: convert the kwargs to several independent args
        # error bar keyword arguments
        eb_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "eb":
                # disabled argument
                if kw == "eb_label":
                    pass
                else:
                    eb_kwargs[kw[3:]] = kwargs[kw]

        for k in eb_kwargs:
            kwargs.pop("eb_" + k)

        # value plot keyword arguments
        vp_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "vp":
                # disabled argument
                if kw == "vp_label":
                    pass
                else:
                    vp_kwargs[kw[3:]] = kwargs[kw]

        for k in vp_kwargs:
            kwargs.pop("vp_" + k)

        # fit plot keyword arguments
        fp_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "fp":
                # disabled argument
                if kw == "fp_label":
                    pass
                else:
                    fp_kwargs[kw[3:]] = kwargs[kw]

        for k in fp_kwargs:
            kwargs.pop("fp_" + k)

        # uncertainty band keyword arguments
        ub_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "ub":
                # disabled arguments
                if kw == "ub_color":
                    pass
                if kw == "ub_label":
                    pass
                else:
                    ub_kwargs[kw[3:]] = kwargs[kw]

        for k in ub_kwargs:
            kwargs.pop("ub_" + k)

        # bar plot keyword arguments
        bar_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:3] == "bar":
                # disabled arguments
                if kw == "bar_width":
                    pass
                if kw == "bar_label":
                    pass
                else:
                    bar_kwargs[kw[4:]] = kwargs[kw]

        for k in bar_kwargs:
            kwargs.pop("bar_" + k)

        # patch plot keyword arguments
        pp_kwargs, pp_num = dict(), 3
        for kw in kwargs.keys():
            if kw[:2] == "pp":
                # new argument
                if kw == "pp_num":
                    pp_num = kwargs[kw]
                    continue
                # disabled argument
                if kw == "pp_label":
                    raise ValueError("'pp_label' not needed")
                pp_kwargs[kw[3:]] = kwargs[kw]

        if "pp_num" in kwargs:
            kwargs.pop("pp_num")
        for k in pp_kwargs:
            kwargs.pop("pp_" + k)

        # judge whether some arguments left
        if kwargs:
            raise ValueError(f"'{list(kwargs.keys())[0]}' not needed")

        # Main: plot the pulls using Matplotlib errorbar and plot methods
        main_res = main_ax.errorbar(
            self.axes.centers[0], self.view(), yerr, label="Histogram Data", **eb_kwargs
        )
        (line,) = main_ax.plot(
            self.axes.centers[0], fit, label="Fitting Value", **fp_kwargs
        )
        main_res = main_ax.fill_between(
            self.axes.centers[0],
            y_nv - y_sd,
            y_nv + y_sd,
            color=line.get_color(),
            label="Uncertainty",
            **ub_kwargs,
        )
        main_ax.legend(loc=0)
        main_ax.set_ylabel("Counts")

        # Pull: plot the pulls using Matplotlib bar method
        left_edge = self.axes.edges[0][0]
        right_edge = self.axes.edges[-1][-1]
        width = (right_edge - left_edge) / len(pulls)
        pull_res = pull_ax.bar(self.axes.centers[0], pulls, width=width, **bar_kwargs)

        patch_height = max(np.abs(pulls)) / pp_num
        patch_width = width * len(pulls)
        for i in range(pp_num):
            # gradient color patches
            if "alpha" in pp_kwargs:
                pp_kwargs["alpha"] *= np.power(0.618, i)
            else:
                pp_kwargs["alpha"] = 0.618 * np.power(0.618, i)

            upRect_startpoint = [left_edge, i * patch_height]
            upRect = patches.Rectangle(
                upRect_startpoint, patch_width, patch_height, **pp_kwargs
            )
            pull_ax.add_patch(upRect)
            downRect_startpoint = [left_edge, -(i + 1) * patch_height]
            downRect = patches.Rectangle(
                downRect_startpoint, patch_width, patch_height, **pp_kwargs
            )
            pull_ax.add_patch(downRect)
        plt.xlim(left_edge, right_edge)

        pull_ax.set_xlabel(self.axes[0].label)
        pull_ax.set_ylabel("Pull")

        return main_res, pull_res
