# -*- coding: utf-8 -*-
from .axestuple import NamedAxesTuple
from .quick_construct import MetaConstructor
from .utils import set_family, HIST_FAMILY
from .storage import Storage

import warnings
import functools
import operator
import histoprint

import numpy as np
import boost_histogram as bh

from typing import Callable, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING
from .svgplots import html_hist, svg_hist_1d, svg_hist_1d_c, svg_hist_2d, svg_hist_nd


if TYPE_CHECKING:
    from mplhep.plot import Hist1DArtists, Hist2DArtists
    import matplotlib.axes


@set_family(HIST_FAMILY)
class BaseHist(bh.Histogram, metaclass=MetaConstructor):
    __slots__ = ()

    def __init__(self, *args, storage: Optional[Storage] = None, metadata=None):
        """
        Initialize BaseHist object. Axis params can contain the names.
        """
        self._hist: Any = None
        self.axes: NamedAxesTuple

        if len(args):
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

    def _repr_html_(self):
        if self.ndim == 1:
            if self.axes[0].options.circular:
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

    def density(self) -> np.ndarray:
        """
        Density numpy array.
        """
        total = self.sum() * functools.reduce(operator.mul, self.axes.widths)
        return self.view() / np.where(total > 0, total, 1)

    def show(self, **kwargs):
        """
        Pretty print histograms to the console.
        """

        return histoprint.print_hist(self, **kwargs)

    def plot(self, *args, **kwargs) -> "Union[Hist1DArtists, Hist2DArtists]":
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
        ax: "Optional[matplotlib.axes.Axes]" = None,
        **kwargs,
    ) -> "Hist1DArtists":
        """
        Plot1d method for BaseHist object.
        """

        import hist.plot

        return hist.plot.histplot(self, ax=ax, **kwargs)

    def plot2d(
        self,
        *,
        ax: "Optional[matplotlib.axes.Axes]" = None,
        **kwargs,
    ) -> "Hist2DArtists":
        """
        Plot2d method for BaseHist object.
        """

        import hist.plot

        return hist.plot.hist2dplot(self, ax=ax, **kwargs)

    def plot2d_full(
        self,
        *,
        ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
        **kwargs,
    ) -> "Tuple[Hist2DArtists, Hist1DArtists, Hist1DArtists]":
        """
        Plot2d_full method for BaseHist object.

        Pass a dict of axes to ``ax_dict``, otherwise, the current figure will be used.
        """
        # Type judgement

        import hist.plot

        return hist.plot.plot2d_full(self, ax_dict=ax_dict, **kwargs)

    def plot_pull(
        self,
        func: Callable,
        *,
        ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
        **kwargs,
    ) -> "Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]":
        """
        Plot_pull method for BaseHist object.
        """

        import hist.plot

        return hist.plot.plot_pull(self, func, ax_dict=ax_dict, **kwargs)
