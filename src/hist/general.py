from .core import BaseHist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from typing import Callable


class Hist(BaseHist):
    def __init__(self, *args, **kwargs):
        """
            Initialize Hist object. Axis params must contain the names.
        """

        super(BaseHist, self).__init__(*args, **kwargs)
        self.names: dict = dict()
        for ax in self.axes:
            if not ax.name:
                raise Exception(
                    "Each axes in the Hist instance should have a name."
                )
            elif ax.name in self.names:
                raise Exception(
                    "Hist instance cannot contain axes with duplicated names."
                )
            else:
                self.names[ax.name] = True
    
    def pull_plot(self, func: Callable, fig=None, ax=None, pull_ax=None, **kwargs):
        """
        Pull_plot method for Hist object.
        """

        # Type judgement
        if callable(func) == False:
            raise TypeError("Callable parameter func is supported in pull plot.")

        """
        Computation and Fit
        """
        # Compute PDF values
        values = func(*self.axes.centers) * self.sum() * self.axes[0].widths
        yerr = np.sqrt(self.view())

        # Compute fit values: using func as fit model
        popt, pcov = curve_fit(f=func, xdata=self.axes.centers[0], ydata=self.view())
        fit = func(self.axes.centers[0], *popt)

        # Compute pulls: containing no INF values
        pulls = (self.view() - values) / yerr
        pulls = [x if np.abs(x) != np.inf else 0 for x in pulls]

        """
        Default Figure: construct the figure and axes
        """
        if fig == None:
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(4, 4, wspace=0.5, hspace=0.5)
        else:
            grid = fig.add_gridspec(4, 4, wspace=0.5, hspace=0.5)

        if ax == None:
            ax = fig.add_subplot(grid[0:3, :])
        else:
            pass

        if pull_ax == None:
            pull_ax = fig.add_subplot(grid[3, :], sharex=ax)
        else:
            pass

        """
        Keyword Argument Conversion: convert the kwargs to several independent args
        """
        # error bar keyword arguments
        eb_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "eb":
                eb_kwargs[kw[3:]] = kwargs[kw]

        for k in eb_kwargs:
            kwargs.pop("eb_" + k)

        # value plot keyword arguments
        vp_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "vp":
                vp_kwargs[kw[3:]] = kwargs[kw]

        for k in vp_kwargs:
            kwargs.pop("vp_" + k)

        # mean plot keyword arguments
        mp_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "mp":
                mp_kwargs[kw[3:]] = kwargs[kw]

        for k in mp_kwargs:
            kwargs.pop("mp_" + k)

        # fit plot keyword arguments
        fp_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:2] == "fp":
                fp_kwargs[kw[3:]] = kwargs[kw]

        for k in fp_kwargs:
            kwargs.pop("fp_" + k)

        # bar plot keyword arguments
        bar_kwargs = dict()
        for kw in kwargs.keys():
            if kw[:3] == "bar":
                # disable bar width arguments
                if kw == "bar_width":
                    raise KeyError("'bar_width' not needed.")
                bar_kwargs[kw[4:]] = kwargs[kw]

        for k in bar_kwargs:
            kwargs.pop("bar_" + k)

        # patch plot keyword arguments
        pp_kwargs, pp_num = dict(), 3
        for kw in kwargs.keys():
            if kw[:2] == "pp":
                # allow pp_num
                if kw == "pp_num":
                    pp_num = kwargs[kw]
                    continue
                pp_kwargs[kw[3:]] = kwargs[kw]

        if "pp_num" in kwargs:
            kwargs.pop("pp_num")
        for k in pp_kwargs:
            kwargs.pop("pp_" + k)

        # judge whether some arguements left
        if len(kwargs):
            raise TypeError(f"'{list(kwargs.keys())[0]}' not needed.")

        """
        Main: plot the pulls using Matplotlib errorbar and plot methods
        """
        ax.errorbar(self.axes.centers[0], self.view(), yerr, **eb_kwargs)
        ax.plot(self.axes.centers[0], values, **vp_kwargs)
        ax.plot(self.axes.centers[0], (self.view() + values) / 2, **mp_kwargs)
        ax.plot(self.axes.centers[0], fit, **fp_kwargs)

        fig.add_axes(ax)

        """
        Pull: plot the pulls using Matplotlib bar method
        """
        left_edge = self.axes.edges[0][0]
        right_edge = self.axes.edges[-1][-1]
        width = (right_edge - left_edge) / len(pulls)
        pull_ax.bar(self.axes.centers[0], pulls, width=width, **bar_kwargs)

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

        fig.add_axes(pull_ax)

        return fig, ax, pull_ax
