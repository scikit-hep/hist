import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from uncertainties import correlated_values, unumpy
from boost_histogram import Histogram
from typing import Callable, Optional, Tuple


class BaseHist(Histogram):
    def __init__(self, *args, **kwargs):
        """
            Initialize Hist object. Axis params must contain the names.
        """

        super().__init__(*args, **kwargs)
        self.names: dict = dict()
        for ax in self.axes:
            if ax.name in self.names:
                raise Exception(
                    "Hist instance cannot contain axes with duplicated names."
                )
            else:
                self.names[ax.name] = True

    def pull_plot(
        self,
        func: Callable,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        pull_ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        **kwargs,
    ) -> Tuple[
        Optional[matplotlib.figure.Figure],
        Optional[matplotlib.axes._subplots.SubplotBase],
        Optional[matplotlib.axes._subplots.SubplotBase],
    ]:
        """
        Pull_plot method for Hist object.
        """

        # Type judgement
        if callable(func) == False:
            raise TypeError("Callable parameter func is supported in pull plot.")
        if len(self.axes) != 1:
            raise TypeError("Only 1D-histogram has pull plot.")

        """
        Computation and Fit
        """

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
        pulls = (self.view() - y_nv) / yerr
        pulls = [x if np.abs(x) != np.inf else 0 for x in pulls]

        """
        Default Figure: construct the figure and axes
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(4, 4, wspace=0.5, hspace=0.5)
        else:
            grid = fig.add_gridspec(4, 4, wspace=0.5, hspace=0.5)

        if ax is None:
            ax = fig.add_subplot(grid[0:3, :])
        else:
            pass

        if pull_ax is None:
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
                    raise ValueError(f"'pp_label' not needed.")
                pp_kwargs[kw[3:]] = kwargs[kw]

        if "pp_num" in kwargs:
            kwargs.pop("pp_num")
        for k in pp_kwargs:
            kwargs.pop("pp_" + k)

        # judge whether some arguements left
        if len(kwargs):
            raise ValueError(f"'{list(kwargs.keys())[0]}' not needed.")

        """
        Main: plot the pulls using Matplotlib errorbar and plot methods
        """
        ax.errorbar(
            self.axes.centers[0], self.view(), yerr, label="Histogram Data", **eb_kwargs
        )
        (line,) = ax.plot(self.axes.centers[0], fit, label="Fitting Value", **fp_kwargs)
        ax.fill_between(
            self.axes.centers[0],
            y_nv - y_sd,
            y_nv + y_sd,
            color=line.get_color(),
            label="Uncertainty",
            **ub_kwargs,
        )
        legend = ax.legend(loc=0)

        ax.set_ylabel("Counts")
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

        if self.axes[0].name:
            pull_ax.set_xlabel(self.axes[0].name)
        else:
            pull_ax.set_xlabel(self.axes[0].title)
        pull_ax.set_ylabel("Pull")

        return fig, ax, pull_ax
