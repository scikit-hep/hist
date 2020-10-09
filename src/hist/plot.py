# -*- coding: utf-8 -*-
import sys
from typing import Dict, Any, Optional, Set, Callable, Tuple
import numpy as np

try:
    from mplhep.plot import histplot, hist2dplot
    from mplhep.plot import Hist1DArtists, Hist2DArtists

    import matplotlib.axes
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms
except ImportError:
    print(
        "Hist requires mplhep to plot, either install hist[plot] or mplhep",
        file=sys.stderr,
    )
    raise


__all__ = ("histplot", "hist2dplot", "plot2d_full", "plot_pull")


def _filter_dict(
    dict: Dict[str, Any], prefix: str, *, ignore: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Keyword argument conversion: convert the kwargs to several independent args, pulling
    them out of the dict given.
    """
    ignore_set: Set[str] = ignore or set()
    return {
        key[len(prefix) :]: dict.pop(key)
        for key in list(dict)
        if key.startswith(prefix) and key not in ignore_set
    }


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
    if self.ndim != 2:
        raise TypeError("Only 2D-histogram has plot2d_full")

    if ax_dict is None:
        ax_dict = dict()

    # Default Figure: construct the figure and axes
    if ax_dict:
        try:
            main_ax = ax_dict["main_ax"]
            top_ax = ax_dict["top_ax"]
            side_ax = ax_dict["side_ax"]
        except KeyError:
            raise ValueError("All axes should be all given or none at all")

    else:
        fig = plt.gcf()

        grid = fig.add_gridspec(
            2, 2, hspace=0, wspace=0, width_ratios=[4, 1], height_ratios=[1, 4]
        )
        main_ax = fig.add_subplot(grid[1, 0])
        top_ax = fig.add_subplot(grid[0, 0], sharex=main_ax)
        side_ax = fig.add_subplot(grid[1, 1], sharey=main_ax)

    # keyword arguments
    main_kwargs = _filter_dict(kwargs, "main_", ignore={"main_cbar"})
    top_kwargs = _filter_dict(kwargs, "top_")
    side_kwargs = _filter_dict(kwargs, "side_")

    # judge whether some arguments left
    if len(kwargs):
        raise ValueError(f"{set(kwargs)} not needed")

    # Plot: plot the 2d-histogram

    # main plot
    main_art = hist2dplot(self, ax=main_ax, cbar=False, **main_kwargs)

    # top plot
    top_art = histplot(
        self.project(self.axes[0].name or 0),
        ax=top_ax,
        **top_kwargs,
    )

    top_ax.spines["top"].set_visible(False)
    top_ax.spines["right"].set_visible(False)
    top_ax.xaxis.set_visible(False)

    top_ax.set_ylabel("Counts")

    # side plot
    base = side_ax.transData
    rot = transforms.Affine2D().rotate_deg(90).scale(-1, 1)

    side_art = histplot(
        self.project(self.axes[1].name or 1),
        ax=side_ax,
        transform=rot + base,
        **side_kwargs,
    )

    side_ax.spines["top"].set_visible(False)
    side_ax.spines["right"].set_visible(False)
    side_ax.yaxis.set_visible(False)
    side_ax.set_xlabel("Counts")

    return main_art, top_art, side_art


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

    try:
        from scipy.optimize import curve_fit
        from uncertainties import correlated_values, unumpy
    except ImportError:
        print(
            "Hist.plot_pull requires scipy and uncertainties. Please install hist[plot] or manually install dependencies.",
            file=sys.stderr,
        )
        raise

    # Type judgement
    if not callable(func):
        msg = f"Callable parameter func is supported for {self.__class__.__name__} in plot pull"
        raise TypeError(msg)

    if self.ndim != 1:
        raise TypeError("Only 1D-histogram supports pull plot, try projecting to 1D")

    if ax_dict:
        try:
            main_ax = ax_dict["main_ax"]
            pull_ax = ax_dict["pull_ax"]
        except KeyError:
            raise ValueError("All axes should be all given or none at all")
    else:
        fig = plt.gcf()
        grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

        main_ax = fig.add_subplot(grid[0])
        pull_ax = fig.add_subplot(grid[1], sharex=main_ax)

    # Computation and Fit
    view = self.view()
    values = view.value if hasattr(view, "value") else view
    yerr = view.variance if hasattr(view, "variance") else np.sqrt(values)

    # Compute fit values: using func as fit model
    popt, pcov = curve_fit(f=func, xdata=self.axes[0].centers, ydata=values)
    fit = func(self.axes[0].centers, *popt)

    # Compute uncertainty
    copt = correlated_values(popt, pcov)
    y_unc = func(self.axes[0].centers, *copt)
    y_nv = unumpy.nominal_values(y_unc)
    y_sd = unumpy.std_devs(y_unc)

    # Compute pulls: containing no INF values
    with np.errstate(divide="ignore"):
        pulls = (values - y_nv) / yerr

    pulls[np.isnan(pulls)] = 0
    pulls[np.isinf(pulls)] = 0

    # Keyword Argument Conversion: convert the kwargs to several independent args

    # error bar keyword arguments
    eb_kwargs = _filter_dict(kwargs, "eb_")
    eb_kwargs.setdefault("label", "Histogram Data")

    # fit plot keyword arguments
    fp_kwargs = _filter_dict(kwargs, "fp_")
    fp_kwargs.setdefault("label", "Fitting Value")

    # uncertainty band keyword arguments
    ub_kwargs = _filter_dict(kwargs, "ub_")
    ub_kwargs.setdefault("label", "Uncertainty")

    # bar plot keyword arguments
    bar_kwargs = _filter_dict(kwargs, "bar_", ignore={"bar_width"})

    # patch plot keyword arguments
    pp_kwargs = _filter_dict(kwargs, "pp_", ignore={"pp_num"})
    pp_num = kwargs.pop("pp_num", 5)

    # Judge whether some arguments are left
    if kwargs:
        raise ValueError(f"{set(kwargs)}' not needed")

    # Main: plot the pulls using Matplotlib errorbar and plot methods
    main_ax.errorbar(self.axes.centers[0], values, yerr, **eb_kwargs)

    (line,) = main_ax.plot(self.axes.centers[0], fit, **fp_kwargs)

    # Uncertainty band
    ub_kwargs.setdefault("color", line.get_color())
    main_ax.fill_between(
        self.axes.centers[0],
        y_nv - y_sd,
        y_nv + y_sd,
        **ub_kwargs,
    )
    main_ax.legend(loc=0)
    main_ax.set_ylabel("Counts")

    # Pull: plot the pulls using Matplotlib bar method
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
            pp_kwargs["alpha"] = 0.5 * np.power(0.618, i)

        upRect_startpoint = (left_edge, i * patch_height)
        upRect = patches.Rectangle(
            upRect_startpoint, patch_width, patch_height, **pp_kwargs
        )
        pull_ax.add_patch(upRect)
        downRect_startpoint = (left_edge, -(i + 1) * patch_height)
        downRect = patches.Rectangle(
            downRect_startpoint, patch_width, patch_height, **pp_kwargs
        )
        pull_ax.add_patch(downRect)

    plt.xlim(left_edge, right_edge)

    pull_ax.set_xlabel(self.axes[0].label)
    pull_ax.set_ylabel("Pull")

    return main_ax, pull_ax
