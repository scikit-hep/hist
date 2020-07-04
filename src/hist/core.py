import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
from scipy.optimize import curve_fit
from uncertainties import correlated_values, unumpy
from boost_histogram import Histogram, loc
from typing import Callable, Optional, Tuple, Union

# typing alias
Plot1D_RetType = Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.SubplotBase]
Plot2D_RetType = Tuple[
    matplotlib.figure.Figure, matplotlib.axes._subplots.SubplotBase,
]
Plot2DFull_RetType = Tuple[
    matplotlib.figure.Figure,
    matplotlib.axes._subplots.SubplotBase,
    matplotlib.axes._subplots.SubplotBase,
    matplotlib.axes._subplots.SubplotBase,
]
PlotPull_RetType = Tuple[
    matplotlib.figure.Figure,
    matplotlib.axes._subplots.SubplotBase,
    matplotlib.axes._subplots.SubplotBase,
]


class BaseHist(Histogram):
    def __init__(self, *args, **kwargs):
        """
            Initialize BaseHist object. Axis params can contain the names.
        """

        super().__init__(*args, **kwargs)
        self.names: dict = dict()
        for ax in self.axes:
            if ax.name in self.names:
                raise Exception(
                    f"{self.__class__.__name__} instance cannot contain axes with duplicated names."
                )
            else:
                self.names[ax.name] = True

    def project(self, *args: Union[int, str]):
        """
        Projection of axis idx.
        """
        if len(args) == 0 or all(isinstance(x, int) for x in args):
            return super().project(*args)

        elif all(isinstance(x, str) for x in args):
            indices: tuple = tuple()
            for name in args:
                for index, axis in enumerate(self.axes):
                    if name == axis.name:
                        indices += (index,)
                        break
                else:
                    raise ValueError("The axis names could not be found")

            return super().project(*indices)

        else:
            raise TypeError(
                f"Only projections by indices and names are supported for {self.__class__.__name__}"
            )

    def plot(self, *args, **kwargs,) -> Union[Plot1D_RetType, Plot2D_RetType]:
        """
        Plot method for BaseHist object.
        """
        if self.rank == 1:
            return self.plot1d(*args, **kwargs), None
        elif self.rank == 2:
            return self.plot2d(*args, **kwargs)
        else:
            raise NotImplemented("Please project to 1D or 2D before calling plot")

    def plot1d(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        **kwargs,
    ) -> Plot1D_RetType:
        """
        Plot1d method for BaseHist object.
        """
        # Type judgement
        if self.rank != 1:
            raise TypeError("Only 1D-histogram has plot1d")

        """
        Default Figure: construct the figure and axes
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(4, 4, hspace=0, wspace=0)

        if ax is None:
            ax = fig.add_subplot(grid[:, :])

        """
        Plot: plot the 1d-histogram
        """
        ax.step(self.axes.edges[0][:-1], self.project(0).view(), **kwargs)
        if self.axes[0].name:
            ax.set_xlabel(self.axes[0].name)
        else:
            ax.set_xlabel(self.axes[0].title)

        ax.set_ylabel("Counts")

        fig.add_axes(ax)

        return fig, ax

    def plot2d(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        **kwargs,
    ) -> Plot2D_RetType:
        """
        Plot2d method for BaseHist object.
        """
        # Type judgement
        if self.rank != 2:
            raise TypeError("Only 2D-histogram has plot2d")

        """
        Default Figure: construct the figure and axes
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(4, 4, hspace=0, wspace=0)

        if ax is None:
            ax = fig.add_subplot(grid[:, :])

        """
        Plot: plot the 2d-histogram
        """
        X, Y = self.axes.edges
        ax.pcolormesh(X.T, Y.T, self.view().T, **kwargs)

        if self.axes[0].name:
            ax.set_xlabel(self.axes[0].name)
        else:
            ax.set_xlabel(self.axes[0].title)

        if self.axes[1].name:
            ax.set_ylabel(self.axes[1].name)
        else:
            ax.set_ylabel(self.axes[1].title)

        fig.add_axes(ax)

        return fig, ax

    def plot2d_full(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        main_ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        top_ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        side_ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        **kwargs,
    ) -> Plot2DFull_RetType:
        """
        Plot2d_full method for BaseHist object.
        """
        # Type judgement
        if self.rank != 2:
            raise TypeError("Only 2D-histogram has plot2d_full")

        """
        Default Figure: construct the figure and axes
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(4, 4, hspace=0, wspace=0)

        if main_ax is None:
            main_ax = fig.add_subplot(grid[1:4, 0:3])

        if top_ax is None:
            top_ax = fig.add_subplot(grid[0:1, 0:3], sharex=main_ax)

        if side_ax is None:
            side_ax = fig.add_subplot(grid[1:4, 3:4], sharey=main_ax)

        """
        Keyword Argument Conversion: convert the kwargs to several independent args
        """
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

        # judge whether some arguements left
        if len(kwargs):
            raise ValueError(f"'{list(kwargs.keys())[0]}' not needed")

        """
        Plot: plot the 2d-histogram
        """
        # main plot
        X, Y = self.axes.edges
        main_ax.pcolormesh(X.T, Y.T, self.view().T, **main_kwargs)

        if self.axes[0].name:
            main_ax.set_xlabel(self.axes[0].name)
        else:
            main_ax.set_xlabel(self.axes[0].title)

        if self.axes[1].name:
            main_ax.set_ylabel(self.axes[1].name)
        else:
            main_ax.set_ylabel(self.axes[1].title)

        # top plot
        top_ax.step(self.axes.edges[1][0][:-1], self.project(0).view(), **top_kwargs)
        top_ax.spines["top"].set_visible(False)
        top_ax.spines["right"].set_visible(False)
        top_ax.xaxis.set_visible(False)

        top_ax.set_ylabel("Counts")

        # side plot
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)
        side_ax.step(
            self.axes.edges[1][0][:-1],
            -self.project(1).view(),
            transform=rot + base,
            **side_kwargs,
        )
        side_ax.spines["top"].set_visible(False)
        side_ax.spines["right"].set_visible(False)
        side_ax.yaxis.set_visible(False)

        side_ax.set_xlabel("Counts")

        fig.add_axes(main_ax)
        fig.add_axes(top_ax)
        fig.add_axes(side_ax)

        return fig, main_ax, top_ax, side_ax

    def plot_pull(
        self,
        func: Callable,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        pull_ax: Optional[matplotlib.axes._subplots.SubplotBase] = None,
        **kwargs,
    ) -> PlotPull_RetType:
        """
        Plot_pull method for BaseHist object.
        """

        # Type judgement
        if callable(func) == False:
            raise TypeError("Callable parameter func is supported in pull plot")
        if self.rank != 1:
            raise TypeError("Only 1D-histogram has pull plot")

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
        with np.errstate(divide="ignore"):
            pulls = (self.view() - y_nv) / yerr

        pulls[np.isnan(pulls)] = 0
        pulls[np.isinf(pulls)] = 0

        """
        Default Figure: construct the figure and axes
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

        if ax is None:
            ax = fig.add_subplot(grid[0])

        if pull_ax is None:
            pull_ax = fig.add_subplot(grid[1], sharex=ax)

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
                    raise ValueError(f"'pp_label' not needed")
                pp_kwargs[kw[3:]] = kwargs[kw]

        if "pp_num" in kwargs:
            kwargs.pop("pp_num")
        for k in pp_kwargs:
            kwargs.pop("pp_" + k)

        # judge whether some arguements left
        if len(kwargs):
            raise ValueError(f"'{list(kwargs.keys())[0]}' not needed")

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

        if self.axes[0].name:
            pull_ax.set_xlabel(self.axes[0].name)
        else:
            pull_ax.set_xlabel(self.axes[0].title)
        pull_ax.set_ylabel("Pull")

        fig.add_axes(ax)
        fig.add_axes(pull_ax)

        return fig, ax, pull_ax

    def __getitem__(self, index):
        """
            Get histogram item.
        """

        if isinstance(index, dict):
            return super().__getitem__(index)

        if not hasattr(index, "__iter__"):
            index = (index,)

        t: tuple = ()
        for idx in index:
            if isinstance(idx, complex):
                t += (loc(idx.imag, int(idx.real)),)
            elif isinstance(idx, str):
                t += (loc(idx),)
            else:
                t += (idx,)

        return super().__getitem__(t)

    def __setitem__(self, index, value):
        """
            Set histogram item.
        """

        if isinstance(index, dict):
            return super().__setitem__(index, value)

        if not hasattr(index, "__iter__"):
            index = (index,)

        t: tuple = ()
        for idx in index:
            if isinstance(idx, complex):
                t += (loc(idx.imag, int(idx.real)),)
            elif isinstance(idx, str):
                t += (loc(idx),)
            else:
                t += (idx,)

        return super().__setitem__(t, value)
