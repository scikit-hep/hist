import inspect
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union

import numpy as np

import hist

try:
    import matplotlib.axes
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    from mplhep.plot import Hist1DArtists, Hist2DArtists, hist2dplot, histplot
    from scipy import stats
except ModuleNotFoundError:
    print(
        "Hist requires mplhep to plot, either install hist[plot] or mplhep",
        file=sys.stderr,
    )
    raise


__all__ = (
    "histplot",
    "hist2dplot",
    "plot2d_full",
    "plot_ratio",
    "plot_pull",
    "plot_pie",
)


def _expand_shortcuts(key: str) -> str:
    if key == "ls":
        return "linestyle"
    return key


def _filter_dict(
    dict: Dict[str, Any], prefix: str, *, ignore: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Keyword argument conversion: convert the kwargs to several independent args, pulling
    them out of the dict given.
    """
    ignore_set: Set[str] = ignore or set()
    return {
        _expand_shortcuts(key[len(prefix) :]): dict.pop(key)
        for key in list(dict)
        if key.startswith(prefix) and key not in ignore_set
    }


def _expr_to_lambda(expr: str) -> Callable[..., Any]:
    """
    Converts a string expression like
        "a+b*np.exp(-c*x+math.pi)"
    into a callable function with 1 variable and N parameters,
        lambda x,a,b,c: "a+b*np.exp(-c*x+math.pi)"
    `x` is assumed to be the main variable, and preventing symbols
    like `foo.bar` or `foo(` from being considered as parameter.
    """
    from collections import OrderedDict
    from io import BytesIO
    from tokenize import NAME, tokenize

    varnames = []
    g = list(tokenize(BytesIO(expr.encode("utf-8")).readline))
    for ix, x in enumerate(g):
        toknum = x[0]
        tokval = x[1]
        if toknum != NAME:
            continue
        if ix > 0 and g[ix - 1][1] in ["."]:
            continue
        if ix < len(g) - 1 and g[ix + 1][1] in [".", "("]:
            continue
        varnames.append(tokval)
    varnames = list(OrderedDict.fromkeys([name for name in varnames if name != "x"]))
    lambdastr = f"lambda x,{','.join(varnames)}: {expr}"
    return eval(lambdastr)  # type: ignore


def _curve_fit_wrapper(
    func: Callable[..., Any],
    xdata: np.ndarray,
    ydata: np.ndarray,
    yerr: np.ndarray,
    likelihood: bool = False,
) -> Tuple[Tuple[float, ...], np.ndarray]:
    """
    Wrapper around `scipy.optimize.curve_fit`. Initial parameters (`p0`)
    can be set in the function definition with defaults for kwargs
    (e.g., `func = lambda x,a=1.,b=2.: x+a+b`, will feed `p0 = [1.,2.]` to `curve_fit`)
    """
    from scipy.optimize import curve_fit, minimize

    params = list(inspect.signature(func).parameters.values())
    p0 = [
        1 if arg.default is inspect.Parameter.empty else arg.default
        for arg in params[1:]
    ]

    mask = yerr != 0.0
    popt, pcov = curve_fit(
        func,
        xdata[mask],
        ydata[mask],
        sigma=yerr[mask],
        absolute_sigma=True,
        p0=p0,
    )
    if likelihood:
        from iminuit import Minuit
        from scipy.special import gammaln

        def fnll(v: Iterable[np.ndarray]) -> float:
            ypred = func(xdata, *v)
            if (ypred <= 0.0).any():
                return 1e6
            return (  # type: ignore
                ypred.sum() - (ydata * np.log(ypred)).sum() + gammaln(ydata + 1).sum()
            )

        # Seed likelihood fit with chi2 fit parameters
        res = minimize(fnll, popt, method="BFGS")
        popt = res.x

        # Better hessian from hesse, seeded with scipy popt
        m = Minuit(fnll, popt)
        m.errordef = 0.5
        m.hesse()
        pcov = np.array(m.covariance)
    return tuple(popt), pcov


def plot2d_full(
    self: hist.BaseHist,
    *,
    ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
    **kwargs: Any,
) -> "Tuple[Hist2DArtists, Hist1DArtists, Hist1DArtists]":
    """
    Plot2d_full method for BaseHist object.

    Pass a dict of axes to ``ax_dict``, otherwise, the current figure will be used.
    """
    # Type judgement
    if self.ndim != 2:
        raise TypeError("Only 2D-histogram has plot2d_full")

    if ax_dict is None:
        ax_dict = {}

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


# Taken from coffea for now
# https://github.com/CoffeaTeam/coffea/blob/12228d5564963234434d5c881efa761e088c298a/coffea/hist/plot.py#L57
# TODO: Check licensing
def poisson_interval(
    values: np.ndarray, variances: np.ndarray, coverage: "Optional[float]" = None
) -> np.ndarray:
    """Frequentist coverage interval for Poisson-distributed observations
    Parameters
    ----------
        values : np.ndarray
            Sum of weights vector
        variances : np.ndarray
            Sum weights squared vector
        coverage : float, optional
            Central coverage interval, defaults to 68%
    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``values**2/variances``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a proper treatment.
    When a bin is zero, the scale of the nearest nonzero bin is substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``values``.
    """
    if coverage is None:
        coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
    scale = np.empty_like(values)
    scale[values != 0] = variances[values != 0] / values[values != 0]
    if np.sum(values == 0) > 0:
        missing = np.where(values == 0)
        available = np.nonzero(values)
        if len(available[0]) == 0:
            raise RuntimeWarning(
                "All values are zero! Cannot compute meaningful error bars",
            )
            return np.vstack([values, values])
        nearest = np.sum(
            [np.square(np.subtract.outer(d, d0)) for d, d0 in zip(available, missing)]
        ).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = values / scale
    interval_min = scale * stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
    interval_max = scale * stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
    interval = np.array([interval_min, interval_max])
    interval[interval == np.nan] = 0.0  # chi2.ppf produces nan for counts=0
    return interval


def clopper_pearson_interval(
    num: np.ndarray, denom: np.ndarray, coverage: "Optional[float]" = None
) -> np.ndarray:
    """Compute Clopper-Pearson coverage interval for a binomial distribution
    Parameters
    ----------
        num : numpy.ndarray
            Numerator, or number of successes, vectorized
        denom : numpy.ndarray
            Denominator or number of trials, vectorized
        coverage : float, optional
            Central coverage interval, defaults to 68%
    c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if coverage is None:
        coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
    # Numerator is subset of denominator
    if np.any(num > denom):
        raise ValueError(
            "Found numerator larger than denominator while calculating binomial uncertainty"
        )
    interval_min = stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    interval_max = stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = np.array([interval_min, interval_max])
    interval[:, num == 0.0] = 0.0
    interval[1, num == denom] = 1.0
    return interval


# TODO: Why is return type "Any"?
def ratio_uncertainty(
    num: np.ndarray, denom: np.ndarray, uncert_type: str = "poisson"
) -> Any:
    ratio = num / denom
    if uncert_type == "poisson":
        ratio_uncert = np.abs(poisson_interval(ratio, num / np.square(denom)) - ratio)
        # TODO: Why does SciPy not work here?
        # coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
        # alpha = 1 - coverage
        # ratio_uncert = stats.poisson.interval(alpha, ratio)
    elif uncert_type == "poisson-ratio":
        # poisson ratio n/m is equivalent to binomial n/(n+m)
        ratio_uncert = np.abs(clopper_pearson_interval(num, num + denom) - ratio)
    return ratio_uncert


def plot_ratio(
    self: hist.BaseHist,
    other: Callable[[np.ndarray], np.ndarray],
    likelihood: bool = False,
    *,
    ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
    **kwargs: Any,
) -> "Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]":
    """
    Plot_ratio method for BaseHist object.
    """

    try:
        from iminuit import Minuit  # noqa: F401
        from scipy.optimize import curve_fit  # noqa: F401
    except ImportError:
        print(
            "Hist.plot_pull requires scipy and iminuit. Please install hist[plot] or manually install dependencies.",
            file=sys.stderr,
        )
        raise

    # Type judgement
    if not callable(other):
        msg = f"Callable parameter other is supported for {self.__class__.__name__} in plot ratio"
        raise TypeError(msg)

    if self.ndim != 1:
        raise TypeError("Only 1D-histogram supports ratio plot, try projecting to 1D")

    if ax_dict:
        try:
            main_ax = ax_dict["main_ax"]
            ratio_ax = ax_dict["ratio_ax"]
        except KeyError:
            raise ValueError("All axes should be all given or none at all")
    else:
        fig = plt.gcf()
        grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

        main_ax = fig.add_subplot(grid[0])
        ratio_ax = fig.add_subplot(grid[1], sharex=main_ax)

    # Computation and Fit
    xdata = self.axes[0].centers
    ydata = self.values()
    variances = self.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )
    yerr = np.sqrt(variances)

    # TODO: Make this only for callable
    # Compute fit values: using other as fit model
    popt, pcov = _curve_fit_wrapper(other, xdata, ydata, yerr, likelihood=likelihood)
    # TODO: Remove this if necessary?
    # perr = np.sqrt(np.diag(pcov))
    fit = other(self.axes[0].centers, *popt)

    if np.isfinite(pcov).all():
        nsamples = 100
        vopts = np.random.multivariate_normal(popt, pcov, nsamples)
        sampled_ydata = np.vstack([other(xdata, *vopt).T for vopt in vopts])
        fit_uncert = np.nanstd(sampled_ydata, axis=0)
    else:
        fit_uncert = np.zeros_like(yerr)

    # Compute ratios: containing no INF values
    with np.errstate(divide="ignore"):
        ratios = ydata / fit
        # TODO: Make this configurable
        ratio_uncert = ratio_uncertainty(num=ydata, denom=fit, uncert_type="poisson")

    # Keyword Argument Conversion: convert the kwargs to several independent args

    # error bar keyword arguments
    eb_kwargs = _filter_dict(kwargs, "eb_")
    eb_kwargs.setdefault("label", "Histogram Data")

    # fit plot keyword arguments
    fp_kwargs = _filter_dict(kwargs, "fp_")
    fp_kwargs.setdefault("label", "Fitted Value")

    # uncertainty band keyword arguments
    ub_kwargs = _filter_dict(kwargs, "ub_")
    ub_kwargs.setdefault("label", "Uncertainty")

    # # patch plot keyword arguments
    # pp_kwargs = _filter_dict(kwargs, "pp_", ignore={"pp_num"})
    # pp_num = kwargs.pop("pp_num", 5)

    # Judge whether some arguments are left
    if kwargs:
        raise ValueError(f"{set(kwargs)}' not needed")

    # Main: plot the ratios using Matplotlib errorbar and plot methods
    main_ax.errorbar(self.axes.centers[0], ydata, yerr, **eb_kwargs)

    (line,) = main_ax.plot(self.axes.centers[0], fit, **fp_kwargs)

    # Uncertainty band for fitted function
    ub_kwargs.setdefault("color", line.get_color())
    main_ax.fill_between(
        self.axes.centers[0],
        fit - fit_uncert,
        fit + fit_uncert,
        **ub_kwargs,
    )

    # TODO: Make configurable
    main_ax.legend(loc="best")
    # TODO: Make configurable
    main_ax.set_ylabel("Counts")

    # ratio: plot the ratios using Matplotlib bar method
    left_edge = self.axes.edges[0][0]
    right_edge = self.axes.edges[-1][-1]
    # # width = (right_edge - left_edge) / len(ratios)
    # ratio_ax.bar(self.axes.centers[0], ratios, width=width, **bar_kwargs)

    # Set 0 and inf to nan to hide during plotting
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    # TODO: Make configurable
    central_value = 1.0

    ratio_ax.axhline(central_value, color="black", linestyle="dashed", linewidth=1.0)
    ratio_ax.errorbar(
        self.axes.centers[0],
        ratios,
        yerr=ratio_uncert,
        color="black",
        marker="o",
        linestyle="none",
    )

    # TODO: Make configurable
    ratio_ylim = None
    if ratio_ylim is None:
        # set nans back to 0 to find extrema
        ratios[np.isnan(ratios)] = 0
        # plot centered around central value with a scaled view range
        # the value _with_ the uncertainty in view is important so base
        # view range on extrema of value +/- uncertainty
        non_zero_ratios_idx = np.where(ratios != 0)
        non_zero_ratios = ratios[non_zero_ratios_idx]
        extrema = np.array(
            [
                non_zero_ratios - ratio_uncert[0][non_zero_ratios_idx],
                non_zero_ratios + ratio_uncert[1][non_zero_ratios_idx],
            ]
        )
        max_delta = np.max(np.abs(extrema - central_value))
        ratio_extrema = np.abs(max_delta + central_value)

        _alpha = 2.0
        scaled_offset = max_delta + (max_delta / (_alpha * ratio_extrema))
        ratio_ylim = [central_value - scaled_offset, central_value + scaled_offset]

    ratio_ax.set_ylim(bottom=ratio_ylim[0], top=ratio_ylim[1])

    plt.xlim(left_edge, right_edge)

    ratio_ax.set_xlabel(self.axes[0].label)
    # TODO: Make configurable
    ratio_ax.set_ylabel("Ratio")

    return main_ax, ratio_ax


def plot_pull(
    self: hist.BaseHist,
    func: Union[Callable[[np.ndarray], np.ndarray], str],
    likelihood: bool = False,
    *,
    ax_dict: "Optional[Dict[str, matplotlib.axes.Axes]]" = None,
    **kwargs: Any,
) -> "Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]":
    """
    Plot_pull method for BaseHist object.
    """

    try:
        from iminuit import Minuit  # noqa: F401
        from scipy.optimize import curve_fit  # noqa: F401
    except ImportError:
        print(
            "Hist.plot_pull requires scipy and iminuit. Please install hist[plot] or manually install dependencies.",
            file=sys.stderr,
        )
        raise

    # Type judgement
    if not callable(func) and not type(func) in [str]:
        msg = f"Parameter func must be callable or a string for {self.__class__.__name__} in plot pull"
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
    xdata = self.axes[0].centers
    ydata = self.values()
    variances = self.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )
    yerr = np.sqrt(variances)

    if isinstance(func, str):
        if func in {"gauss", "gaus"}:
            # gaussian with reasonable initial guesses for parameters
            constant = float(ydata.max())
            mean = (ydata * xdata).sum() / ydata.sum()
            sigma = (ydata * (xdata - mean) ** 2.0).sum() / ydata.sum()

            def func(
                x: np.ndarray,
                constant: float = constant,
                mean: float = mean,
                sigma: float = sigma,
            ) -> np.ndarray:
                return constant * np.exp(-((x - mean) ** 2.0) / (2 * sigma ** 2))  # type: ignore

        else:
            func = _expr_to_lambda(func)

    assert not isinstance(func, str)

    parnames = list(inspect.signature(func).parameters)[1:]

    # Compute fit values: using func as fit model
    popt, pcov = _curve_fit_wrapper(func, xdata, ydata, yerr, likelihood=likelihood)
    perr = np.diag(pcov) ** 0.5
    yfit = func(self.axes[0].centers, *popt)

    if np.isfinite(pcov).all():
        nsamples = 100
        vopts = np.random.multivariate_normal(popt, pcov, nsamples)
        sampled_ydata = np.vstack([func(xdata, *vopt).T for vopt in vopts])
        yfiterr = np.nanstd(sampled_ydata, axis=0)
    else:
        yfiterr = np.zeros_like(yerr)

    # Compute pulls: containing no INF values
    with np.errstate(divide="ignore"):
        pulls = (ydata - yfit) / yerr

    pulls[np.isnan(pulls)] = 0
    pulls[np.isinf(pulls)] = 0

    # Keyword Argument Conversion: convert the kwargs to several independent args

    # error bar keyword arguments
    eb_kwargs = _filter_dict(kwargs, "eb_")
    eb_kwargs.setdefault("label", "Histogram Data")

    # fit plot keyword arguments
    label = "Fit"
    for name, value, error in zip(parnames, popt, perr):
        label += "\n  "
        label += rf"{name} = {value:.3g} $\pm$ {error:.3g}"
    fp_kwargs = _filter_dict(kwargs, "fp_")
    fp_kwargs.setdefault("label", label)

    # uncertainty band keyword arguments
    ub_kwargs = _filter_dict(kwargs, "ub_")
    ub_kwargs.setdefault("label", "Uncertainty")
    ub_kwargs.setdefault("alpha", 0.5)

    # bar plot keyword arguments
    bar_kwargs = _filter_dict(kwargs, "bar_", ignore={"bar_width"})

    # patch plot keyword arguments
    pp_kwargs = _filter_dict(kwargs, "pp_", ignore={"pp_num"})
    pp_num = kwargs.pop("pp_num", 5)

    # Judge whether some arguments are left
    if kwargs:
        raise ValueError(f"{set(kwargs)}' not needed")

    # Main: plot the pulls using Matplotlib errorbar and plot methods
    main_ax.errorbar(self.axes.centers[0], ydata, yerr, **eb_kwargs)

    (line,) = main_ax.plot(self.axes.centers[0], yfit, **fp_kwargs)

    # Uncertainty band
    ub_kwargs.setdefault("color", line.get_color())
    main_ax.fill_between(
        self.axes.centers[0],
        yfit - yfiterr,
        yfit + yfiterr,
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


def get_center(x: Union[str, int, Tuple[float, float]]) -> Union[str, float]:
    if isinstance(x, tuple):
        return (x[0] + x[1]) / 2
    else:
        return x


def plot_pie(
    self: hist.BaseHist,
    *,
    ax: "Optional[matplotlib.axes.Axes]" = None,
    **kwargs: Any,
) -> Any:

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(111)

    data = self.density()

    labels = [str(get_center(x)) for x in self.axes[0]]

    return ax.pie(data, labels=labels, **kwargs)
