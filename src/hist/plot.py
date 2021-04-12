import inspect
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

import hist

from .intervals import ratio_uncertainty
from .typing import Literal

try:
    import matplotlib.axes
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    from mplhep.plot import Hist1DArtists, Hist2DArtists, hist2dplot, histplot
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
    "plot_ratio_array",
    "plot_pull_array",
    "plot_pie",
)


class FitResultArtists(NamedTuple):
    line: matplotlib.lines.Line2D
    errorbar: matplotlib.container.ErrorbarContainer
    band: matplotlib.collections.PolyCollection


class RatioErrorbarArtists(NamedTuple):
    line: matplotlib.lines.Line2D
    errorbar: matplotlib.container.ErrorbarContainer


class RatioBarArtists(NamedTuple):
    line: matplotlib.lines.Line2D
    dots: matplotlib.collections.PathCollection
    bar: matplotlib.container.BarContainer


class PullArtists(NamedTuple):
    bar: matplotlib.container.BarContainer
    patch_artist: List[matplotlib.patches.Rectangle]


MainAxisArtists = Union[FitResultArtists, Hist1DArtists]

RatioArtists = Union[RatioErrorbarArtists, RatioBarArtists]
RatiolikeArtists = Union[RatioArtists, PullArtists]


def __dir__() -> Tuple[str, ...]:
    return __all__


def _expand_shortcuts(key: str) -> str:
    if key == "ls":
        return "linestyle"
    return key


def _filter_dict(
    __dict: Dict[str, Any], prefix: str, *, ignore: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Keyword argument conversion: convert the kwargs to several independent args, pulling
    them out of the dict given. Prioritize prefix_kw dict.
    """

    # If passed explicitly, use that
    if f"{prefix}kw" in __dict:
        res: Dict[str, Any] = __dict.pop(f"{prefix}kw")
        return {_expand_shortcuts(k): v for k, v in res.items()}

    ignore_set: Set[str] = ignore or set()
    return {
        _expand_shortcuts(key[len(prefix) :]): __dict.pop(key)
        for key in list(__dict)
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
    ax_dict: Optional[Dict[str, matplotlib.axes.Axes]] = None,
    **kwargs: Any,
) -> Tuple[Hist2DArtists, Hist1DArtists, Hist1DArtists]:
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


def _construct_gaussian_callable(
    __hist: hist.BaseHist,
) -> Callable[[np.ndarray], np.ndarray]:
    x_values = __hist.axes[0].centers
    hist_values = __hist.values()

    # gaussian with reasonable initial guesses for parameters
    constant = float(hist_values.max())
    mean = (hist_values * x_values).sum() / hist_values.sum()
    sigma = (hist_values * np.square(x_values - mean)).sum() / hist_values.sum()

    # gauss is a closure that will get evaluated in _fit_callable_to_hist
    def gauss(
        x: np.ndarray,
        constant: float = constant,
        mean: float = mean,
        sigma: float = sigma,
    ) -> np.ndarray:
        # Note: Force np.ndarray type as numpy ufuncs have type "Any"
        ret: np.ndarray = constant * np.exp(
            -np.square(x - mean) / (2 * np.square(sigma))
        )
        return ret

    return gauss


def _fit_callable_to_hist(
    model: Callable[[np.ndarray], np.ndarray],
    histogram: hist.BaseHist,
    likelihood: bool = False,
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]":
    """
    Fit a model, a callable function, to the histogram values.
    """
    variances = histogram.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )
    hist_uncert = np.sqrt(variances)

    # Infer best fit model parameters and covariance matrix
    xdata = histogram.axes[0].centers
    popt, pcov = _curve_fit_wrapper(
        model, xdata, histogram.values(), hist_uncert, likelihood=likelihood
    )
    model_values = model(xdata, *popt)

    if np.isfinite(pcov).all():
        n_samples = 100
        vopts = np.random.multivariate_normal(popt, pcov, n_samples)
        sampled_ydata = np.vstack([model(xdata, *vopt).T for vopt in vopts])
        model_uncert = np.nanstd(sampled_ydata, axis=0)
    else:
        model_uncert = np.zeros_like(hist_uncert)

    return model_values, model_uncert, hist_uncert, (popt, pcov)


def _plot_fit_result(
    __hist: hist.BaseHist,
    model_values: np.ndarray,
    model_uncert: np.ndarray,
    ax: matplotlib.axes.Axes,
    eb_kwargs: Dict[str, Any],
    fp_kwargs: Dict[str, Any],
    ub_kwargs: Dict[str, Any],
) -> FitResultArtists:
    """
    Plot fit of model to histogram data
    """
    x_values = __hist.axes[0].centers
    variances = __hist.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )
    hist_uncert = np.sqrt(variances)

    errorbars = ax.errorbar(x_values, __hist.values(), hist_uncert, **eb_kwargs)

    # Ensure zorder draws data points above model
    line_zorder = errorbars[0].get_zorder() - 1
    (line,) = ax.plot(x_values, model_values, **fp_kwargs, zorder=line_zorder)

    # Uncertainty band for fitted function
    ub_kwargs.setdefault("color", line.get_color())
    if ub_kwargs["color"] == line.get_color():
        ub_kwargs.setdefault("alpha", 0.3)
    uncertainty_band = ax.fill_between(
        x_values,
        model_values - model_uncert,
        model_values + model_uncert,
        **ub_kwargs,
    )

    return FitResultArtists(line, errorbars, uncertainty_band)


def plot_ratio_array(
    __hist: hist.BaseHist,
    ratio: np.ndarray,
    ratio_uncert: np.ndarray,
    ax: matplotlib.axes.Axes,
    **kwargs: Any,
) -> RatioArtists:
    """
    Plot a ratio plot on the given axes
    """
    x_values = __hist.axes[0].centers
    left_edge = __hist.axes.edges[0][0]
    right_edge = __hist.axes.edges[-1][-1]

    # Set 0 and inf to nan to hide during plotting
    ratio[ratio == 0] = np.nan
    ratio[np.isinf(ratio)] = np.nan

    central_value = kwargs.pop("central_value", 1.0)
    central_value_artist = ax.axhline(
        central_value, color="black", linestyle="dashed", linewidth=1.0
    )

    # Type now due to control flow
    axis_artists: Union[RatioErrorbarArtists, RatioBarArtists]

    uncert_draw_type = kwargs.pop("uncert_draw_type", "line")
    if uncert_draw_type == "line":
        errorbar_artists = ax.errorbar(
            x_values,
            ratio,
            yerr=ratio_uncert,
            color="black",
            marker="o",
            linestyle="none",
        )
        axis_artists = RatioErrorbarArtists(central_value_artist, errorbar_artists)
    elif uncert_draw_type == "bar":
        bar_width = (right_edge - left_edge) / len(ratio)

        bar_top = ratio + ratio_uncert[1]
        bar_bottom = ratio - ratio_uncert[0]
        # bottom can't be nan
        bar_bottom[np.isnan(bar_bottom)] = 0
        bar_height = bar_top - bar_bottom

        _ratio_points = ax.scatter(x_values, ratio, color="black")

        # Ensure zorder draws data points above uncertainty bars
        bar_zorder = _ratio_points.get_zorder() - 1
        bar_artists = ax.bar(
            x_values,
            height=bar_height,
            width=bar_width,
            bottom=bar_bottom,
            fill=False,
            linewidth=0,
            edgecolor="gray",
            hatch=3 * "/",
            zorder=bar_zorder,
        )
        axis_artists = RatioBarArtists(central_value_artist, _ratio_points, bar_artists)

    ratio_ylim = kwargs.pop("ylim", None)
    if ratio_ylim is None:
        # plot centered around central value with a scaled view range
        # the value _with_ the uncertainty in view is important so base
        # view range on extrema of value +/- uncertainty
        valid_ratios_idx = np.where(np.isnan(ratio) == False)  # noqa: E712
        valid_ratios = ratio[valid_ratios_idx]
        extrema = np.array(
            [
                valid_ratios - ratio_uncert[0][valid_ratios_idx],
                valid_ratios + ratio_uncert[1][valid_ratios_idx],
            ]
        )
        max_delta = np.max(np.abs(extrema - central_value))
        ratio_extrema = np.abs(max_delta + central_value)

        _alpha = 2.0
        scaled_offset = max_delta + (max_delta / (_alpha * ratio_extrema))
        ratio_ylim = [central_value - scaled_offset, central_value + scaled_offset]

    ax.set_xlim(left_edge, right_edge)
    ax.set_ylim(bottom=ratio_ylim[0], top=ratio_ylim[1])

    ax.set_xlabel(__hist.axes[0].label)
    ax.set_ylabel(kwargs.pop("ylabel", "Ratio"))

    return axis_artists


def plot_pull_array(
    __hist: hist.BaseHist,
    pulls: np.ndarray,
    ax: matplotlib.axes.Axes,
    bar_kwargs: Dict[str, Any],
    pp_kwargs: Dict[str, Any],
) -> PullArtists:
    """
    Plot a pull plot on the given axes
    """
    x_values = __hist.axes[0].centers
    left_edge = __hist.axes.edges[0][0]
    right_edge = __hist.axes.edges[-1][-1]

    # Pull: plot the pulls using Matplotlib bar method
    width = (right_edge - left_edge) / len(pulls)
    bar_artists = ax.bar(x_values, pulls, width=width, **bar_kwargs)

    pp_num = pp_kwargs.pop("num", 5)
    patch_height = max(np.abs(pulls)) / pp_num
    patch_width = width * len(pulls)
    patch_artists = []
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
        ax.add_patch(upRect)
        downRect_startpoint = (left_edge, -(i + 1) * patch_height)
        downRect = patches.Rectangle(
            downRect_startpoint, patch_width, patch_height, **pp_kwargs
        )
        ax.add_patch(downRect)
        patch_artists.append((downRect, upRect))

    ax.set_xlim(left_edge, right_edge)

    ax.set_xlabel(__hist.axes[0].label)
    ax.set_ylabel("Pull")

    return PullArtists(bar_artists, patch_artists)


def _plot_ratiolike(
    self: hist.BaseHist,
    other: Union[hist.BaseHist, Callable[[np.ndarray], np.ndarray], str],
    likelihood: bool = False,
    *,
    ax_dict: Optional[Dict[str, matplotlib.axes.Axes]] = None,
    view: Literal["ratio", "pull"],
    fit_fmt: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[MainAxisArtists, RatiolikeArtists]:
    r"""
    Plot ratio-like plots (ratio plots and pull plots) for BaseHist

    ``fit_fmt`` can be a string such as ``r"{name} = {value:.3g} $\pm$ {error:.3g}"``
    """

    try:
        from iminuit import Minuit  # noqa: F401
        from scipy.optimize import curve_fit  # noqa: F401
    except ModuleNotFoundError:
        print(
            f"Hist.plot_{view} requires scipy and iminuit. Please install hist[plot] or manually install dependencies.",
            file=sys.stderr,
        )
        raise

    if self.ndim != 1:
        raise TypeError(
            f"Only 1D-histogram supports ratio plot, try projecting {self.__class__.__name__} to 1D"
        )
    if isinstance(other, hist.hist.Hist) and other.ndim != 1:
        raise TypeError(
            f"Only 1D-histogram supports ratio plot, try projecting other={other.__class__.__name__} to 1D"
        )

    if ax_dict:
        try:
            main_ax = ax_dict["main_ax"]
            subplot_ax = ax_dict[f"{view}_ax"]
        except KeyError:
            raise ValueError("All axes should be all given or none at all")
    else:
        fig = plt.gcf()
        grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

        main_ax = fig.add_subplot(grid[0])
        subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
        plt.setp(main_ax.get_xticklabels(), visible=False)

    # Keyword Argument Conversion: convert the kwargs to several independent args
    # error bar keyword arguments
    eb_kwargs = _filter_dict(kwargs, "eb_")
    eb_kwargs.setdefault("label", "Histogram Data")
    # Use "fmt" over "marker" to avoid UserWarning on keyword precedence
    eb_kwargs.setdefault("fmt", "o")
    eb_kwargs.setdefault("linestyle", "none")

    # fit plot keyword arguments
    fp_kwargs = _filter_dict(kwargs, "fp_")
    fp_kwargs.setdefault("label", "Counts")

    # bar plot keyword arguments
    bar_kwargs = _filter_dict(kwargs, "bar_", ignore={"bar_width"})

    # uncertainty band keyword arguments
    ub_kwargs = _filter_dict(kwargs, "ub_")
    ub_kwargs.setdefault("label", "Uncertainty")

    # ratio plot keyword arguments
    rp_kwargs = _filter_dict(kwargs, "rp_")
    rp_kwargs.setdefault("uncertainty_type", "poisson")
    rp_kwargs.setdefault("legend_loc", "best")
    rp_kwargs.setdefault("num_label", None)
    rp_kwargs.setdefault("denom_label", None)

    # patch plot keyword arguments
    pp_kwargs = _filter_dict(kwargs, "pp_")

    # Judge whether some arguments are left
    if kwargs:
        raise ValueError(f"{set(kwargs)}' not needed")

    main_ax.set_ylabel(fp_kwargs["label"])

    # Computation and Fit
    hist_values = self.values()

    main_ax_artists: MainAxisArtists  # Type now due to control flow
    if callable(other) or isinstance(other, str):
        if isinstance(other, str):
            if other in {"gauss", "gaus", "normal"}:
                other = _construct_gaussian_callable(self)
            else:
                other = _expr_to_lambda(other)

        (
            compare_values,
            model_uncert,
            hist_values_uncert,
            bestfit_result,
        ) = _fit_callable_to_hist(other, self, likelihood)

        if fit_fmt is not None:
            parnames = list(inspect.signature(other).parameters)[1:]
            popt, pcov = bestfit_result
            perr = np.sqrt(np.diag(pcov))

            fp_label = "Fit"
            for name, value, error in zip(parnames, popt, perr):
                fp_label += "\n  "
                fp_label += fit_fmt.format(name=name, value=value, error=error)
            fp_kwargs["label"] = fp_label
        else:
            fp_kwargs["label"] = "Fitted value"

        main_ax_artists = _plot_fit_result(
            self,
            model_values=compare_values,
            model_uncert=model_uncert,
            ax=main_ax,
            eb_kwargs=eb_kwargs,
            fp_kwargs=fp_kwargs,
            ub_kwargs=ub_kwargs,
        )
    else:
        compare_values = other.values()

        self_artists = histplot(self, ax=main_ax, label=rp_kwargs["num_label"])
        other_artists = histplot(other, ax=main_ax, label=rp_kwargs["denom_label"])

        main_ax_artists = self_artists, other_artists

    subplot_ax_artists: RatiolikeArtists  # Type now due to control flow
    # Compute ratios: containing no INF values
    with np.errstate(divide="ignore", invalid="ignore"):
        if view == "ratio":
            ratios = hist_values / compare_values
            ratio_uncert = ratio_uncertainty(
                num=hist_values,
                denom=compare_values,
                uncertainty_type=rp_kwargs["uncertainty_type"],
            )
            # ratio: plot the ratios using Matplotlib errorbar or bar
            subplot_ax_artists = plot_ratio_array(
                self, ratios, ratio_uncert, ax=subplot_ax, **rp_kwargs
            )

        elif view == "pull":
            pulls = (hist_values - compare_values) / hist_values_uncert

            pulls[np.isnan(pulls) | np.isinf(pulls)] = 0

            # Pass dicts instead of unpacking to avoid conflicts
            subplot_ax_artists = plot_pull_array(
                self, pulls, ax=subplot_ax, bar_kwargs=bar_kwargs, pp_kwargs=pp_kwargs
            )

    if main_ax.get_legend_handles_labels()[0]:  # Don't plot an empty legend
        main_ax.legend(loc=rp_kwargs["legend_loc"])

    return main_ax_artists, subplot_ax_artists


def get_center(x: Union[str, int, Tuple[float, float]]) -> Union[str, float]:
    if isinstance(x, tuple):
        return (x[0] + x[1]) / 2
    else:
        return x


def plot_pie(
    self: hist.BaseHist,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> Any:

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(111)

    data = self.density()

    labels = [str(get_center(x)) for x in self.axes[0]]

    return ax.pie(data, labels=labels, **kwargs)
