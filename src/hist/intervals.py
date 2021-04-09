from typing import Any, Optional, Tuple

import numpy as np

from .typing import Literal

try:
    from scipy import stats
except ModuleNotFoundError:
    from sys import stderr

    print(
        "hist.intervals requires scipy. Please install hist[plot] or manually install scipy.",
        file=stderr,
    )
    raise

__all__ = ("poisson_interval", "clopper_pearson_interval", "ratio_uncertainty")


def __dir__() -> Tuple[str, ...]:
    return __all__


def poisson_interval(
    values: np.ndarray, variances: np.ndarray, coverage: "Optional[float]" = None
) -> np.ndarray:
    r"""
    The Frequentist coverage interval for Poisson-distributed observations.

    What is calculated is the "Garwood" interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html.
    For weighted data, this approximates the observed count by
    ``values**2/variances``, which effectively scales the unweighted
    Poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/abs/1309.1287
    for a proper treatment.

    When a bin is zero, the scale of the nearest nonzero bin is substituted to
    scale the nominal upper bound.

    Args:
        values: Sum of weights.
        variances: Sum of weights squared.
        coverage: Central coverage interval.
          Default is one standard deviation, which is roughly ``0.68``.

    Returns:
        The Poisson central coverage interval.
    """
    # Parts originally contributed to coffea
    # https://github.com/CoffeaTeam/coffea/blob/8c58807e199a7694bf15e3803dbaf706d34bbfa0/LICENSE
    if coverage is None:
        coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
    scale = np.empty_like(values)
    scale[values != 0] = variances[values != 0] / values[values != 0]
    if np.sum(values == 0) > 0:
        missing = np.where(values == 0)
        available = np.nonzero(values)
        if len(available[0]) == 0:
            raise RuntimeError(
                "All values are zero! Cannot compute meaningful uncertainties.",
            )
        nearest = np.sum(
            [np.square(np.subtract.outer(d, d0)) for d, d0 in zip(available, missing)]
        ).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = values / scale
    interval_min = scale * stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
    interval_max = scale * stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
    interval = np.stack((interval_min, interval_max))
    interval[interval == np.nan] = 0.0  # chi2.ppf produces nan for counts=0
    return interval


def clopper_pearson_interval(
    num: np.ndarray, denom: np.ndarray, coverage: "Optional[float]" = None
) -> np.ndarray:
    r"""
    Compute the Clopper-Pearson coverage interval for a binomial distribution.
    c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Args:
        num: Numerator or number of successes.
        denom: Denominator or number of trials.
        coverage: Central coverage interval.
          Default is one standard deviation, which is roughly ``0.68``.

    Returns:
        The Clopper-Pearson central coverage interval.
    """
    # Parts originally contributed to coffea
    # https://github.com/CoffeaTeam/coffea/blob/8c58807e199a7694bf15e3803dbaf706d34bbfa0/LICENSE
    if coverage is None:
        coverage = stats.norm.cdf(1) - stats.norm.cdf(-1)
    # Numerator is subset of denominator
    if np.any(num > denom):
        raise ValueError(
            "Found numerator larger than denominator while calculating binomial uncertainty"
        )
    interval_min = stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    interval_max = stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = np.stack((interval_min, interval_max))
    interval[:, num == 0.0] = 0.0
    interval[1, num == denom] = 1.0
    return interval


def ratio_uncertainty(
    num: np.ndarray,
    denom: np.ndarray,
    uncertainty_type: Literal["poisson", "poisson-ratio"] = "poisson",
) -> Any:
    r"""
    Calculate the uncertainties for the values of the ratio ``num/denom`` using
    the specified coverage interval approach.

    Args:
        num: Numerator or number of successes.
        denom: Denominator or number of trials.
        uncertainty_type: Coverage interval type to use in the calculation of
         the uncertainties.
         ``"poisson"`` (default) implements the Poisson interval for the
         numerator scaled by the denominator.
         ``"poisson-ratio"`` implements the Clopper-Pearson interval for Poisson
         distributed ``num`` and ``denom``.

    Returns:
        The uncertainties for the ratio.
    """
    # Note: As return is a numpy ufuncs the type is "Any"
    with np.errstate(divide="ignore"):
        ratio = num / denom
    if uncertainty_type == "poisson":
        ratio_uncert = np.abs(poisson_interval(ratio, num / np.square(denom)) - ratio)
    elif uncertainty_type == "poisson-ratio":
        # poisson ratio n/m is equivalent to binomial n/(n+m)
        ratio_uncert = np.abs(clopper_pearson_interval(num, num + denom) - ratio)
    else:
        raise TypeError(
            f"'{uncertainty_type}' is an invalid option for uncertainty_type."
        )
    return ratio_uncert
