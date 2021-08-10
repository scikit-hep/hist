from __future__ import annotations

from typing import Any

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


def __dir__() -> tuple[str, ...]:
    return __all__


def poisson_interval(
    values: np.typing.NDArray[Any],
    variances: np.typing.NDArray[Any] | None = None,
    coverage: float | None = None,
) -> np.typing.NDArray[Any]:
    r"""
    The Frequentist coverage interval for Poisson-distributed observations.

    What is calculated is the "Garwood" interval, c.f.
    `V. Patil, H. Kulkarni (Revstat, 2012) <https://www.ine.pt/revstat/pdf/rs120203.pdf>`_
    or http://ms.mcmaster.ca/peter/s743/poissonalpha.html.
    If ``variances`` is supplied, the data is assumed to be weighted, and the
    unweighted count is approximated by ``values**2/variances``, which effectively
    scales the unweighted Poisson interval by the average weight.
    This may not be the optimal solution: see
    `10.1016/j.nima.2014.02.021 <https://doi.org/10.1016/j.nima.2014.02.021>`_
    (`arXiv:1309.1287 <https://arxiv.org/abs/1309.1287>`_) for a proper treatment.

    In cases where the value is zero, an upper limit is well-defined only in the case of
    unweighted data, so if ``variances`` is supplied, the upper limit for a zero value
    will be set to ``NaN``.

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
    if variances is None:
        interval_min = stats.chi2.ppf((1 - coverage) / 2, 2 * values) / 2.0
        interval_min[values == 0.0] = 0.0  # chi2.ppf produces NaN for values=0
        interval_max = stats.chi2.ppf((1 + coverage) / 2, 2 * (values + 1)) / 2.0
    else:
        scale = np.ones_like(values)
        mask = np.isfinite(values) & (values != 0)
        np.divide(variances, values, out=scale, where=mask)
        counts: np.typing.NDArray[Any] = values / scale
        interval_min = scale * stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
        interval_min[values == 0.0] = 0.0  # chi2.ppf produces NaN for values=0
        interval_max = (
            scale * stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
        )
        interval_max[values == 0.0] = np.nan
    interval = np.stack((interval_min, interval_max))
    return interval


def clopper_pearson_interval(
    num: np.typing.NDArray[Any],
    denom: np.typing.NDArray[Any],
    coverage: float | None = None,
) -> np.typing.NDArray[Any]:
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
    interval[0, num == 0.0] = 0.0
    interval[1, num == denom] = 1.0
    return interval


def ratio_uncertainty(
    num: np.typing.NDArray[Any],
    denom: np.typing.NDArray[Any],
    uncertainty_type: Literal["poisson", "poisson-ratio", "efficiency"] = "poisson",
) -> Any:
    r"""
    Calculate the uncertainties for the values of the ratio ``num/denom`` using
    the specified coverage interval approach.

    Args:
        num: Numerator or number of successes.
        denom: Denominator or number of trials.
        uncertainty_type: Coverage interval type to use in the calculation of
         the uncertainties.

         * ``"poisson"`` (default) implements the Garwood confidence interval for
           a Poisson-distributed numerator scaled by the denominator.
           See :func:`hist.intervals.poisson_interval` for further details.
         * ``"poisson-ratio"`` implements a confidence interval for the ratio ``num / denom``
           assuming it is an estimator of the ratio of the expected rates from
           two independent Poisson distributions.
           It over-covers to a similar degree as the Clopper-Pearson interval
           does for the Binomial efficiency parameter estimate.
         * ``"efficiency"`` implements the Clopper-Pearson confidence interval
           for the ratio ``num / denom`` assuming it is an estimator of a Binomial
           efficiency parameter.
           This is only valid if the entries contributing to ``num`` are a strict
           subset of those contributing to ``denom``.

    Returns:
        The uncertainties for the ratio.
    """
    # Note: As return is a numpy ufuncs the type is "Any"
    with np.errstate(divide="ignore", invalid="ignore"):
        # Nota bene: x/0 = inf, 0/0 = nan
        ratio = num / denom
    if uncertainty_type == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_variance = num * np.power(denom, -2.0)
        ratio_uncert = np.abs(poisson_interval(ratio, ratio_variance) - ratio)
    elif uncertainty_type == "poisson-ratio":
        # Details: see https://github.com/scikit-hep/hist/issues/279
        p_lim = clopper_pearson_interval(num, num + denom)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_lim: np.typing.NDArray[Any] = p_lim / (1 - p_lim)
            ratio_uncert = np.abs(r_lim - ratio)
    elif uncertainty_type == "efficiency":
        ratio_uncert = np.abs(clopper_pearson_interval(num, denom) - ratio)
    else:
        raise TypeError(
            f"'{uncertainty_type}' is an invalid option for uncertainty_type."
        )
    return ratio_uncert
