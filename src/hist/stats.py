from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats as spstats

import hist


def chisquare_1samp(
    self: hist.BaseHist,
    distribution: str | Callable[..., Any],
    args: Any = (),
    kwds: Any = None,
) -> Any:
    kwds = {} if kwds is None else kwds

    if self.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports chisquare_1samp, try projecting {self.__class__.__name__} to 1D"
        )
    if isinstance(distribution, str):
        if not hasattr(spstats, distribution):
            raise ValueError(
                f"Unknown distribution {distribution}, try one of the defined scipy distributions"
            )
        cdf = getattr(spstats, distribution).cdf
    elif callable(distribution):
        cdf = distribution
    else:
        raise TypeError(
            f"Unknown distribution type {distribution}, try one of the defined scipy distributions or a callable CDF"
        )

    variances = self.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )

    observed = self.values()
    totalentries = self.sum()
    expected = np.diff(cdf(self.axes[0].edges, *args, **kwds)) * totalentries
    where = variances != 0
    squares = (expected[where] - observed[where]) ** 2 / variances[where]
    ndof = len(observed) - 1
    chisq = np.sum(squares)
    pvalue = spstats.chi2.sf(chisq, ndof)

    return chisq, ndof, pvalue


def chisquare_2samp(self: hist.BaseHist, other: hist.BaseHist) -> Any:
    if self.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports chisquare_2samp, try projecting {self.__class__.__name__} to 1D"
        )
    if isinstance(other, hist.hist.Hist) and other.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports chisquare_2samp, try projecting other={other.__class__.__name__} to 1D"
        )
    if not isinstance(other, hist.hist.Hist):
        raise TypeError(
            f"Unknown type {other.__class__.__name__}, other must be a hist.Hist object"
        )

    variances1 = self.variances()
    variances2 = other.variances()
    if variances1 is None or variances2 is None:
        raise RuntimeError(
            "Cannot compute from variance-less histograms, try a Weight storage"
        )

    counts1 = self.values()
    counts2 = other.values()
    squares = (
        counts1 * np.sqrt(counts2.sum() / counts1.sum())
        - counts2 * np.sqrt(counts1.sum() / counts2.sum())
    ) ** 2
    variances = variances1 + variances2
    where = variances != 0
    chisq = np.sum(squares[where] / variances[where])
    k = where.sum()
    ndof = k - 1 if self.sum() == other.sum() else k
    pvalue = spstats.chi2.sf(chisq, ndof)

    return chisq, ndof, pvalue


def ks_1samp(
    self: hist.BaseHist,
    distribution: str | Callable[..., Any],
    args: Any = (),
    kwds: Any = None,
) -> Any:
    kwds = {} if kwds is None else kwds

    if self.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports ks_1samp, try projecting {self.__class__.__name__} to 1D"
        )
    if isinstance(distribution, str):
        if not hasattr(spstats, distribution):
            raise ValueError(
                f"Unknown distribution {distribution}, try one of the defined scipy distributions"
            )
        cdf = getattr(spstats, distribution).cdf
    elif callable(distribution):
        cdf = distribution
    else:
        raise TypeError(
            f"Unknown distribution type {distribution}, try one of the defined scipy distributions or a callable CDF"
        )

    variances = self.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )

    return cdf(self.axes[0].edges, *args, **kwds)  # placeholder to pass pre-commit


def ks_2samp(self: hist.BaseHist, other: hist.BaseHist) -> Any:
    if self.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports ks_2samp, try projecting {self.__class__.__name__} to 1D"
        )
    if isinstance(other, hist.hist.Hist) and other.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports ks_2samp, try projecting other={other.__class__.__name__} to 1D"
        )
    if not isinstance(other, hist.hist.Hist):
        raise TypeError(
            f"Unknown type {other.__class__.__name__}, other must be a hist.Hist object"
        )

    variances1 = self.variances()
    variances2 = other.variances()
    if variances1 is None or variances2 is None:
        raise RuntimeError(
            "Cannot compute from variance-less histograms, try a Weight storage"
        )

    return "Performing ks two sample test"
