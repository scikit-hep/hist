from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats as spstats
from scipy.stats import rv_continuous, rv_discrete

import hist


def _get_cdf_if_valid(obj: Any) -> Any:
    if isinstance(obj, (rv_continuous, rv_discrete)):
        return obj.cdf
    if isinstance(obj, str) and hasattr(spstats, obj):
        dist = getattr(spstats, obj)
        if isinstance(dist, (rv_continuous, rv_discrete)):
            return dist.cdf
    if hasattr(obj, "cdf") and callable(obj.cdf):
        return obj.cdf
    if callable(obj):
        return obj
    raise TypeError(
        f"Unknown distribution type {obj}, try one of the scipy distributions, an object with a cdf method, or a callable cdf implementation"
    )


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
    cdf = _get_cdf_if_valid(distribution)

    variances = self.variances()
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )

    observed = self.values()
    totalentries = self.sum()
    expected = np.diff(cdf(self.axes[0].edges, *args, **kwds)) * totalentries
    variances = (
        expected  # TODO: check if variances or expected should go in the denominator
    )
    where = variances != 0
    squares = (expected - observed) ** 2
    ndof = where.sum() - 1
    chisq = np.sum(squares[where] / variances[where])
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
    # TODO: add support for compatible rebinning
    if self.size != other.size:
        raise NotImplementedError(
            "Cannot compute chi2 from histograms with different binning, try rebinning"
        )
    if not np.allclose(self.axes[0].edges, other.axes[0].edges):
        raise NotImplementedError(
            "Cannot compute chi2 from histograms with different binning, try rebinning"
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
    ndof = k if self.sum() == other.sum() else k - 1
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
    cdf = _get_cdf_if_valid(distribution)

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
    # TODO: add support for compatible rebinning
    if self.size != other.size:
        raise NotImplementedError(
            "Cannot compute chi2 from histograms with different binning, try rebinning"
        )
    if not np.allclose(self.axes[0].edges, other.axes[0].edges):
        raise NotImplementedError(
            "Cannot compute chi2 from histograms with different binning, try rebinning"
        )

    variances1 = self.variances()
    variances2 = other.variances()
    if variances1 is None or variances2 is None:
        raise RuntimeError(
            "Cannot compute from variance-less histograms, try a Weight storage"
        )

    return "Performing ks two sample test"
