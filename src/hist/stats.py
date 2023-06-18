from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy import stats as spstats
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._stats_py import _attempt_exact_2kssamp

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
    totalentries = self.values(flow=True).sum()
    expected = np.diff(cdf(self.axes[0].edges, *args, **kwds)) * totalentries
    # TODO: check if variances or expected should go in the denominator
    where = expected != 0
    squares = (expected - observed) ** 2
    ndof = np.sum(where) - 1
    chisq = np.sum(squares[where] / expected[where])
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
    totalentries1 = self.values(flow=True).sum()
    totalentries2 = other.values(flow=True).sum()
    squares = (
        counts1 * np.sqrt(totalentries2 / totalentries1)
        - counts2 * np.sqrt(totalentries1 / totalentries2)
    ) ** 2
    variances = variances1 + variances2
    where = variances != 0
    chisq = np.sum(squares[where] / variances[where])
    k = np.sum(where)
    # TODO: check if ndof = k if totalentries1 == totalentries2 else k - 1 is correct
    ndof = k - 1
    pvalue = spstats.chi2.sf(chisq, ndof)

    return chisq, ndof, pvalue


def ks_1samp(
    self: hist.BaseHist,
    distribution: str | Callable[..., Any],
    args: Any = (),
    kwds: Any = None,
    alternative: str = "two-sided",
    mode: str = "auto",
) -> Any:
    kwds = {} if kwds is None else kwds

    if self.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports ks_1samp, try projecting {self.__class__.__name__} to 1D"
        )

    if mode not in ["auto", "exact", "asymp"]:
        raise ValueError(f"Invalid value for mode: {mode}")
    alternative = {"t": "two-sided", "g": "greater", "l": "less"}.get(
        alternative.lower()[0], alternative
    )
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(f"Invalid value for alternative: {alternative}")

    cdf = _get_cdf_if_valid(distribution)

    cdflocs = self.axes[0].edges[:-1]
    cdfvals = cdf(cdflocs, *args, **kwds)
    observed = self.values(flow=True)
    totalentries = observed.sum()
    ecdf = np.cumsum(observed) / totalentries
    ecdfplus = ecdf[1:-1]
    ecdfminus = ecdf[0:-2]
    dplus_index = (ecdfplus - cdfvals).argmax()
    dminus_index = (cdfvals - ecdfminus).argmax()
    Dplus = (ecdfplus - cdfvals)[dplus_index]
    Dminus = (cdfvals - ecdfminus)[dminus_index]
    dplus_location = cdflocs[dplus_index]
    dminus_location = cdflocs[dminus_index]

    if Dplus > Dminus:
        D = Dplus
        d_location = dplus_location
        d_sign = 1
    else:
        D = Dminus
        d_location = dminus_location
        d_sign = -1

    if alternative == "greater":
        pvalue = spstats.ksone.sf(Dplus, totalentries)
        return Dplus, dplus_location, 1, pvalue
    if alternative == "less":
        pvalue = spstats.ksone.sf(Dminus, totalentries)
        return Dminus, dminus_location, -1, pvalue

    if mode == "auto":  # Always select exact
        mode = "exact"
    if mode == "exact":
        pvalue = spstats.kstwo.sf(D, totalentries)
    elif mode == "asymp":
        pvalue = spstats.kstwobign.sf(D * np.sqrt(totalentries))
    elif mode == "approx":
        pvalue = 2 * spstats.ksone.sf(D, totalentries)

    pvalue = np.clip(pvalue, 0, 1)

    return D, Dplus, Dminus, dplus_location, dminus_location, d_location, d_sign, pvalue


def ks_2samp(
    self: hist.BaseHist,
    other: hist.BaseHist,
    equal_bins: bool = True,
    alternative: str = "two-sided",
    mode: str = "auto",
) -> Any:
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

    if equal_bins:
        if self.size != other.size:
            raise NotImplementedError(
                "Cannot compute KS from histograms with different binning, try rebinning"
            )
        if not np.allclose(self.axes[0].edges, other.axes[0].edges):
            raise NotImplementedError(
                "Cannot compute KS from histograms with different binning, try rebinning"
            )

    if mode not in ["auto", "exact", "asymp"]:
        raise ValueError(f"Invalid value for mode: {mode}")
    alternative = {"t": "two-sided", "g": "greater", "l": "less"}.get(
        alternative.lower()[0], alternative
    )
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(f"Invalid value for alternative: {alternative}")

    data1 = self.values(flow=True)
    data2 = other.values(flow=True)
    n1 = data1.sum()
    n2 = data2.sum()
    edges1 = self.axes[0].edges[:-1]
    edges2 = other.axes[0].edges[:-1]
    cdf1 = np.cumsum(data1) / n1
    cdf2 = np.cumsum(data2) / n2

    if equal_bins:
        cdflocs = edges1

        cdf1 = cdf1[1:-1]
        cdf2 = cdf2[1:-1]
        cddiffs = cdf1 - cdf2

    elif not equal_bins:
        edges_all = np.sort(np.concatenate([edges1, edges2]))
        cdflocs = edges_all

        # Use np.searchsorted to get the CDF values at all bin edges
        cdf1_all = cdf1[np.searchsorted(edges1, edges_all, side="right")]
        cdf2_all = cdf2[np.searchsorted(edges2, edges_all, side="right")]
        cddiffs = cdf1_all - cdf2_all

    # Identify the location of the statistic
    argminS = np.argmin(cddiffs)
    argmaxS = np.argmax(cddiffs)
    loc_minS = cdflocs[argminS]
    loc_maxS = cdflocs[argmaxS]

    # Ensure sign of minS is not negative.
    minS = np.clip(-cddiffs[argminS], 0, 1)
    maxS = cddiffs[argmaxS]

    if alternative == "less" or (alternative == "two-sided" and minS > maxS):
        d = minS
        d_location = loc_minS
        d_sign = -1
    else:
        d = maxS
        d_location = loc_maxS
        d_sign = 1

    g = np.gcd(int(n1), int(n2))
    n1g = n1 // g
    n2g = n2 // g
    pvalue = -np.inf

    if mode == "auto":  # Always select exact
        mode = "exact"
    if mode == "exact" and n1g >= np.iinfo(np.int32).max / n2g:
        # If lcm(n1, n2) is too big, switch from exact to asymp
        mode = "asymp"
        warnings.warn(
            f"Exact ks_2samp calculation not possible with samples sizes "
            f"{n1} and {n2}. Switching to 'asymp'.",
            RuntimeWarning,
            stacklevel=3,
        )

    if mode == "exact":
        success, d, pvalue = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = "asymp"
            warnings.warn(
                f"ks_2samp: Exact calculation unsuccessful. "
                f"Switching to method={mode}.",
                RuntimeWarning,
                stacklevel=3,
            )

    if mode == "asymp":
        # The product n1*n2 is large.  Use Smirnov's asymptoptic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == "two-sided":
            pvalue = spstats.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
            pvalue = np.exp(expt)

    pvalue = np.clip(pvalue, 0, 1)
    D = d
    Dplus = maxS
    Dminus = minS
    dplus_location = loc_maxS
    dminus_location = loc_minS

    return D, Dplus, Dminus, dplus_location, dminus_location, d_location, d_sign, pvalue
