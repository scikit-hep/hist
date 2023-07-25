from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from scipy import special
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


def _compute_outer_prob_inside_method(m: int, n: int, g: int, h: int) -> Any:
    """
    Count the proportion of paths that do not stay strictly inside two
    diagonal lines.

    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)

    Returns
    -------
    p : float
        The proportion of paths that do not stay inside the two lines.

    The classical algorithm counts the integer lattice paths from (0, 0)
    to (m, n) which satisfy |x/m - y/n| < h / lcm(m, n).
    The paths make steps of size +1 in either positive x or positive y
    directions.
    We are, however, interested in 1 - proportion to computes p-values,
    so we change the recursion to compute 1 - p directly while staying
    within the "inside method" a described by Hodges.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    For the recursion for 1-p see
    Viehmann, T.: "Numerically more stable computation of the p-values
    for the two-sample Kolmogorov-Smirnov test," arXiv: 2102.08037

    """
    # Probability is symmetrical in m, n.  Computation below uses m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Count the integer lattice paths from (0, 0) to (m, n) which satisfy
    # |nx/g - my/g| < h.
    # Compute matrix A such that:
    #  A(x, 0) = A(0, y) = 1
    #  A(x, y) = A(x, y-1) + A(x-1, y), for x,y>=1, except that
    #  A(x, y) = 0 if |x/m - y/n|>= h
    # Probability is A(m, n)/binom(m+n, n)
    # Optimizations exist for m==n, m==n*p.
    # Only need to preserve a single column of A, and only a
    # sliding window of it.
    # minj keeps track of the slide.
    minj, maxj = 0, min(int(np.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    # Make a vector long enough to hold maximum window needed.
    lenA = min(2 * maxj + 2, n + 1)
    # This is an integer calculation, but the entries are essentially
    # binomial coefficients, hence grow quickly.
    # Scaling after each column is computed avoids dividing by a
    # large binomial coefficient at the end, but is not sufficient to avoid
    # the large dyanamic range which appears during the calculation.
    # Instead we rescale based on the magnitude of the right most term in
    # the column and keep track of an exponent separately and apply
    # it at the end of the calculation.  Similarly when multiplying by
    # the binomial coefficient
    dtype = np.float64
    A = np.ones(lenA, dtype=dtype)
    # Initialize the first column
    A[minj:maxj] = 0.0
    for i in range(1, m + 1):
        # Generate the next column.
        # First calculate the sliding window
        lastminj, lastlen = minj, curlen
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 1.0
        # Now fill in the values. We cannot use cumsum, unfortunately.
        val = 0.0 if minj == 0 else 1.0
        for jj in range(maxj - minj):
            j = jj + minj
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j)
            A[jj] = val
        curlen = maxj - minj
        if lastlen > curlen:
            # Set some carried-over elements to 1
            A[maxj - minj : maxj - minj + (lastlen - curlen)] = 1

    return A[maxj - minj - 1]


def _compute_prob_outside_square(n: int, h: int) -> Any:
    """
    Compute the proportion of paths that pass outside the two diagonal lines.

    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n

    Returns
    -------
    p : float
        The proportion of paths that pass outside the lines x-y = +/-h.

    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )
    # / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with
        # h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


def _count_paths_outside_method(m: int, n: int, g: int, h: int) -> Any:
    """Count the number of paths that pass outside the specified diagonal.

    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)

    Returns
    -------
    p : float
        The number of paths that go low.
        The calculation may overflow - check for a finite answer.

    Count the integer lattice paths from (0, 0) to (m, n), which at some
    point (x, y) along the path, satisfy:
      m*y <= n*x - h*g
    The paths make steps of size +1 in either positive x or positive y
    directions.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    """
    # Compute #paths which stay lower than x/m-y/n = h/lcm(m,n)
    # B(x, y) = #{paths from (0,0) to (x,y) without
    #             previously crossing the boundary}
    #         = binom(x, y) - #{paths which already reached the boundary}
    # Multiply by the number of path extensions going from (x, y) to (m, n)
    # Sum.

    # Probability is symmetrical in m, n.  Computation below assumes m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Not every x needs to be considered.
    # xj holds the list of x values to be checked.
    # Wherever n*x/m + ng*h crosses an integer
    lxj = n + (mg - h) // mg
    xj = [(h + mg * j + ng - 1) // ng for j in range(lxj)]
    # B is an array just holding a few values of B(x,y), the ones needed.
    # B[j] == B(x_j, j)
    if lxj == 0:
        return special.binom(m + n, n)
    B = np.zeros(lxj)
    B[0] = 1
    # Compute the B(x, y) terms
    for j in range(1, lxj):
        Bj = special.binom(xj[j] + j, j)
        for i in range(j):
            bin = special.binom(xj[j] - xj[i] + j - i, j - i)
            Bj -= bin * B[i]
        B[j] = Bj
    # Compute the number of path extensions...
    num_paths = 0
    for j in range(lxj):
        bin = special.binom((m - xj[j]) + (n - j), n - j)
        term = B[j] * bin
        num_paths += term
    return num_paths


def _attempt_exact_2kssamp(n1: int, n2: int, g: int, d: float, alternative: str) -> Any:
    """Attempts to compute the exact 2sample probability.

    Parameters
    ----------
    n1 : integer
        sample size of sample 1
    n2 : integer
        sample size of sample 2
    g : integer
        greatest common divisor of n1 and n2
    d : float
        max difference in ECDFs
    alternative : string
        alternative hypothesis, either 'two-sided', 'less' or 'greater'

    Returns
    -------
    success : bool
        True if the calculation was successful
    d : float
        max difference in ECDFs
    prob : float
        The probability of the test statistic under the null hypothesis.
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    saw_fp_error, prob = False, np.nan
    try:
        with np.errstate(invalid="raise", over="raise"):
            if alternative == "two-sided":
                if n1 == n2:
                    prob = _compute_prob_outside_square(n1, h)
                else:
                    prob = _compute_outer_prob_inside_method(n1, n2, g, h)
            else:
                if n1 == n2:
                    # prob = binom(2n, n-h) / binom(2n, n)
                    # Evaluating in that form incurs roundoff errors
                    # from special.binom. Instead calculate directly
                    jrange = np.arange(h)
                    prob = float(np.prod((n1 - jrange) / (n1 + jrange + 1.0)))
                else:
                    with np.errstate(over="raise"):
                        num_paths = _count_paths_outside_method(n1, n2, g, h)
                    bin = special.binom(n1 + n2, n1)
                    if num_paths > bin or np.isinf(bin):
                        saw_fp_error = True
                    else:
                        prob = num_paths / bin

    except (FloatingPointError, OverflowError):
        saw_fp_error = True

    if saw_fp_error:
        return False, d, np.nan
    if not 0 <= prob <= 1:
        return False, d, prob
    return True, d, prob


def chisquare_1samp(
    self: hist.BaseHist,
    distribution: str | Callable[..., Any],
    flow: bool = False,
    args: Any = (),
    kwds: Any = None,
) -> Any:
    kwds = {} if kwds is None else kwds

    if self.ndim != 1:
        raise NotImplementedError(
            f"Only 1D-histogram supports chisquare_1samp, try projecting {self.__class__.__name__} to 1D"
        )
    cdf = _get_cdf_if_valid(distribution)

    variances = self.variances(flow=flow)
    if variances is None:
        raise RuntimeError(
            "Cannot compute from a variance-less histogram, try a Weight storage"
        )

    observed = self.values(flow=flow)
    totalentries = self.values(flow=True).sum(dtype=int)
    cdfvals = cdf(self.axes[0].edges, *args, **kwds)
    if flow:
        cdfvals = np.concatenate([[0], cdfvals, [1]])
    expected = np.diff(cdfvals) * totalentries
    # TODO: check if variances or expected should go in the denominator
    # variances fails if bins have low statistics
    where = variances != 0
    squares = (expected - observed) ** 2
    ndof = np.sum(where) - 1
    chisq = np.sum(squares[where] / variances[where])
    pvalue = spstats.chi2.sf(chisq, ndof)

    return chisq, ndof, pvalue


def chisquare_2samp(
    self: hist.BaseHist, other: hist.BaseHist, flow: bool = False
) -> Any:
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

    variances1 = self.variances(flow=flow)
    variances2 = other.variances(flow=flow)
    if variances1 is None or variances2 is None:
        raise RuntimeError(
            "Cannot compute from variance-less histograms, try a Weight storage"
        )

    counts1 = self.values(flow=flow)
    counts2 = other.values(flow=flow)
    totalentries1 = self.values(flow=True).sum(dtype=int)
    totalentries2 = other.values(flow=True).sum(dtype=int)
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

    cdflocs = self.axes[0].centers
    cdfvals = cdf(cdflocs, *args, **kwds)
    observed = self.values(flow=True)
    totalentries = observed.sum(dtype=int)
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
    flow: bool = False,
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
            raise ValueError(
                "Cannot compute KS from histograms with different binning, use equal_bins=False"
            )
        if not np.allclose(self.axes[0].edges, other.axes[0].edges):
            raise ValueError(
                "Cannot compute KS from histograms with different binning, use equal_bins=False"
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
    n1 = data1.sum(dtype=int)
    n2 = data2.sum(dtype=int)
    edges1 = self.axes[0].centers
    edges2 = other.axes[0].centers
    cdf1 = np.cumsum(data1) / n1
    cdf2 = np.cumsum(data2) / n2
    if flow:
        edges1 = np.concatenate([[-np.inf], edges1, [np.inf]])
        edges2 = np.concatenate([[-np.inf], edges2, [np.inf]])

    if equal_bins:
        cdflocs = edges1

        if not flow:
            cdf1 = cdf1[1:-1]
            cdf2 = cdf2[1:-1]

        cddiffs = cdf1 - cdf2

    elif not equal_bins:
        edges_all = np.sort(np.concatenate([edges1, edges2]))
        cdflocs = edges_all

        if flow:
            cdf1 = np.concatenate([[0], cdf1])
            cdf2 = np.concatenate([[0], cdf2])

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
