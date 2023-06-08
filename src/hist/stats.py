from __future__ import annotations

import hist


def chisquare_1samp(self: hist.BaseHist) -> str | NotImplementedError:
    if self.ndim != 1:
        raise NotImplementedError("chisquare_1samp is only supported for 1D histograms")
    return "Performing chi square one sample test"


def chisquare_2samp(self: hist.BaseHist) -> str | NotImplementedError:
    if self.ndim != 1:
        raise NotImplementedError("chisquare_2samp is only supported for 1D histograms")
    return "Performing chi square two sample test"


def ks_1samp(self: hist.BaseHist) -> str | NotImplementedError:
    if self.ndim != 1:
        raise NotImplementedError("ks_1samp is only supported for 1D histograms")
    return "Performing ks one sample test"


def ks_2samp(self: hist.BaseHist) -> str | NotImplementedError:
    if self.ndim != 1:
        raise NotImplementedError("ks_2samp is only supported for 1D histograms")
    return "Performing ks two sample test"
