from __future__ import annotations

import numpy as np
import pytest

import hist
from hist import Hist


def test_stattest():
    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000))

    assert h.chisquare_1samp() == "Performing chi square one sample test"
    assert h.chisquare_2samp() == "Performing chi square two sample test"
    assert h.ks_1samp() == "Performing ks one sample test"
    assert h.ks_2samp() == "Performing ks two sample test"

    h = Hist(hist.axis.Regular(20, -5, 5), hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), np.random.normal(size=1000))

    with pytest.raises(
        NotImplementedError, match="chisquare_1samp is only supported for 1D histograms"
    ):
        h.chisquare_1samp()
    with pytest.raises(
        NotImplementedError, match="chisquare_2samp is only supported for 1D histograms"
    ):
        h.chisquare_2samp()
    with pytest.raises(
        NotImplementedError, match="ks_1samp is only supported for 1D histograms"
    ):
        h.ks_1samp()
    with pytest.raises(
        NotImplementedError, match="ks_2samp is only supported for 1D histograms"
    ):
        h.ks_2samp()
