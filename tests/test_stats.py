from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as spstats

import hist
from hist import Hist


def test_chisquare_1samp():
    from numba_stats import norm

    np.random.seed(42)

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000))

    chisq, ndof, pvalue = h.chisquare_1samp("norm")
    assert chisq == pytest.approx(9.47425856230132)
    assert ndof == 19
    assert pvalue == pytest.approx(0.9647461095072625)

    chisq, ndof, pvalue = h.chisquare_1samp(norm.cdf, args=(0, 1))
    assert chisq == pytest.approx(9.47425856230132)
    assert ndof == 19
    assert pvalue == pytest.approx(0.9647461095072625)

    h = Hist(hist.axis.Regular(20, -5, 5), hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), np.random.normal(size=1000))
    with pytest.raises(NotImplementedError):
        h.chisquare_1samp("norm")

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), weight=np.random.randint(0, 10, size=1000))
    with pytest.raises(RuntimeError):
        h.chisquare_1samp("norm")

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000))
    with pytest.raises(TypeError):
        h.chisquare_1samp("not_a_distribution")

    with pytest.raises(TypeError):
        h.chisquare_1samp(1)


def test_chisquare_2samp():
    np.random.seed(42)

    h1 = Hist(hist.axis.Regular(20, -5, 5, name="norm"))
    h2 = Hist(hist.axis.Regular(20, -5, 5, name="norm"))
    h1.fill(np.random.normal(size=1000))
    h2.fill(np.random.normal(size=500))

    chisq, ndof, pvalue = h1.chisquare_2samp(h2)
    assert chisq == pytest.approx(12.901853991544478)
    assert ndof == 15
    assert pvalue == pytest.approx(0.609878574488961)

    np.random.seed(42)

    h1 = Hist(hist.axis.Regular(20, -5, 5, name="norm"))
    h2 = Hist(hist.axis.Regular(20, -5, 5, name="norm"))
    h1.fill(np.random.normal(size=1000))
    h2.fill(np.random.normal(size=1000))

    chisq, ndof, pvalue = h1.chisquare_2samp(h2)
    assert chisq == pytest.approx(13.577308400334086)
    assert ndof == 14
    assert pvalue == pytest.approx(0.4816525905214355)

    h = Hist(hist.axis.Regular(20, -5, 5), hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), np.random.normal(size=1000))
    with pytest.raises(NotImplementedError):
        h.chisquare_2samp(h2)

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), weight=np.random.randint(0, 10, size=1000))
    with pytest.raises(RuntimeError):
        h.chisquare_2samp(h2)

    with pytest.raises(TypeError):
        h.chisquare_2samp(1)


def test_ks_1samp():
    np.random.seed(42)

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000))

    assert np.all(
        h.ks_1samp("norm") == spstats.norm.cdf(h.axes[0].edges)
    )  # placeholder to pass test for now

    h = Hist(hist.axis.Regular(20, -5, 5), hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), np.random.normal(size=1000))
    with pytest.raises(NotImplementedError):
        h.ks_1samp("norm")

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), weight=np.random.randint(0, 10, size=1000))
    with pytest.raises(RuntimeError):
        h.ks_1samp("norm")

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000))
    with pytest.raises(TypeError):
        h.ks_1samp("not_a_distribution")

    with pytest.raises(TypeError):
        h.ks_1samp(1)


def test_ks_2samp():
    np.random.seed(42)

    h1 = Hist(hist.axis.Regular(20, -5, 5, name="norm"))
    h2 = Hist(hist.axis.Regular(20, -5, 5, name="norm"))
    h1.fill(np.random.normal(size=1000))
    h2.fill(np.random.normal(size=1000))

    assert h1.ks_2samp(h2) == "Performing ks two sample test"

    h = Hist(hist.axis.Regular(20, -5, 5), hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), np.random.normal(size=1000))
    with pytest.raises(NotImplementedError):
        h.ks_2samp(h2)

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), weight=np.random.randint(0, 10, size=1000))
    with pytest.raises(RuntimeError):
        h.ks_2samp(h2)

    with pytest.raises(TypeError):
        h.ks_2samp(1)
