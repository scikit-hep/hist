from __future__ import annotations

import numpy as np
import pytest

import hist
from hist import Hist
from numba_stats import norm
import functools

cdf =functools.partial(norm.cdf, loc=0, scale=1)

pytest.importorskip("scipy")

@pytest.mark.parametrize("arg", ["norm", cdf])
def test_chisquare_1samp(arg):
    np.random.seed(42)

    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000))

    chisq, ndof, pvalue = h.chisquare_1samp(arg)
    assert chisq == pytest.approx(9.47425856230132)
    assert ndof == 14
    assert pvalue == pytest.approx(0.7995235818496339)
    
def test_chisquare_1samp_2d():
    np.random.seed(42)

    h = Hist(hist.axis.Regular(20, -5, 5), hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), np.random.normal(size=1000))
    with pytest.raises(NotImplementedError):
        h.chisquare_1samp("norm")

def test_chisquare_1samp_weight():
    np.random.seed(42)
    h = Hist(hist.axis.Regular(20, -5, 5), storage=hist.storage.Weight())
    h.fill(np.random.normal(size=1000), weight=np.random.randint(0, 10, size=1000))
    h.chisquare_1samp("norm")   

def test_chisquare_1samp_weight_raises():
    np.random.seed(42)
    h = Hist(hist.axis.Regular(20, -5, 5))
    h.fill(np.random.normal(size=1000), weight=np.random.randint(0, 10, size=1000))
    with pytest.raises(RuntimeError):
        h.chisquare_1samp("norm")

def test_chisquare_1samp_invalid_args():
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
    assert ndof == 14
    assert pvalue == pytest.approx(0.5342659447391381)

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
    pass


def test_ks_2samp():
    pass
