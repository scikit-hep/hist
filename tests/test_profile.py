from __future__ import annotations

import numpy as np
from pytest import approx

import hist


def test_example():
    xy = np.array(
        [
            [-2, 1.5],
            [-2, -3.5],
            [-2, 1.5],  # x = -2
            [0.0, -2.0],
            [0.0, -2.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],  # x = 0
            [2, 1.5],  # x = +2
        ]
    )
    h = hist.Hist(
        hist.axis.Regular(5, -5, 5, name="x"), hist.axis.Regular(5, -5, 5, name="y")
    ).fill(*xy.T)

    # Profile out the y-axis
    hp = h.profile("y")

    # Exclude edge bins since no values from above will fall into them
    # When there are values there, ROOT does something funky in those bins,
    # despite these bins not being in the axis that is profiled out, and
    # despite there being no overflow... to be understood.
    assert hp.values()[1:-1] == approx(np.array([0.0, 0.4, 2.0]))
    assert hp.variances()[1:-1] == approx(
        np.array([2.66666667, 1.088, float("nan")]), nan_ok=True
    )


def test_profile_empty_bins():
    h = hist.Hist(
        hist.axis.Regular(3, 0, 3, name="x"), hist.axis.Regular(4, 0, 4, name="y")
    ).fill([0.5, 1.5], [1.5, 2.5])

    hp = h.profile("y")
    # An empty bin matches a Mean storage that was never filled
    expected = hist.Hist(hist.axis.Regular(3, 0, 3, name="x"), storage="mean")
    expected.fill([0.5, 1.5], sample=[1.5, 2.5])
    assert hp == expected

    # Filling an empty bin afterwards gives a finite mean
    hp.fill(x=[2.5], sample=[3.0])
    assert hp[2].count == 1
    assert hp[2].value == approx(3.0)

    # Profiling down to zero dimensions still works
    assert hist.Hist(hist.axis.Regular(3, 0, 3)).profile(0).values() == 0
