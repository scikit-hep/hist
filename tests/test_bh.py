# -*- coding: utf-8 -*-
import boost_histogram as bh
import hist


def test_bh_conversion():
    h = bh.Histogram(bh.axis.Regular(3, 2, 1, metadata={"name": "x"}))
    h.axes[0].name = "y"

    h2 = hist.Hist(h)

    assert isinstance(h2.axes[0], hist.axis.Regular)
    assert h2.axes[0].name == "y"
    assert h2.axes[0].metadata == {"name": "x"}

    h3 = bh.Histogram(h2)

    assert not isinstance(h3.axes[0], hist.axis.Regular)
    assert h2.axes[0].name == "y"
    assert h3.axes[0].metadata == {"name": "x"}
