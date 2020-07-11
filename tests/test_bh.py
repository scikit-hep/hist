# -*- coding: utf-8 -*-
import boost_histogram as bh
import hist


def test_bh_conversion():
    h = bh.Histogram(bh.axis.Regular(3, 2, 1, metadata={"name": "x"}))

    h2 = hist.Hist(h)

    assert isinstance(h2.axes[0], hist.axis.Regular)
    assert h2.axes[0].name == "x"
    assert h2.axes[0].metadata == {"name": "x"}  # MAY CHANGE

    h3 = bh.Histogram(h2)

    assert not isinstance(h3.axes[0], hist.axis.Regular)
    assert h3.axes[0].metadata == {"name": "x"}
