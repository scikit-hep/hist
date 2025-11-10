from __future__ import annotations

import pytest

from hist import Hist


def test_axestuple():
    """
    Test axistuples -- whether axistuple setattr works.
    """

    h = Hist.new.Regular(50, -3, 3, name="X").Regular(20, -3, 3, name="Y").Double()
    h.axes.name = ("A", "B")
    h.axes.label = ("A-unit", "B-unit")

    assert h.axes[0].name == "A"
    assert h.axes[1].name == "B"
    assert h.axes[0].label == "A-unit"
    assert h.axes[1].label == "B-unit"

    with (
        pytest.warns(UserWarning, match="weight is a protected keyword"),
        pytest.warns(UserWarning, match="sample is a protected keyword"),
    ):
        h.axes.name = ("weight", "sample")

    with pytest.raises(Exception, match="cannot contain axes with duplicated names"):
        h.axes.name = ("A", "A")

    # Our backport has a simpler error message
    with pytest.raises(
        Exception,
        match=r"argument 2 is longer than argument 1|arguments are not the same length",
    ):
        h.axes.name = ("A", "B", "C")
