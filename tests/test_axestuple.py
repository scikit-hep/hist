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

    with pytest.warns(UserWarning):
        h.axes.name = ("weight", "sample")

    with pytest.raises(Exception):
        h.axes.name = ("A", "A")

    with pytest.raises(Exception):
        h.axes.name = ("A", "B", "C")
