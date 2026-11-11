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


@pytest.mark.parametrize(
    ("names", "error"),
    [
        pytest.param(("duplicate", "duplicate"), KeyError, id="duplicate"),
        pytest.param(("changed",), ValueError, id="too-short"),
        pytest.param(("changed", "second", "excess"), ValueError, id="too-long"),
    ],
)
@pytest.mark.parametrize("as_generator", [False, True])
def test_failed_rename_preserves_names(names, error, as_generator):
    h = Hist.new.Reg(2, 0, 2, name="x").Reg(2, 0, 2, name="y").Double()
    h.fill(x=0.5, y=0.5)
    proposed = (name for name in names) if as_generator else names

    with pytest.raises(error):
        h.axes.name = proposed

    assert h.axes.name == ("x", "y")
    h.fill(x=0.5, y=1.5)
    assert h[0, 0] == 1
    assert h[0, 1] == 1


def test_failed_name_generator_preserves_names():
    h = Hist.new.Reg(2, 0, 2, name="x").Reg(2, 0, 2, name="y").Double()

    def proposed_names():
        yield "changed"
        msg = "cannot read the remaining names"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="cannot read the remaining names"):
        h.axes.name = proposed_names()

    assert h.axes.name == ("x", "y")
    h.fill(x=0.5, y=1.5)
    assert h[0, 1] == 1


def test_rejected_name_warning_preserves_names():
    h = Hist.new.Reg(2, 0, 2, name="x").Reg(2, 0, 2, name="y").Double()

    with pytest.raises(UserWarning, match="weight is a protected keyword"):
        h.axes.name = ("weight", "changed")

    assert h.axes.name == ("x", "y")
    h.fill(x=0.5, y=1.5)
    assert h[0, 1] == 1


@pytest.mark.parametrize("names", [("renamed_x", "renamed_y"), ("", "")])
def test_rename_accepts_generator(names):
    h = Hist.new.Reg(2, 0, 2, name="x").Reg(2, 0, 2, name="y").Double()

    h.axes.name = (name for name in names)

    assert h.axes.name == names
    h.fill(0.5, 1.5)
    assert h[0, 1] == 1
