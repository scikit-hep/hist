# -*- coding: utf-8 -*-
from hist import axis


def test_axis_names():
    """
    Test axis names -- whether axis names work.
    """

    assert axis.Regular(50, -3, 3, name="x0")
    assert axis.Boolean(name="x_")
    assert axis.Variable(range(-3, 3), name="xx")
    assert axis.Integer(-3, 3, name="x_x")
    assert axis.IntCategory(range(-3, 3), name="X__X")
    assert axis.StrCategory("FT", name="X00")

    assert axis.Regular(50, -3, 3, name="")
    assert axis.Boolean(name="")
    assert axis.Variable(range(-3, 3))
    assert axis.Integer(-3, 3, name="")
    assert axis.IntCategory(range(-3, 3), name="")
    assert axis.StrCategory("FT")


def test_axis_flow():
    assert axis.Regular(9, 0, 8, flow=False) == axis.Regular(
        9, 0, 8, underflow=False, overflow=False
    )
    assert axis.Variable([1, 2, 3], flow=False) == axis.Variable(
        [1, 2, 3], underflow=False, overflow=False
    )
    assert axis.Integer(0, 8, flow=False) == axis.Integer(
        0, 8, underflow=False, overflow=False
    )

    assert axis.Regular(9, 0, 8, flow=False, underflow=True) == axis.Regular(
        9, 0, 8, overflow=False
    )
    assert axis.Variable([1, 2, 3], flow=False, underflow=True) == axis.Variable(
        [1, 2, 3], overflow=False
    )
    assert axis.Integer(0, 8, flow=False, underflow=True) == axis.Integer(
        0, 8, overflow=False
    )

    assert axis.Regular(9, 0, 8, flow=False, overflow=True) == axis.Regular(
        9, 0, 8, underflow=False
    )
    assert axis.Variable([1, 2, 3], flow=False, overflow=True) == axis.Variable(
        [1, 2, 3], underflow=False
    )
    assert axis.Integer(0, 8, flow=False, overflow=True) == axis.Integer(
        0, 8, underflow=False
    )
