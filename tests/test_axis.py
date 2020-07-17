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
