# -*- coding: utf-8 -*-
from hist import axis
import pytest


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

    # protected or private prefix
    with pytest.raises(Exception):
        axis.Regular(50, -3, 3, name="_x")

    with pytest.raises(Exception):
        axis.Boolean(name="__x")

    with pytest.raises(Exception):
        axis.Variable(range(-3, 3), name="_x_")

    with pytest.raises(Exception):
        axis.Integer(-3, 3, name="_x0")

    with pytest.raises(Exception):
        axis.IntCategory(range(-3, 3), name="_0x")

    with pytest.raises(Exception):
        axis.StrCategory("FT", name="_xX")

    # number prefix
    with pytest.raises(Exception):
        axis.Regular(50, -3, 3, name="0")

    with pytest.raises(Exception):
        axis.Boolean(name="00")

    with pytest.raises(Exception):
        axis.Variable(range(-3, 3), name="0x")

    with pytest.raises(Exception):
        axis.Integer(-3, 3, name="0_x")

    with pytest.raises(Exception):
        axis.IntCategory(range(-3, 3), name="00x")

    with pytest.raises(Exception):
        axis.StrCategory("FT", name="0xx")

    # unsupported chr
    with pytest.raises(Exception):
        axis.Regular(50, -3, 3, name="-x")

    with pytest.raises(Exception):
        axis.Boolean(name="x-")

    with pytest.raises(Exception):
        axis.Variable(range(-3, 3), name="?x")

    with pytest.raises(Exception):
        axis.Integer(-3, 3, name="%x")

    with pytest.raises(Exception):
        axis.IntCategory(range(-3, 3), name="#x")

    with pytest.raises(Exception):
        axis.StrCategory("FT", name="*x")
