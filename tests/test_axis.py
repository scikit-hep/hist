from hist import axis
import boost_histogram as bh
import pytest
import numpy as np


def test_basic_usage():
    """
        Test basic usage -- whether axis names work.
    """

    # right axis names
    assert axis.Regular(50, -3, 3, name="x0")
    assert axis.Bool(name="x_")
    assert axis.Variable(range(-3, 3), name="xx")
    assert axis.Integer(-3, 3, name="x_x")
    assert axis.IntCategory(range(-3, 3), name="X__X")
    assert axis.StrCategory("FT", name="X00")

    assert axis.Regular(50, -3, 3, name="")
    assert axis.Bool(name="")
    assert axis.Variable(range(-3, 3))
    assert axis.Integer(-3, 3, name="")
    assert axis.IntCategory(range(-3, 3), name="")
    assert axis.StrCategory("FT")


def test_errors():
    """
        Test errors -- whether name exceptions are thrown.
    """

    # wrong axis names: protected or private prefix
    with pytest.raises(Exception):
        axis.Regular(50, -3, 3, name="_x")

    with pytest.raises(Exception):
        axis.Bool(name="__x")

    with pytest.raises(Exception):
        axis.Variable(range(-3, 3), name="_x_")

    with pytest.raises(Exception):
        axis.Integer(-3, 3, name="_x0")

    with pytest.raises(Exception):
        axis.IntCategory(range(-3, 3), name="_0x")

    with pytest.raises(Exception):
        axis.StrCategory("FT", name="_xX")

    # wrong axis names: number prefix
    with pytest.raises(Exception):
        axis.Regular(50, -3, 3, name="0")

    with pytest.raises(Exception):
        axis.Bool(name="00")

    with pytest.raises(Exception):
        axis.Variable(range(-3, 3), name="0x")

    with pytest.raises(Exception):
        axis.Integer(-3, 3, name="0_x")

    with pytest.raises(Exception):
        axis.IntCategory(range(-3, 3), name="00x")

    with pytest.raises(Exception):
        axis.StrCategory("FT", name="0xx")

    # wrong axis names: unsupported chr
    with pytest.raises(Exception):
        axis.Regular(50, -3, 3, name="-x")

    with pytest.raises(Exception):
        axis.Bool(name="x-")

    with pytest.raises(Exception):
        axis.Variable(range(-3, 3), name="?x")

    with pytest.raises(Exception):
        axis.Integer(-3, 3, name="%x")

    with pytest.raises(Exception):
        axis.IntCategory(range(-3, 3), name="#x")

    with pytest.raises(Exception):
        axis.StrCategory("FT", name="*x")
