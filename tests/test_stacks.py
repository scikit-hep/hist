import sys

import numpy as np
import pytest

from hist import Hist, NamedHist, Stack, axis

# different histograms here!
reg_hist = Hist(axis.Regular(10, 0, 1)).fill(np.random.randn(10))
boo_hist = Hist(axis.Boolean()).fill([True, False, True])
var_hist = Hist(axis.Variable(range(-3, 3))).fill(np.random.randn(10))
int_hist = Hist(axis.Integer(-3, 3)).fill(np.random.randn(10))
int_cat_hist = Hist(axis.IntCategory(range(-3, 3))).fill(np.random.randn(10))
str_cat_hist = Hist(axis.StrCategory(["F", "T"])).fill(
    ["T", "F", "T"]
)  # why this happens
named_reg_hist = NamedHist(axis.Regular(10, 0, 1, name="A")).fill(A=np.random.randn(10))
named_boo_hist = NamedHist(axis.Boolean(name="B")).fill(B=[True, False, True])
named_var_hist = NamedHist(axis.Variable(range(-3, 3), name="C")).fill(
    C=np.random.randn(10)
)
named_int_hist = NamedHist(axis.Integer(-3, 3, name="D")).fill(D=np.random.randn(10))
named_int_cat_hist = NamedHist(axis.IntCategory(range(-3, 3), name="E")).fill(
    E=np.random.randn(10)
)
named_str_cat_hist = NamedHist(axis.StrCategory(["F", "T"], name="F")).fill(
    F=["T", "F", "T"]
)
reg_hist_2d = Hist(axis.Regular(10, 0, 1), axis.Regular(10, 0, 1)).fill(
    np.random.randn(10), np.random.randn(10)
)
boo_hist_2d = Hist(axis.Boolean(), axis.Boolean()).fill(
    [True, False, True], [True, False, True]
)
var_hist_2d = Hist(axis.Variable(range(-3, 3)), axis.Variable(range(-3, 3))).fill(
    np.random.randn(10), np.random.randn(10)
)
int_hist_2d = Hist(axis.Integer(-3, 3), axis.Integer(-3, 3)).fill(
    np.random.randn(10), np.random.randn(10)
)
int_cat_hist_2d = Hist(
    axis.IntCategory(range(-3, 3)), axis.IntCategory(range(-3, 3))
).fill(np.random.randn(10), np.random.randn(10))
str_cat_hist_2d = Hist(axis.StrCategory(["F", "T"]), axis.StrCategory(["F", "T"])).fill(
    ["T", "F", "T"], ["T", "F", "T"]
)


@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires Python 3.7 or higher")
def test_stack_init():
    """
    Test stack init -- whether Stack can be properly initialized.
    """
    # allow to construct stack with same-type and same-type-axis histograms
    assert Stack(reg_hist, reg_hist, reg_hist)
    assert Stack(boo_hist, boo_hist, boo_hist)
    assert Stack(var_hist, var_hist, var_hist)
    assert Stack(int_hist, int_hist, int_hist)
    assert Stack(int_cat_hist, int_cat_hist, int_cat_hist)
    assert Stack(str_cat_hist, str_cat_hist, str_cat_hist)

    # allow to construct stack with different-type but same-type-axis histograms
    # assert Stack(reg_hist, named_reg_hist)
    # assert Stack(boo_hist, named_boo_hist)
    # assert Stack(var_hist, named_var_hist)
    # assert Stack(int_hist, named_int_hist)
    # assert Stack(int_cat_hist, named_int_cat_hist)
    # assert Stack(str_cat_hist, named_str_cat_hist)

    # not allow to construct stack with same-type but different-type-axis histograms
    with pytest.raises(Exception):
        Stack(reg_hist, boo_hist, var_hist)
    with pytest.raises(Exception):
        Stack(int_hist, int_cat_hist, str_cat_hist)

    # allow to construct stack with 2d histograms
    assert Stack(reg_hist_2d, reg_hist_2d, reg_hist_2d)
    assert Stack(boo_hist_2d, boo_hist_2d, boo_hist_2d)
    assert Stack(var_hist_2d, var_hist_2d, var_hist_2d)
    assert Stack(int_hist_2d, int_hist_2d, int_hist_2d)
    assert Stack(int_cat_hist_2d, int_cat_hist_2d, int_cat_hist_2d)
    assert Stack(str_cat_hist_2d, str_cat_hist_2d, str_cat_hist_2d)

    # not allow to constuct stack with different ndim
    with pytest.raises(Exception):
        Stack(reg_hist, reg_hist_2d)
    with pytest.raises(Exception):
        Stack(boo_hist, boo_hist_2d)
    with pytest.raises(Exception):
        Stack(var_hist, var_hist_2d)
    with pytest.raises(Exception):
        Stack(int_hist, int_hist_2d)
    with pytest.raises(Exception):
        Stack(int_cat_hist, int_cat_hist_2d)
    with pytest.raises(Exception):
        Stack(str_cat_hist, str_cat_hist_2d)


@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires Python 3.7 or higher")
def test_stack_plot():
    """
    Test stack plot -- whether Stack can be properly plot.
    """
    # allow to plot stack with 1d histogram
    assert Stack(reg_hist, reg_hist, reg_hist).plot()
    assert Stack(boo_hist, boo_hist, boo_hist).plot()
    assert Stack(var_hist, var_hist, var_hist).plot()
    assert Stack(int_hist, int_hist, int_hist).plot()
    assert Stack(int_cat_hist, int_cat_hist, int_cat_hist).plot()
    assert Stack(str_cat_hist, str_cat_hist, str_cat_hist).plot()

    # allow to plot stack with projection of 2d histograms
    assert Stack(reg_hist_2d.project(0)).plot()
    assert Stack(boo_hist_2d.project(0)).plot()
    assert Stack(var_hist_2d.project(0)).plot()
    assert Stack(int_hist_2d.project(0)).plot()
    assert Stack(int_cat_hist_2d.project(0)).plot()
    assert Stack(str_cat_hist_2d.project(0)).plot()

    # not allow to plot stack with 2d histograms
    with pytest.raises(Exception):
        Stack(reg_hist_2d).plot()

    with pytest.raises(Exception):
        Stack(boo_hist_2d).plot()

    with pytest.raises(Exception):
        Stack(var_hist_2d).plot()

    with pytest.raises(Exception):
        Stack(int_hist_2d).plot()

    with pytest.raises(Exception):
        Stack(int_cat_hist_2d).plot()

    with pytest.raises(Exception):
        Stack(str_cat_hist_2d).plot()
