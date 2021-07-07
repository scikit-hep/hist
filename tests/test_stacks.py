import sys

import numpy as np
import pytest

from hist import Hist, NamedHist, Stack, axis

# different histograms here!
reg_ax = axis.Regular(10, 0, 1)
boo_ax = axis.Boolean()
var_ax = axis.Variable(range(-3, 3))
int_ax = axis.Integer(-3, 3)
int_cat_ax = axis.IntCategory(range(-3, 3))
str_cat_ax = axis.StrCategory(["F", "T"])

reg_hist = Hist(reg_ax).fill(np.random.randn(10))
boo_hist = Hist(boo_ax).fill([True, False, True])
var_hist = Hist(var_ax).fill(np.random.randn(10))
int_hist = Hist(int_ax).fill(np.random.randn(10))
int_cat_hist = Hist(int_cat_ax).fill(np.random.randn(10))
str_cat_hist = Hist(str_cat_ax).fill(["T", "F", "T"])

named_reg_ax = axis.Regular(10, 0, 1, name="A")
named_boo_ax = axis.Boolean(name="B")
named_var_ax = axis.Variable(range(-3, 3), name="C")
named_int_ax = axis.Integer(-3, 3, name="D")
named_int_cat_ax = axis.IntCategory(range(-3, 3), name="E")
named_str_cat_ax = axis.StrCategory(["F", "T"], name="F")

named_reg_hist = NamedHist(named_reg_ax).fill(A=np.random.randn(10))
named_boo_hist = NamedHist(named_boo_ax).fill(B=[True, False, True])
named_var_hist = NamedHist(named_var_ax).fill(C=np.random.randn(10))
named_int_hist = NamedHist(named_int_ax).fill(D=np.random.randn(10))
named_int_cat_hist = NamedHist(named_int_cat_ax).fill(E=np.random.randn(10))
named_str_cat_hist = NamedHist(named_str_cat_ax).fill(F=["T", "F", "T"])

reg_hist_2d = Hist(reg_ax, reg_ax).fill(np.random.randn(10), np.random.randn(10))

boo_hist_2d = Hist(boo_ax, boo_ax).fill([True, False, True], [True, False, True])
var_hist_2d = Hist(var_ax, var_ax).fill(np.random.randn(10), np.random.randn(10))
int_hist_2d = Hist(int_ax, int_ax).fill(np.random.randn(10), np.random.randn(10))
int_cat_hist_2d = Hist(int_cat_ax, int_cat_ax).fill(
    np.random.randn(10), np.random.randn(10)
)
str_cat_hist_2d = Hist(str_cat_ax, str_cat_ax).fill(["T", "F", "T"], ["T", "F", "T"])

axs = (reg_ax, boo_ax, var_ax, int_ax, int_cat_ax, str_cat_ax)
fills = (int, bool, int, int, int, str)
ids = ("reg", "boo", "var", "int", "icat", "scat")


@pytest.fixture(params=zip(axs, fills), ids=ids)
def hist_1d(request):
    def make_hist():
        ax, fill = request.param
        h = Hist(ax)
        if fill is int:
            h.fill(np.random.randn(10))
        elif fill is bool:
            h.fill(np.random.randint(0, 1, size=10) == 1)
        elif fill is str:
            h.fill(np.random.choice(("T", "F"), size=10))
        return h

    return make_hist


def test_stack_init(hist_1d):
    """
    Test stack init -- whether Stack can be properly initialized.
    """
    h1 = hist_1d()
    h2 = hist_1d()
    h3 = hist_1d()

    # Allow to construct stack with same-type and same-type-axis histograms
    stack = Stack(h1, h2, h3)
    assert stack[0] == h1
    assert stack[1] == h2
    assert stack[2] == h3

    assert tuple(stack) == (h1, h2, h3)


def test_stack_constructor_fails():
    # Don't allow construction directly from axes with no Histograms
    with pytest.raises(Exception):
        assert Stack(reg_ax)

    with pytest.raises(Exception):
        assert Stack(reg_ax, reg_ax, reg_ax)

    # not allow to construct stack with different-type but same-type-axis histograms
    with pytest.raises(Exception):
        Stack(reg_hist, named_reg_hist)
    with pytest.raises(Exception):
        assert Stack(boo_hist, named_boo_hist)
    with pytest.raises(Exception):
        Stack(var_hist, named_var_hist)
    with pytest.raises(Exception):
        Stack(int_hist, named_int_hist)
    with pytest.raises(Exception):
        Stack(int_cat_hist, named_int_cat_hist)
    with pytest.raises(Exception):
        Stack(str_cat_hist, named_str_cat_hist)

    # not allow to construct stack with same-type but different-type-axis histograms
    with pytest.raises(Exception):
        Stack(reg_hist, boo_hist, var_hist)
    with pytest.raises(Exception):
        Stack(int_hist, int_cat_hist, str_cat_hist)

    # allow to construct stack with 2d histograms
    Stack(reg_hist_2d, reg_hist_2d, reg_hist_2d)
    Stack(boo_hist_2d, boo_hist_2d, boo_hist_2d)
    Stack(var_hist_2d, var_hist_2d, var_hist_2d)
    Stack(int_hist_2d, int_hist_2d, int_hist_2d)
    Stack(int_cat_hist_2d, int_cat_hist_2d, int_cat_hist_2d)
    Stack(str_cat_hist_2d, str_cat_hist_2d, str_cat_hist_2d)

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

    # not allow to struct stack from histograms with different axes
    with pytest.raises(Exception):
        NamedHist(named_reg_ax, axis.Regular(10, 0, 1, name="X")).stack("A", "X")
    with pytest.raises(Exception):
        NamedHist(named_boo_ax, axis.Boolean(name="X")).stack("B", "X")
    with pytest.raises(Exception):
        NamedHist(named_var_ax, axis.Variable(range(-3, 3), name="X")).stack("C", "X")
    with pytest.raises(Exception):
        NamedHist(named_int_ax, axis.Integer(-3, 3, name="X")).stack("D", "X")
    with pytest.raises(Exception):
        NamedHist(named_int_cat_ax, axis.IntCategory(range(-3, 3), name="X")).stack(
            "E", "X"
        )
    with pytest.raises(Exception):
        NamedHist(named_str_cat_ax, axis.StrCategory(["F", "T"], name="X")).stack(
            "F", "X"
        )


@pytest.mark.skipif(sys.version_info < (3, 7), reason="requires Python 3.7 or higher")
def test_stack_plot_construct():
    """
    Test stack plot -- whether Stack can be properly plot.
    """
    # not allow axes stack to plot
    with pytest.raises(Exception):
        Stack(reg_ax, reg_ax, reg_ax).plot()
    with pytest.raises(Exception):
        Stack(boo_ax, boo_ax, boo_ax).plot()
    with pytest.raises(Exception):
        Stack(var_ax, var_ax, var_ax).plot()
    with pytest.raises(Exception):
        Stack(int_ax, int_ax, int_ax).plot()
    with pytest.raises(Exception):
        Stack(int_cat_ax, int_cat_ax, int_cat_ax).plot()
    with pytest.raises(Exception):
        Stack(str_cat_ax, str_cat_ax, str_cat_ax).plot()

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


def test_stack_method():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    assert s[0].axes[0] == h.axes[0]
    assert s[0].name == "one"
    assert s[1].name == "two"

    s2 = h.stack("str")
    assert s2[0].axes[0] == h.axes[0]
    assert s2[0].name == "one"
    assert s2[1].name == "two"


def collect(*args, **kwargs):
    return args, kwargs


def test_stack_plot(monkeypatch):
    import hist.plot

    monkeypatch.setattr(hist.plot, "histplot", collect)

    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)

    args, kwargs = s.plot(silly=...)

    assert len(s) == 2
    assert len(list(s)) == 2

    assert args == (list(s),)
    assert kwargs == {"label": ["one", "two"], "silly": ...}
