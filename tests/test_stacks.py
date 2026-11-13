from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest import approx

from hist import Hist, NamedHist, Stack, axis

np.random.seed(42)

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
int_hist = Hist(int_ax).fill(np.random.randint(-3, 3, size=10))
int_cat_hist = Hist(int_cat_ax).fill(np.random.randint(-3, 3, size=10))
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
named_int_hist = NamedHist(named_int_ax).fill(D=np.random.randint(-3, 3, size=10))
named_int_cat_hist = NamedHist(named_int_cat_ax).fill(
    E=np.random.randint(-3, 3, size=10)
)
named_str_cat_hist = NamedHist(named_str_cat_ax).fill(F=["T", "F", "T"])

reg_hist_2d = Hist(reg_ax, reg_ax).fill(np.random.randn(10), np.random.randn(10))

boo_hist_2d = Hist(boo_ax, boo_ax).fill([True, False, True], [True, False, True])
var_hist_2d = Hist(var_ax, var_ax).fill(np.random.randn(10), np.random.randn(10))
int_hist_2d = Hist(int_ax, int_ax).fill(
    np.random.randint(-3, 3, size=10), np.random.randint(-3, 3, size=10)
)
int_cat_hist_2d = Hist(int_cat_ax, int_cat_ax).fill(
    np.random.randint(-3, 3, size=10), np.random.randint(-3, 3, size=10)
)
str_cat_hist_2d = Hist(str_cat_ax, str_cat_ax).fill(["T", "F", "T"], ["T", "F", "T"])

axs = (reg_ax, boo_ax, var_ax, int_ax, int_cat_ax, str_cat_ax)
fills = (int, bool, int, int, int, str)
ids = ("reg", "boo", "var", "int", "icat", "scat")


@pytest.fixture(params=zip(axs, fills, strict=True), ids=ids)
def hist_1d(request):
    def make_hist():
        ax, fill = request.param
        h = Hist(ax)
        if fill is int:
            h.fill(np.random.randint(-3, 3, size=10))
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
    assert stack.axes == h1.axes
    assert stack.axes == h2.axes
    assert stack.axes == h3.axes

    assert tuple(stack) == (h1, h2, h3)


def test_stack_from_iter(hist_1d):
    h1 = hist_1d()
    h2 = hist_1d()
    h3 = hist_1d()

    stack = Stack.from_iter([h1, h2, h3])

    assert stack[0] == h1
    assert stack[1] == h2
    assert stack[2] == h3


def test_stack_from_dict(hist_1d):
    h1 = hist_1d()
    h2 = hist_1d()
    h3 = hist_1d()

    d = {"one": h1, "two": h2, "three": h3}
    stack = Stack.from_dict(d)

    assert stack[0].name == "one"
    assert stack[1].name == "two"
    assert stack[2].name == "three"

    assert stack["one"].name == "one"
    assert stack["two"].name == "two"
    assert stack["three"].name == "three"

    assert stack[0:1] == stack["one":"two"]


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
    s1 = Stack(reg_hist_2d, reg_hist_2d, reg_hist_2d)
    s2 = Stack(boo_hist_2d, boo_hist_2d, boo_hist_2d)
    s3 = Stack(var_hist_2d, var_hist_2d, var_hist_2d)
    s4 = Stack(int_hist_2d, int_hist_2d, int_hist_2d)
    s5 = Stack(int_cat_hist_2d, int_cat_hist_2d, int_cat_hist_2d)
    s6 = Stack(str_cat_hist_2d, str_cat_hist_2d, str_cat_hist_2d)

    assert s1.axes == reg_hist_2d.axes
    assert s2.axes == boo_hist_2d.axes
    assert s3.axes == var_hist_2d.axes
    assert s4.axes == int_hist_2d.axes
    assert s5.axes == int_cat_hist_2d.axes
    assert s6.axes == str_cat_hist_2d.axes

    # not allow to construct stack with different ndim
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


def test_stack_plot_construct():
    """
    Test stack plot -- whether Stack can be properly plot.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

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
    return MagicMock(), args, kwargs


def test_stack_plot(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

    import hist.plot

    monkeypatch.setattr(hist.plot, "histplot", collect)
    monkeypatch.setattr(hist.plot, "_plot_keywords_wrapper", collect)

    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    _, args, kwargs = s.plot(silly=...)

    assert len(s) == 2
    assert len(list(s)) == 2

    assert args == (list(s),)
    assert kwargs["label"] == ["one", "two"]
    assert kwargs["silly"] == ...


def test_stack_add():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s += 1

    assert s[0].values() == approx(np.ones(10))
    assert s[1].values() == approx(np.ones(10))

    assert (s + 1)[0].values() == approx(np.ones(10) * 2)


def test_stack_sub():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s -= 1

    assert s[0].values() == approx(-np.ones(10))
    assert s[1].values() == approx(-np.ones(10))

    assert (s - 1)[0].values() == approx(-np.ones(10) * 2)


def test_stack_mul():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s += 1
    s *= 2

    assert s[0].values() == approx(np.ones(10) * 2)
    assert s[1].values() == approx(np.ones(10) * 2)

    assert (s * 2)[0].values() == approx(np.ones(10) * 4)


def test_stack_div():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s += 4
    s /= 2

    assert s[0].values() == approx(np.ones(10) * 2)
    assert s[1].values() == approx(np.ones(10) * 2)

    assert (s / 2)[0].values() == approx(np.ones(10))
    assert (s / np.arange(1, 11))[0].values() == approx(2 / np.arange(1, 11))
    assert s[0].values() == approx(np.ones(10) * 2)


def test_stack_array_add():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s += np.arange(10)

    assert s[0].values() == approx(np.arange(10))
    assert s[1].values() == approx(np.arange(10))

    assert (s + np.arange(10))[0].values() == approx(np.arange(10) * 2)


def test_stack_array_mul():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s += 1
    s *= np.arange(10)

    assert s[0].values() == approx(np.arange(10))
    assert s[1].values() == approx(np.arange(10))

    assert (s * np.arange(10))[0].values() == approx(np.arange(10) ** 2)


def test_project():
    h = (
        Hist.new.Regular(10, 0, 1, name="first")
        .StrCategory(["one", "two"], name="str")
        .StrCategory(["other", "thing"], name="again")
        .Double()
    )
    s = h.stack("str")
    s += np.ones((10, 2))
    s = s.project("first")

    assert s[0].values() == approx(2 * np.ones(10))


def test_set_item():
    h = Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
    s = h.stack(1)
    s[0] = Hist.new.Regular(10, 0, 1).Double() + 3
    assert s[0].values() == approx(np.ones(10) * 3)

    with pytest.raises(ValueError):
        s[0] = (
            Hist.new.Regular(10, 0, 1).StrCategory(["one", "two"], name="str").Double()
        )

    with pytest.raises(ValueError):
        s[0] = Hist.new.Regular(10, 0, 2).Double()

    with pytest.raises(ValueError):
        s[0] = Hist.new.Regular(11, 0, 1).Double()


@pytest.mark.parametrize(
    "storage_name", ["Double", "Int64", "Weight", "Mean", "WeightedMean"]
)
def test_to_hist_preserves_storage_and_flows(storage_name):
    import hist

    storage = getattr(hist.storage, storage_name)()
    h = Hist(axis.Regular(2, 0, 2, name="x", label="Position"), storage=storage)
    kwargs = {"sample": [2, 4, 6, 8]} if "Mean" in storage_name else {}
    if storage_name in {"Weight", "WeightedMean"}:
        kwargs["weight"] = [1, 2, 3, 4]
    h.fill([-1, 0.5, 1.5, 3], **kwargs)
    stack = Stack.from_dict({"first": h, "second": h + h})
    result = stack.to_hist(name="source", label="Source")
    assert list(result.axes[0]) == ["first", "second"]
    assert result.axes[0].label == "Source"
    assert result.axes[1] == h.axes[0]
    assert result.storage_type == h.storage_type
    for i, original in enumerate(stack):
        np.testing.assert_array_equal(
            result.view(flow=True)[i], original.view(flow=True)
        )
        assert result[{"source": original.name}] == original
    result.reset()
    np.testing.assert_array_equal(stack[0].view(flow=True), h.view(flow=True))


def test_to_hist_rejects_ambiguous_labels_and_storage():
    import hist

    h = Hist(axis.Regular(2, 0, 2, name="x"))
    with pytest.raises(ValueError, match="string or all int names"):
        Stack(h).to_hist()
    with pytest.raises(ValueError, match="unique"):
        Stack(Hist(h, name="a"), Hist(h, name="a")).to_hist()
    with pytest.raises(ValueError, match="conflicts"):
        Stack.from_dict({"a": h}).to_hist(name="x")
    weighted = Hist(*h.axes, storage=hist.storage.Weight(), name="b")
    with pytest.raises(ValueError, match="same storage"):
        Stack(Hist(h, name="a"), weighted).to_hist()


@pytest.mark.parametrize("hist_class", [Hist, NamedHist])
def test_to_hist_multidimensional_and_unknown_variance(hist_class):
    h = hist_class(
        axis.Regular(2, 0, 2, name="x", underflow=False),
        axis.Boolean(name="flag"),
    ).fill(x=[0.5, 3], flag=[True, False], weight=[2, 3])
    result = Stack.from_dict({"a": h}).to_hist()
    np.testing.assert_array_equal(result.values(flow=True)[0], h.values(flow=True))
    assert result.variances() is None
    assert h.variances() is None


def test_to_hist_scalar():
    h = Hist(name="scalar")
    result = Stack(h).to_hist()
    assert result.shape == (1,)
    np.testing.assert_array_equal(result.values(), [0])


def test_to_hist_copies_axes():
    h = Hist(axis.Regular(2, 0, 2, name="x", label="Position"))
    stack = Stack.from_dict({"a": h, "b": h})
    result = stack.to_hist()
    result.axes[1].label = "changed"
    assert stack[0].axes[0].label == "Position"


def test_to_hist_keeps_class():
    h = NamedHist(axis.Regular(2, 0, 2, name="x"))
    result = Stack.from_dict({"a": h}).to_hist()
    assert type(result) is NamedHist


def test_to_hist_int_category():
    h = (
        Hist.new.IntCat([1, 2], name="c")
        .Reg(2, 0, 1, name="x")
        .Double()
        .fill(c=[1, 2, 2], x=[0.2, 0.2, 0.7])
    )
    result = h.stack("c").to_hist(name="c")
    assert isinstance(result.axes[0], axis.IntCategory)
    np.testing.assert_array_equal(result.values(), h.values())
