from __future__ import annotations

from hist import Stack, axis


def test_1D_empty_repr(named_hist):
    h = named_hist.new.Reg(10, -1, 1, name="x", label="y").Double()
    html = h._repr_html_()
    assert html
    assert "name='x'" in repr(h)
    assert "label='y'" in repr(h)


def test_1D_var_empty_repr(named_hist):
    h = named_hist.new.Var(range(10), name="x", label="y").Double()
    html = h._repr_html_()
    assert html
    assert "name='x'" in repr(h)
    assert "label='y'" in repr(h)


def test_1D_int_empty_repr(named_hist):
    h = named_hist.new.Int(-9, 9, name="x", label="y").Double()
    html = h._repr_html_()
    assert html
    assert "name='x'" in repr(h)
    assert "label='y'" in repr(h)


def test_1D_intcat_empty_repr(named_hist):
    h = named_hist.new.IntCat([1, 3, 5], name="x", label="y").Double()
    html = h._repr_html_()
    assert html
    assert "name='x'" in repr(h)
    assert "label='y'" in repr(h)


def test_1D_strcat_empty_repr(named_hist):
    h = named_hist.new.StrCat(["1", "3", "5"], name="x", label="y").Double()
    html = h._repr_html_()
    assert html
    assert "name='x'" in repr(h)
    assert "label='y'" in repr(h)


def test_2D_empty_repr(named_hist):
    h = (
        named_hist.new.Reg(10, -1, 1, name="x", label="y")
        .Int(0, 15, name="p", label="q")
        .Double()
    )
    html = h._repr_html_()
    assert html
    assert "name='x'" in repr(h)
    assert "name='p'" in repr(h)
    assert "label='y'" in repr(h)
    assert "label='q'" in repr(h)


def test_1D_circ_empty_repr(named_hist):
    h = named_hist.new.Reg(10, -1, 1, circular=True, name="R", label="r").Double()
    html = h._repr_html_()
    assert html
    assert "name='R'" in repr(h)
    assert "label='r'" in repr(h)


def test_ND_empty_repr(named_hist):
    h = (
        named_hist.new.Reg(10, -1, 1, name="x", label="y")
        .Reg(12, -3, 3, name="p", label="q")
        .Reg(15, -2, 4, name="a", label="b")
        .Double()
    )
    html = h._repr_html_()
    assert html is None


def test_empty_mega_repr(named_hist):
    h = named_hist.new.Reg(1001, -1, 1, name="x").Double()
    html = h._repr_html_()
    assert html is None

    h = named_hist.new.Reg(201, -1, 1, name="x").Reg(100, 0, 1, name="y").Double()
    html = h._repr_html_()
    assert html is None


def test_stack_repr(named_hist):
    a1 = axis.Regular(
        50, -5, 5, name="A", label="a [unit]", underflow=False, overflow=False
    )
    a2 = axis.Regular(
        50, -5, 5, name="A", label="a [unit]", underflow=False, overflow=False
    )
    assert "name='A'" in repr(Stack(named_hist(a1), named_hist(a2)))
    assert "label='a [unit]'" in repr(Stack(named_hist(a1), named_hist(a2)))
