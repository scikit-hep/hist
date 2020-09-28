# -*- coding: utf-8 -*-
def test_1D_empty_repr(named_hist):

    h = named_hist.new.Reg(10, -1, 1, name="x").Double()
    html = h._repr_html_()
    assert html


def test_1D_intcat_empty_repr(named_hist):

    h = named_hist.new.IntCat([1, 3, 5], name="x").Double()
    html = h._repr_html_()
    assert html


def test_1D_strcat_empty_repr(named_hist):

    h = named_hist.new.StrCat(["1", "3", "5"], name="x").Double()
    html = h._repr_html_()
    assert html


def test_2D_empty_repr(named_hist):

    h = named_hist.new.Reg(10, -1, 1, name="x").Int(0, 15, name="y").Double()
    html = h._repr_html_()
    assert html


def test_1D_circ_empty_repr(named_hist):

    h = named_hist.new.Reg(10, -1, 1, circular=True, name="r").Double()
    html = h._repr_html_()
    assert html


def test_ND_empty_repr(named_hist):

    h = (
        named_hist.new.Reg(10, -1, 1, name="x")
        .Reg(12, -3, 3, name="y")
        .Reg(15, -2, 4, name="z")
        .Double()
    )
    html = h._repr_html_()
    assert html
