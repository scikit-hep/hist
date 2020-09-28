# -*- coding: utf-8 -*-
from hist import Hist, NamedHist, axis

import pytest
import numpy as np

unp = pytest.importorskip("uncertainties.unumpy")
plt = pytest.importorskip("matplotlib.pyplot")


def test_general_plot1d():
    """
    Test general plot1d -- whether 1d-Hist can be plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10))

    assert h.plot1d(color="green", ls="--", lw=3)
    plt.close("all")

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        h.plot1d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot1d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot1d(ls="red")

    plt.close("all")


def test_general_plot2d():
    """
    Test general plot2d -- whether 2d-Hist can be plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot2d(cmap="cividis")

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d(cmap=0.1)

    plt.close("all")


def test_general_plot2d_full():
    """
    Test general plot2d_full -- whether 2d-Hist can be fully plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot2d_full(
        main_cmap="cividis",
        top_ls="--",
        top_color="orange",
        top_lw=2,
        side_ls="-.",
        side_lw=1,
        side_color="steelblue",
    )

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d_full()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d_full(abc="red")

    with pytest.raises(Exception):
        h.plot2d_full(color="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d_full(main_cmap=0.1, side_lw="autumn")

    plt.close("all")


def test_general_plot():
    """
    Test general plot -- whether Hist can be plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10))

    assert h.plot(color="green", ls="--", lw=3)

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot(cmap="cividis")

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="C", label="c [units]", underflow=False, overflow=False
        ),
    ).fill(
        np.random.normal(size=10), np.random.normal(size=10), np.random.normal(size=10)
    )

    with pytest.raises(Exception):
        h.plot()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot(abc="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot(ls="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(cmap=0.1)

    plt.close("all")


def test_general_plot_pull():
    """
    Test general plot_pull -- whether 1d-Hist can be plotted pull properly.
    """

    h = Hist(
        axis.Regular(
            50, -4, 4, name="S", label="s [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=10))

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        exp = unp.exp if a.dtype == np.dtype("O") else np.exp
        return a * exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    assert h.plot_pull(
        pdf,
        eb_ecolor="crimson",
        eb_mfc="crimson",
        eb_mec="crimson",
        eb_fmt="o",
        eb_ms=6,
        eb_capsize=1,
        eb_capthick=2,
        eb_alpha=0.8,
        fp_c="chocolate",
        fp_ls="-",
        fp_lw=3,
        fp_alpha=1.0,
        bar_fc="orange",
        pp_num=6,
        pp_fc="orange",
        pp_alpha=0.618,
        pp_ec=None,
    )

    # dimension error
    hh = Hist(
        axis.Regular(
            50, -4, 4, name="X", label="s [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="Y", label="s [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        hh.plot_pull(pdf)

    # not callable
    with pytest.raises(Exception):
        h.plot_pull("1")

    with pytest.raises(Exception):
        h.plot_pull(1)

    with pytest.raises(Exception):
        h.plot_pull(0.1)

    with pytest.raises(Exception):
        h.plot_pull((1, 2))

    with pytest.raises(Exception):
        h.plot_pull([1, 2])

    with pytest.raises(Exception):
        h.plot_pull({"a": 1})

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot_pull(pdf, abc="crimson", xyz="crimson")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ecolor="crimson", mfc="crimson")

    # not disabled params
    h.plot_pull(pdf, eb_label="value")

    h.plot_pull(pdf, fp_label="value")

    h.plot_pull(pdf, ub_label="value")

    h.plot_pull(pdf, bar_label="value")

    h.plot_pull(pdf, pp_label="value")

    # disabled params
    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_width="value")

    # wrong kwargs types
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_ecolor=1.0, eb_mfc=1.0)  # kwargs should be str

    plt.close("all")


def test_named_plot1d():
    """
    Test named plot1d -- whether 1d-NamedHist can be plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(A=np.random.normal(size=10))

    assert h.plot1d(color="green", ls="--", lw=3)
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    with pytest.raises(Exception):
        h.plot1d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot1d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot1d(ls="red")

    plt.close("all")


def test_named_plot2d():
    """
    Test named plot2d -- whether 2d-NamedHist can be plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    assert h.plot2d(cmap="cividis")
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d(cmap=0.1)

    plt.close("all")


def test_named_plot2d_full():
    """
    Test named plot2d_full -- whether 2d-NamedHist can be fully plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    assert h.plot2d_full(
        main_cmap="cividis",
        top_ls="--",
        top_color="orange",
        top_lw=2,
        side_ls="-.",
        side_lw=1,
        side_color="steelblue",
    )
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d_full()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d_full(abc="red")

    with pytest.raises(Exception):
        h.plot2d_full(color="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d_full(main_cmap=0.1, side_lw="autumn")

    plt.close("all")


def test_named_plot():
    """
    Test named plot -- whether NamedHist can be plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(A=np.random.normal(size=10))

    assert h.plot(color="green", ls="--", lw=3)

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    assert h.plot(cmap="cividis")
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="C", label="c [units]", underflow=False, overflow=False
        ),
    ).fill(
        A=np.random.normal(size=10),
        B=np.random.normal(size=10),
        C=np.random.normal(size=10),
    )

    with pytest.raises(Exception):
        h.plot()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot(abc="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot(ls="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(cmap=0.1)

    plt.close("all")


def test_named_plot_pull():
    """
    Test named plot_pull -- whether 1d-NamedHist can be plotted pull properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -4, 4, name="S", label="s [units]", underflow=False, overflow=False
        )
    ).fill(S=np.random.normal(size=10))

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        exp = unp.exp if a.dtype == np.dtype("O") else np.exp
        return a * exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    assert h.plot_pull(
        pdf,
        eb_ecolor="crimson",
        eb_mfc="crimson",
        eb_mec="crimson",
        eb_fmt="o",
        eb_ms=6,
        eb_capsize=1,
        eb_capthick=2,
        eb_alpha=0.8,
        fp_c="chocolate",
        fp_ls="-",
        fp_lw=3,
        fp_alpha=1.0,
        bar_fc="orange",
        pp_num=6,
        pp_fc="orange",
        pp_alpha=0.618,
        pp_ec=None,
    )

    # dimension error
    hh = NamedHist(
        axis.Regular(
            50, -4, 4, name="X", label="s [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="Y", label="s [units]", underflow=False, overflow=False
        ),
    ).fill(X=np.random.normal(size=10), Y=np.random.normal(size=10))

    with pytest.raises(Exception):
        hh.plot_pull(pdf)

    # not callable
    with pytest.raises(Exception):
        h.plot_pull("1")

    with pytest.raises(Exception):
        h.plot_pull(1)

    with pytest.raises(Exception):
        h.plot_pull(0.1)

    with pytest.raises(Exception):
        h.plot_pull((1, 2))

    with pytest.raises(Exception):
        h.plot_pull([1, 2])

    with pytest.raises(Exception):
        h.plot_pull({"a": 1})
    plt.close("all")

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot_pull(pdf, abc="crimson", xyz="crimson")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ecolor="crimson", mfc="crimson")

    # not disabled params
    h.plot_pull(pdf, eb_label="value")

    h.plot_pull(pdf, fp_label="value")

    h.plot_pull(pdf, ub_label="value")

    h.plot_pull(pdf, bar_label="value")

    h.plot_pull(pdf, pp_label="value")

    # disabled params
    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_width="value")

    # wrong kwargs types
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_ecolor=1.0, eb_mfc=1.0)  # kwargs should be str

    plt.close("all")
