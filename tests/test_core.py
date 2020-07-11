# -*- coding: utf-8 -*-
import hist
from hist import axis, BaseHist
import boost_histogram as bh
import pytest
import numpy as np
from uncertainties import unumpy as unp


def test_version():
    assert hist.__version__ is not None


def test_core_init():
    """
        Test core init -- whether BaseHist can be properly initialized.
    """

    # basic
    h = BaseHist(axis.Regular(10, 0, 1)).fill([0.35, 0.35, 0.45])

    for idx in range(10):
        if idx == 3:
            assert h[idx] == h[{0: idx}] == 2
        elif idx == 4:
            assert h[idx] == h[{0: idx}] == 1
        else:
            assert h[idx] == h[{0: idx}] == 0

    # with named axes
    assert BaseHist(
        axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
    ).fill(np.random.randn(10), np.random.randn(10))

    assert BaseHist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
        [True, False, True], [True, False, True]
    )

    assert BaseHist(
        axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
    ).fill(np.random.randn(10), np.random.randn(10))

    assert BaseHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")).fill(
        np.random.randn(10), np.random.randn(10)
    )

    assert BaseHist(
        axis.IntCategory(range(-3, 3), name="x"),
        axis.IntCategory(range(-3, 3), name="y"),
    ).fill(np.random.randn(10), np.random.randn(10))

    assert BaseHist(
        axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
    ).fill(["T", "F", "T"], ["T", "F", "T"])

    # with no-named axes
    assert BaseHist(axis.Regular(50, -3, 3, name=""), axis.Regular(50, -3, 3, name="x"))

    assert BaseHist(axis.Boolean(name=""), axis.Boolean(name="y"))

    assert BaseHist(
        axis.Variable(range(-3, 3)), axis.Variable(range(-3, 3), name="x")
    )  # name=None will be converted to name=''

    assert BaseHist(axis.Integer(-3, 3, name=""), axis.Integer(-3, 3, name="x"))

    assert BaseHist(
        axis.IntCategory(range(-3, 3), name=""),
        axis.IntCategory(range(-3, 3), name="x"),
    )

    assert BaseHist(
        axis.StrCategory("TF"), axis.StrCategory(["T", "F"], name="x")
    )  # name=None will be converted to name=''

    # with duplicated names
    with pytest.raises(Exception):
        BaseHist(axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="x"))

    with pytest.raises(Exception):
        BaseHist(axis.Boolean(name="y"), axis.Boolean(name="y"))

    with pytest.raises(Exception):
        BaseHist(
            axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="x")
        )

    with pytest.raises(Exception):
        BaseHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="x"))

    with pytest.raises(Exception):
        BaseHist(
            axis.IntCategory(range(-3, 3), name="x"),
            axis.IntCategory(range(-3, 3), name="x"),
        )

    with pytest.raises(Exception):
        BaseHist(
            axis.StrCategory("TF", name="y"), axis.StrCategory(["T", "F"], name="y")
        )


def test_core_fill():
    """
        Test core fill -- whether BaseHist can be properly filled.
    """

    # Regular
    h = BaseHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(10, 0, 1, name="y"),
        axis.Regular(2, 0, 2, name="z"),
    ).fill(
        [0.35, 0.35, 0.35, 0.45, 0.55, 0.55, 0.55],
        [0.35, 0.35, 0.45, 0.45, 0.45, 0.45, 0.45],
        [0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{2: bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 4 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0

    # Boolean
    h = BaseHist(
        axis.Boolean(name="x"), axis.Boolean(name="y"), axis.Boolean(name="z"),
    ).fill(
        [True, True, True, True, True, False, True],
        [False, True, True, False, False, True, False],
        [False, False, True, True, True, True, True],
    )

    z_one_only = h[{2: bh.loc(True)}]
    assert z_one_only[False, False] == 0
    assert z_one_only[False, True] == 1
    assert z_one_only[True, False] == 3
    assert z_one_only[True, True] == 1

    # Variable
    h = BaseHist(
        axis.Variable(range(11), name="x"),
        axis.Variable(range(11), name="y"),
        axis.Variable(range(3), name="z"),
    ).fill(
        [3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        [3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        [0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{2: bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 4 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0

    # Integer
    h = BaseHist(
        axis.Integer(0, 10, name="x"),
        axis.Integer(0, 10, name="y"),
        axis.Integer(0, 2, name="z"),
    ).fill(
        [3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        [3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        [0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{2: bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 4 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0

    # IntCategory
    h = BaseHist(
        axis.IntCategory(range(10), name="x"),
        axis.IntCategory(range(10), name="y"),
        axis.IntCategory(range(2), name="z"),
    ).fill(
        [3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        [3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        [0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{2: bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 4 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0

    # StrCategory
    h = BaseHist(
        axis.StrCategory("FT", name="x"),
        axis.StrCategory(list("FT"), name="y"),
        axis.StrCategory(["F", "T"], name="z"),
    ).fill(
        ["T", "T", "T", "T", "T", "F", "T"],
        ["F", "T", "T", "F", "F", "T", "F"],
        ["F", "F", "T", "T", "T", "T", "T"],
    )

    z_one_only = h[{2: bh.loc("T")}]
    assert z_one_only[bh.loc("F"), bh.loc("F")] == 0
    assert z_one_only[bh.loc("F"), bh.loc("T")] == 1
    assert z_one_only[bh.loc("T"), bh.loc("F")] == 3
    assert z_one_only[bh.loc("T"), bh.loc("T")] == 1

    # with names
    with pytest.raises(Exception):
        BaseHist(
            axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
        ).fill(x=np.random.randn(10), y=np.random.randn(10))

    with pytest.raises(Exception):
        BaseHist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
            x=[True, False, True], y=[True, False, True]
        )

    with pytest.raises(Exception):
        BaseHist(
            axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
        ).fill(x=np.random.randn(10), y=np.random.randn(10))

    with pytest.raises(Exception):
        BaseHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")).fill(
            x=np.random.randn(10), y=np.random.randn(10)
        )

    with pytest.raises(Exception):
        BaseHist(
            axis.IntCategory(range(-3, 3), name="x"),
            axis.IntCategory(range(-3, 3), name="y"),
        ).fill(x=np.random.randn(10), y=np.random.randn(10))

    with pytest.raises(Exception):
        BaseHist(
            axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
        ).fill(x=["T", "F", "T"], y=["T", "F", "T"])

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        exp = unp.exp if a.dtype == np.dtype("O") else np.exp
        return a * exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    h = BaseHist(
        axis.Regular(
            50, -4, 4, name="X", title="s [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=10))


def test_core_access():
    """
        Test core access -- whether BaseHist bins can be accessed.
    """

    h = BaseHist(axis.Regular(10, -5, 5, name="X", title="x [units]")).fill(
        np.random.normal(size=1000)
    )

    assert h[6] == h[bh.loc(1)] == h[1j] == h[0j + 1] == h[-3j + 4] == h[bh.loc(1, 0)]
    h[6] = h[bh.loc(1)] = h[1j] = h[0j + 1] = h[-3j + 4] = h[bh.loc(1, 0)] = 0

    h = BaseHist(
        axis.Regular(50, -5, 5, name="Norm", title="normal distribution"),
        axis.Regular(50, -5, 5, name="Unif", title="uniform distribution"),
        axis.StrCategory(["hi", "hello"], name="Greet"),
        axis.Boolean(name="Yes"),
        axis.Integer(0, 1000, name="Int"),
    ).fill(
        np.random.normal(size=1000),
        np.random.uniform(size=1000),
        ["hi"] * 800 + ["hello"] * 200,
        [True] * 600 + [False] * 400,
        np.ones(1000),
    )

    assert h[0j, -0j + 2, "hi", True, 1]


def test_core_project():
    """
        Test core project -- whether BaseHist can be projected properly.
    """

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Boolean(name="B", title="b [units]"),
        axis.Variable(range(11), name="C", title="c [units]"),
        axis.Integer(0, 10, name="D", title="d [units]"),
        axis.IntCategory(range(10), name="E", title="e [units]"),
        axis.StrCategory("FT", name="F", title="f [units]"),
    )

    # via indices
    assert h.project()
    assert h.project(0, 1)
    assert h.project(0, 1, 2, 3, 4, 5)

    # via names
    assert h.project()
    assert h.project("A", "B")
    assert h.project("A", "B", "C", "D", "E", "F")

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Boolean(name="B", title="b [units]"),
        axis.Variable(range(11), name="C", title="c [units]"),
        axis.Integer(0, 10, name="D", title="d [units]"),
        axis.IntCategory(range(10), name="E", title="e [units]"),
        axis.StrCategory("FT", name="F", title="f [units]"),
    )

    # duplicated
    with pytest.raises(Exception):
        h.project(0, 0)

    with pytest.raises(Exception):
        h.project("A", "A")

    # wrong/mixed types
    with pytest.raises(Exception):
        h.project(2, "A")

    with pytest.raises(Exception):
        h.project(True, "A")

    # cannot found
    with pytest.raises(Exception):
        h.project(-1, 9)

    with pytest.raises(Exception):
        h.project("G", "H")


def test_core_plot1d():
    """
        Test core plot1d -- whether 1d-BaseHist can be plotted properly.
    """

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10))

    assert h.plot1d(color="green", ls="--", lw=3)

    # dimension error
    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
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


def test_core_plot2d():
    """
        Test core plot2d -- whether 2d-BaseHist can be plotted properly.
    """

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot2d(cmap="cividis")

    # dimension error
    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
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


def test_core_plot2d_full():
    """
        Test core plot2d_full -- whether 2d-BaseHist can be fully plotted properly.
    """

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
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
    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
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


def test_core_plot():
    """
        Test core plot -- whether BaseHist can be plotted properly.
    """

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10))

    assert h.plot(color="green", ls="--", lw=3)

    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot(cmap="cividis")

    # dimension error
    h = BaseHist(
        axis.Regular(
            50, -5, 5, name="A", title="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", title="b [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="C", title="c [units]", underflow=False, overflow=False
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


def test_core_plot_pull():
    """
        Test core plot_pull -- whether 1d-BaseHist can be plotted pull properly.
    """

    h = BaseHist(
        axis.Regular(
            50, -4, 4, name="S", title="s [units]", underflow=False, overflow=False
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
        vp_c="gold",
        vp_ls="-",
        vp_lw=8,
        vp_alpha=0.6,
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
    hh = BaseHist(
        axis.Regular(
            50, -4, 4, name="X", title="s [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="Y", title="s [units]", underflow=False, overflow=False
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

    # disabled params
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, vp_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, fp_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ub_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, pp_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ub_color="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_width="value")

    # wrong kwargs types
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_ecolor=1.0, eb_mfc=1.0)  # kwargs should be str
