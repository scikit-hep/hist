# -*- coding: utf-8 -*-
from hist import axis, NamedHist
import boost_histogram as bh
import pytest
import numpy as np
import ctypes
import math
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt

# ToDo: specify what error is raised


def test_named_init():
    """
        Test named init -- whether NamedHist can be properly initialized.
    """

    # basic
    h = NamedHist(
        axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")
    ).fill(x=[0.35, 0.35, 0.45], y=[0.35, 0.35, 0.45])

    for idx in range(10):
        if idx == 3:
            assert h[idx, idx] == 2
            assert h[{"x": idx, "y": idx}] == 2
            with pytest.raises(Exception):
                h[{0: idx, 1: idx}] == 2
        elif idx == 4:
            assert h[idx, idx] == 1
            with pytest.raises(Exception):
                h[{0: idx, 1: idx}] == 1
        else:
            assert h[idx, idx] == 0
            with pytest.raises(Exception):
                h[{0: idx, 1: idx}] == 0

    # with named axes
    assert NamedHist(
        axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert NamedHist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
        y=[True, False, True], x=[True, False, True]
    )

    assert NamedHist(
        axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert NamedHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")).fill(
        x=np.random.randn(10), y=np.random.randn(10)
    )

    assert NamedHist(
        axis.IntCategory(range(-3, 3), name="x"),
        axis.IntCategory(range(-3, 3), name="y"),
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert NamedHist(
        axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
    ).fill(y=["T", "F", "T"], x=["T", "F", "T"])

    # cannot access via index
    h = NamedHist(axis.Regular(10, 0, 1, name="x")).fill(x=[0.35, 0.35, 0.45])

    for idx in range(10):
        if idx == 3:
            with pytest.raises(Exception):
                h[idx] == h[{0: idx}] == 2
        elif idx == 4:
            with pytest.raises(Exception):
                h[idx] == h[{0: idx}] == 1
        else:
            with pytest.raises(Exception):
                h[idx] == h[{0: idx}] == 0

    # with no-named axes
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3), axis.Regular(50, -3, 3)).fill(
            x=np.random.randn(10), y=np.random.randn(10)
        )

    with pytest.raises(Exception):
        NamedHist(axis.Boolean(), axis.Boolean()).fill(
            y=[True, False, True], x=[True, False, True]
        )

    with pytest.raises(Exception):
        NamedHist(axis.Variable(range(-3, 3)), axis.Variable(range(-3, 3))).fill(
            x=np.random.randn(10), y=np.random.randn(10)
        )

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3), axis.Integer(-3, 3)).fill(
            x=np.random.randn(10), y=np.random.randn(10)
        )

    with pytest.raises(Exception):
        NamedHist(axis.IntCategory(range(-3, 3)), axis.IntCategory(range(-3, 3)),).fill(
            x=np.random.randn(10), y=np.random.randn(10)
        )

    with pytest.raises(Exception):
        NamedHist(axis.StrCategory(["F", "T"]), axis.StrCategory("FT")).fill(
            y=["T", "F", "T"], x=["T", "F", "T"]
        )

    # with duplicated names
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="x"))

    with pytest.raises(Exception):
        NamedHist(axis.Boolean(name="y"), axis.Boolean(name="y"))

    with pytest.raises(Exception):
        NamedHist(
            axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="x")
        )

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="x"))

    with pytest.raises(Exception):
        NamedHist(
            axis.IntCategory(range(-3, 3), name="x"),
            axis.IntCategory(range(-3, 3), name="x"),
        )

    with pytest.raises(Exception):
        NamedHist(
            axis.StrCategory("TF", name="y"), axis.StrCategory(["T", "F"], name="y")
        )


def test_named_fill():
    """
        Test named fill -- whether NamedHist can be properly filled.
    """

    # Regular
    h = NamedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(10, 0, 1, name="y"),
        axis.Regular(2, 0, 2, name="z"),
    ).fill(
        x=[0.35, 0.35, 0.35, 0.45, 0.55, 0.55, 0.55],
        y=[0.35, 0.35, 0.45, 0.45, 0.45, 0.45, 0.45],
        z=[0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{"z": bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 4 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 5 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 3
                )
            else:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 0
                )

    # Boolean
    h = NamedHist(
        axis.Boolean(name="x"), axis.Boolean(name="y"), axis.Boolean(name="z"),
    ).fill(
        x=[True, True, True, True, True, False, True],
        y=[False, True, True, False, False, True, False],
        z=[False, False, True, True, True, True, True],
    )

    z_one_only = h[{"z": bh.loc(True)}]
    assert z_one_only[False, False] == z_one_only[{"x": False, "y": False}] == 0
    assert z_one_only[False, True] == z_one_only[{"x": False, "y": True}] == 1
    assert z_one_only[True, False] == z_one_only[{"x": True, "y": False}] == 3
    assert z_one_only[True, True] == z_one_only[{"x": True, "y": True}] == 1

    # Variable
    h = NamedHist(
        axis.Variable(range(11), name="x"),
        axis.Variable(range(11), name="y"),
        axis.Variable(range(3), name="z"),
    ).fill(
        x=[3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        y=[3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        z=[0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{"z": bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 4 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 5 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 3
                )
            else:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 0
                )

    # Integer
    h = NamedHist(
        axis.Integer(0, 10, name="x"),
        axis.Integer(0, 10, name="y"),
        axis.Integer(0, 2, name="z"),
    ).fill(
        x=[3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        y=[3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        z=[0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{"z": bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 4 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 5 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 3
                )
            else:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 0
                )

    # IntCategory
    h = NamedHist(
        axis.IntCategory(range(10), name="x"),
        axis.IntCategory(range(10), name="y"),
        axis.IntCategory(range(2), name="z"),
    ).fill(
        x=[3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        y=[3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        z=[0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{"z": bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 4 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 5 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 3
                )
            else:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 0
                )

    # StrCategory
    h = NamedHist(
        axis.StrCategory("FT", name="x"),
        axis.StrCategory(list("FT"), name="y"),
        axis.StrCategory(["F", "T"], name="z"),
    ).fill(
        x=["T", "T", "T", "T", "T", "F", "T"],
        y=["F", "T", "T", "F", "F", "T", "F"],
        z=["F", "F", "T", "T", "T", "T", "T"],
    )

    z_one_only = h[{"z": bh.loc("T")}]
    assert z_one_only[bh.loc("F"), bh.loc("F")] == 0
    assert z_one_only[bh.loc("F"), bh.loc("T")] == 1
    assert z_one_only[bh.loc("T"), bh.loc("F")] == 3
    assert z_one_only[bh.loc("T"), bh.loc("T")] == 1

    # without names
    with pytest.raises(Exception):
        NamedHist(
            axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
        ).fill(np.random.randn(10), np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
            [True, False, True], [True, False, True]
        )

    with pytest.raises(Exception):
        NamedHist(
            axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
        ).fill(np.random.randn(10), np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")).fill(
            np.random.randn(10), np.random.randn(10)
        )

    with pytest.raises(Exception):
        NamedHist(
            axis.IntCategory(range(-3, 3), name="x"),
            axis.IntCategory(range(-3, 3), name="y"),
        ).fill(np.random.randn(10), np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(
            axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
        ).fill(["T", "F", "T"], ["T", "F", "T"])

    # wrong names
    with pytest.raises(Exception):
        NamedHist(
            axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
        ).fill(x=np.random.randn(10), z=np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
            y=[True, False, True], z=[True, False, True]
        )

    with pytest.raises(Exception):
        NamedHist(
            axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
        ).fill(z=np.random.randn(10), x=np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")).fill(
            x=np.random.randn(10), z=np.random.randn(10)
        )

    with pytest.raises(Exception):
        NamedHist(
            axis.IntCategory(range(-3, 3), name="x"),
            axis.IntCategory(range(-3, 3), name="y"),
        ).fill(y=np.random.randn(10), z=np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(
            axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
        ).fill(z=["T", "F", "T"], x=["T", "F", "T"])

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        exp = unp.exp if a.dtype == np.dtype("O") else np.exp
        return a * exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    h = NamedHist(
        axis.Regular(
            50, -4, 4, name="X", label="s [units]", underflow=False, overflow=False
        )
    ).fill(X=np.random.normal(size=10))


def test_named_access():
    """
        Test named access -- whether NamedHist bins can be accessed.
    """

    h = NamedHist(axis.Regular(10, -5, 5, name="X", label="x [units]")).fill(
        X=np.random.normal(size=1000)
    )

    assert h[6] == h[bh.loc(1)] == h[1j] == h[0j + 1] == h[-3j + 4] == h[bh.loc(1, 0)]
    h[6] = h[bh.loc(1)] = h[1j] = h[0j + 1] = h[-3j + 4] = h[bh.loc(1, 0)] = 0

    h = NamedHist(
        axis.Regular(50, -5, 5, name="Norm", label="normal distribution"),
        axis.Regular(50, -5, 5, name="Unif", label="uniform distribution"),
        axis.StrCategory(["hi", "hello"], name="Greet"),
        axis.Boolean(name="Yes"),
        axis.Integer(0, 1000, name="Int"),
    ).fill(
        Norm=np.random.normal(size=1000),
        Unif=np.random.uniform(size=1000),
        Greet=["hi"] * 800 + ["hello"] * 200,
        Yes=[True] * 600 + [False] * 400,
        Int=np.ones(1000),
    )

    assert h[0j, -0j + 2, "hi", True, 1]

    # mis-match dimension
    with pytest.raises(Exception):
        h[0j, -0j + 2, "hi", True]


class test_named_storage_proxy:
    """
        Test named storage proxy suite -- whether NamedHist storage proxy \
        works properly.
    """

    def test_double(self):
        h = (
            NamedHist()
            .Reg(10, 0, 1, name="x")
            .Reg(10, 0, 1, name="y")
            .Double()
            .fill(x=[0.5, 0.5], y=[0.2, 0.6])
        )

        assert h[0.5j, 0.2j] == 1
        assert h[bh.loc(0.5), bh.loc(0.6)] == 1
        assert isinstance(h[0.5j, 0.5j], int)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Double()

    def test_int64(self):
        h = NamedHist.Reg(10, 0, 1, name="x").Int64().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], float)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Int64()

    def test_automic_int64(self):
        h = NamedHist(axis.Regular(10, 0, 1, name="x")).AutomicInt64().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], int)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.AutomicInt64()

    def test_weight(self):
        h = NamedHist.Reg(10, 0, 1, name="x").Weight().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], float)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Weight()

    def test_mean(self):
        h = NamedHist(axis.Regular(10, 0, 1, name="x")).Mean().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], float)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Mean()

    def test_weighted_mean(self):
        h = NamedHist.Reg(10, 0, 1, name="x").WeightedMean().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], float)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.WeightedMean()

    def test_unlimited(self):
        h = NamedHist(axis.Regular(10, 0, 1, name="x")).Unlimited().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], any)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Unlimited()


def test_named_project():
    """
        Test named project -- whether NamedHist can be projected properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Boolean(name="B", label="b [units]"),
        axis.Variable(range(11), name="C", label="c [units]"),
        axis.Integer(0, 10, name="D", label="d [units]"),
        axis.IntCategory(range(10), name="E", label="e [units]"),
        axis.StrCategory("FT", name="F", label="f [units]"),
    )

    # via names
    assert h.project()
    assert h.project("A", "B")
    assert h.project("A", "B", "C", "D", "E", "F")

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Boolean(name="B", label="b [units]"),
        axis.Variable(range(11), name="C", label="c [units]"),
        axis.Integer(0, 10, name="D", label="d [units]"),
        axis.IntCategory(range(10), name="E", label="e [units]"),
        axis.StrCategory("FT", name="F", label="f [units]"),
    )

    # via indices
    with pytest.raises(Exception):
        h.project(0, 1)
    with pytest.raises(Exception):
        h.project(0, 1, 2, 3, 4, 5)

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

    # disabled params
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, vp_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, fp_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ub_label="value")

    plt.close("all")

    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, pp_label="value")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ub_color="green")

    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_width=1.0)

    # wrong kwargs types
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_ecolor=1.0, eb_mfc=1.0)  # kwargs should be str

    plt.close("all")


def test_named_index_access():
    """
        Test named index access -- whether NamedHist can be accessed by index.
    """

    h = NamedHist(
        axis.Regular(10, -5, 5, name="Ones"),
        axis.Regular(10, -5, 5, name="Twos"),
        axis.StrCategory(["hi", "hello"], name="Greet"),
        axis.Boolean(name="Yes"),
        axis.Integer(0, 10, name="Int"),
    ).fill(
        Ones=np.ones(10),
        Twos=np.ones(10) * 2,
        Greet=["hi"] * 8 + ["hello"] * 2,
        Yes=[True] * 6 + [False] * 4,
        Int=np.ones(10),
    )

    assert h[1j, 2j, "hi", True, 1] == 6
    assert (
        h[
            {
                "Ones": 6,
                "Twos": 7,
                "Greet": bh.loc("hi"),
                "Yes": bh.loc(True),
                "Int": bh.loc(1),
            }
        ]
        == 6
    )
    assert h[0j + 1, -2j + 4, "hi", True, 1] == 6
    assert (
        h[
            {
                "Ones": bh.loc(1, 0),
                "Twos": bh.loc(3, -1),
                "Greet": "hi",
                "Yes": True,
                "Int": 1,
            }
        ]
        == 6
    )

    with pytest.raises(Exception):
        h[0 : bh.loc(1, 0), 1 : bh.loc(3, -1), 2:"hi", 3:True, 4:1] == 6

    with pytest.raises(Exception):
        h[0 : bh.loc(1, 0), 1 : bh.loc(3, -1), "Greet":"hi", 3:True, 4:1] == 6

    assert h[0:10:2j, 0:5:5j, "hello", False, 5]
    assert len(h[::2j, 0:5, :, :, :].axes[1]) == 5
    assert len(h[:, 0:5, :, :, :].axes[1]) == 5

    # wrong loc shortcut
    with pytest.raises(Exception):
        h[0.5, 1 / 2, "hi", True, 1]

    with pytest.raises(Exception):
        h[0.5 + 1j, 1 / 2 + 1j, "hi", True, 1]

    # wrong rebin shortcut
    with pytest.raises(Exception):
        h[0:10:0.2j, 0:5:0.5j, "hello", False, 5]

    with pytest.raises(Exception):
        h[0 : 10 : 1 + 2j, 0 : 5 : 1 + 5j, "hello", False, 5]

    with pytest.raises(Exception):
        h[0:10:20j, 0:5:10j, "hello", False, 5]


def test_named_transform_proxy():
    """
        Test named transform proxy -- whether NamedHist transform proxy works properly.
    """

    h0 = NamedHist().Sqrt(3, 4, 25, name="x").Sqrt(4, 25, 81, name="y")
    h0.fill(x=[5, 10, 17, 17], y=[26, 37, 50, 65])
    assert h0[0, 0] == 1
    assert h0[1, 1] == 1
    assert h0[2, 2] == 1
    assert h0[2, 3] == 1

    # based on existing axis
    with pytest.raises(Exception):
        NamedHist().Regular(3, 4, 25, name="x").Sqrt()

    # wrong value
    with pytest.raises(Exception):
        NamedHist().Sqrt(3, -4, 25, name="x")

    h1 = NamedHist().Log(4, 1, 10_000, name="x").Log(3, 1 / 1_000, 1, name="y")
    h1.fill(x=[2, 11, 101, 1_001], y=[1 / 999, 1 / 99, 1 / 9, 1 / 9])
    assert h1[0, 0] == 1
    assert h1[1, 1] == 1
    assert h1[2, 2] == 1
    assert h1[3, 2] == 1

    # based on existing axis
    with pytest.raises(Exception):
        NamedHist().Regular(4, 1, 10_000, name="x").Log()

    # wrong value
    with pytest.raises(Exception):
        NamedHist().Log(3, -1, 10_000, name="x")

    h2 = NamedHist().Pow(24, 1, 5, power=2, name="x").Pow(124, 1, 5, power=3, name="y")
    h2.fill(x=[1, 2, 3, 4], y=[1, 2, 3, 4])
    assert h2[0, 0] == 1
    assert h2[3, 7] == 1
    assert h2[8, 26] == 1
    assert h2[15, 63] == 1

    # based on existing axis
    with pytest.raises(Exception):
        NamedHist().Regular(24, 1, 5, name="x").Pow(2)

    # wrong value
    with pytest.raises(Exception):
        NamedHist().Pow(24, -1, 5, power=1 / 2, name="x")

    # lack args
    with pytest.raises(Exception):
        NamedHist().Pow(24, 1, 5, name="x")

    ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    h3 = (
        NamedHist()
        .Func(4, 1, 5, forward=ftype(math.log), inverse=ftype(math.exp), name="x")
        .Func(4, 1, 5, forward=ftype(np.log), inverse=ftype(np.exp), name="y")
    )
    h3.fill(x=[1, 2, 3, 4], y=[1, 2, 3, 4])
    assert h3[0, 0] == 1
    assert h3[1, 1] == 1
    assert h3[2, 2] == 1
    assert h3[3, 3] == 1

    # based on existing axis
    with pytest.raises(Exception):
        NamedHist().Regular(24, 1, 5, name="x").Func(ftype(math.log), ftype(math.exp))

    # wrong value
    assert NamedHist().Func(
        4, -1, 5, name="x", forward=ftype(math.log), inverse=ftype(math.log)
    )
    with pytest.raises(Exception):
        NamedHist().Func(
            4, -1, 5, name="x", forward=ftype(np.log), inverse=ftype(np.log)
        )

    # lack args
    with pytest.raises(Exception):
        NamedHist().Func(4, 1, 5, name="x")


def test_named_hist_proxy():
    """
        Test named hist proxy -- whether NamedHist hist proxy works properly.
    """

    h = NamedHist.Reg(10, 0, 1, name="x").fill(x=[0.5, 0.5])
    assert h[0.5j] == 2

    h = (
        NamedHist()
        .Reg(10, 0, 1, name="x")
        .Reg(10, 0, 1, name="y")
        .fill(x=[0.5, 0.5], y=[0.2, 0.6])
    )

    assert h[0.5j, 0.2j] == 1
    assert h[bh.loc(0.5), bh.loc(0.6)] == 1

    h = NamedHist.Bool(name="x").fill(x=[True, True])
    assert h[bh.loc(True)] == 2

    h = NamedHist().Bool(name="x").Bool(name="y").fill(x=[True, True], y=[True, False])

    assert h[True, True] == 1
    assert h[True, False] == 1

    h = NamedHist.Var(range(10), name="x").fill(x=[5, 5])
    assert h[5j] == 2

    h = (
        NamedHist()
        .Var(range(10), name="x")
        .Var(range(10), name="y")
        .fill(x=[5, 5], y=[2, 6])
    )

    assert h[5j, 2j] == 1
    assert h[bh.loc(5), bh.loc(6)] == 1

    h = NamedHist.Int(0, 10, name="x").fill(x=[5, 5])
    assert h[5j] == 2

    h = NamedHist().Int(0, 10, name="x").Int(0, 10, name="y").fill(x=[5, 5], y=[2, 6])

    assert h[5j, 2j] == 1
    assert h[bh.loc(5), bh.loc(6)] == 1

    h = NamedHist.IntCat(range(10), name="x").fill(x=[5, 5])
    assert h[5j] == 2

    h = (
        NamedHist()
        .IntCat(range(10), name="x")
        .IntCat(range(10), name="y")
        .fill(x=[5, 5], y=[2, 6])
    )

    assert h[5j, 2j] == 1
    assert h[bh.loc(5), bh.loc(6)] == 1

    h = NamedHist.StrCat("TF", name="x").fill(x=["T", "T"])
    assert h["T"] == 2

    h = (
        NamedHist()
        .StrCat("TF", name="x")
        .StrCat("TF", name="y")
        .fill(x=["T", "T"], y=["T", "F"])
    )

    assert h["T", "T"] == 1
    assert h["T", "F"] == 1

    # add axes to existing histogram
    with pytest.raises(Exception):
        NamedHist().Reg(10, 0, 1, name="x").fill(x=[0.5, 0.5]).Reg(10, -1, 1, name="y")

    with pytest.raises(Exception):
        NamedHist(axis.Reg(10, 0, 1, name="x")).Reg(10, -1, 1, name="y")

    with pytest.raises(Exception):
        NamedHist().Bool(name="x").fill(x=[True, True]).Bool(name="y")

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="x")).Bool(name="y")

    with pytest.raises(Exception):
        NamedHist().Var(range(0, 1, 10), name="x").fill(x=[0.5, 0.5]).Var(
            range(0, 1, 10), name="y"
        )

    with pytest.raises(Exception):
        NamedHist(axis.Var(range(0, 1, 10), name="x")).Var(range(0, 1, 10), name="y")

    with pytest.raises(Exception):
        NamedHist().Int(0, 10, name="x").fill(x=[0.5, 0.5]).Int(0, 10, name="y")

    with pytest.raises(Exception):
        NamedHist(axis.Int(0, 10, name="x")).Int(0, 10, name="y")

    with pytest.raises(Exception):
        NamedHist().IntCat(range(0, 1, 10), name="x").fill(x=[0.5, 0.5]).IntCat(
            range(0, 1, 10), name="y"
        )

    with pytest.raises(Exception):
        NamedHist(axis.IntCat(range(0, 1, 10), name="x")).IntCat(
            range(0, 1, 10), name="y"
        )

    with pytest.raises(Exception):
        NamedHist().StrCat("TF", name="x").fill(x=["T", "T"]).StrCat("TF", name="y")

    with pytest.raises(Exception):
        NamedHist(axis.StrCat("TF", name="x")).StrCat("TF", name="y")


def test_named_density():
    """
        Test named density -- whether NamedHist density work properly.
    """

    for data in range(10, 20, 10):
        h = NamedHist(axis.Regular(10, -3, 3, name="x")).fill(x=np.random.randn(data))
        assert pytest.approx(sum(h.density()), 2) == pytest.approx(10 / 6, 2)


def test_named_axestuple():
    """
        Test named axes tuple -- whether NamedHist axes tuple work properly.
    """

    h = NamedHist(
        axis.Regular(20, 0, 12, name="A"),
        axis.Regular(10, 1, 3, name="B", label="Beta"),
        axis.Regular(15, 3, 5, name="C"),
        axis.Regular(5, 3, 2, name="D", label="Axis 3"),
    )

    assert h.axes.name == ("A", "B", "C", "D")
    assert h.axes.label == ("A", "Beta", "C", "Axis 3")

    assert h.axes[0].size == 20
    assert h.axes["A"].size == 20

    assert h.axes[1].size == 10
    assert h.axes["B"].size == 10

    assert h.axes[2].size == 15

    assert h.axes[:2].size == (20, 10)
    assert h.axes["A":"B"].size == (20,)
    assert h.axes[:"B"].size == (20,)
    assert h.axes["B":].size == (10, 15, 5)
