# -*- coding: utf-8 -*-
from hist import Hist, axis

import boost_histogram as bh
import pytest
import numpy as np
import ctypes
import math


# TODO: specify what error is raised


def test_init_and_fill(unnamed_hist):
    """
    Test general init -- whether Hist can be properly initialized.
    Also tests filling.
    """
    np.random.seed(42)

    # basic
    h = unnamed_hist(
        axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")
    ).fill([0.35, 0.35, 0.45], [0.35, 0.35, 0.45])

    for idx in range(10):
        if idx == 3:
            assert h[idx, idx] == 2
            assert h[{0: idx, 1: idx}] == 2
            assert h[{"x": idx, "y": idx}] == 2
        elif idx == 4:
            assert h[idx, idx] == 1
            assert h[{0: idx, 1: idx}] == 1
            assert h[{"x": idx, "y": idx}] == 1
        else:
            assert h[idx, idx] == 0
            assert h[{0: idx, 1: idx}] == 0
            assert h[{"x": idx, "y": idx}] == 0

    assert unnamed_hist(
        axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
    ).fill(np.random.randn(10), np.random.randn(10))

    assert unnamed_hist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
        [True, False, True], [True, False, True]
    )

    assert unnamed_hist(
        axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
    ).fill(np.random.randn(10), np.random.randn(10))

    assert unnamed_hist(
        axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")
    ).fill(np.random.randn(10), np.random.randn(10))

    assert unnamed_hist(
        axis.IntCategory(range(-3, 3), name="x"),
        axis.IntCategory(range(-3, 3), name="y"),
    ).fill(np.random.randn(10), np.random.randn(10))

    assert unnamed_hist(
        axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
    ).fill(["T", "F", "T"], ["T", "F", "T"])


def test_no_named_init(unnamed_hist):
    # with no-named axes
    assert unnamed_hist(axis.Regular(50, -3, 3), axis.Regular(50, -3, 3))

    assert unnamed_hist(axis.Boolean(), axis.Boolean())

    assert unnamed_hist(axis.Variable(range(-3, 3)), axis.Variable(range(-3, 3)))

    assert unnamed_hist(axis.Integer(-3, 3), axis.Integer(-3, 3))

    assert unnamed_hist(
        axis.IntCategory(range(-3, 3)),
        axis.IntCategory(range(-3, 3)),
    )

    assert unnamed_hist(axis.StrCategory("TF"), axis.StrCategory(["T", "F"]))


def test_duplicated_names_init(named_hist):
    with pytest.raises(Exception):
        named_hist(axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="x"))

    with pytest.raises(Exception):
        named_hist(axis.Boolean(name="y"), axis.Boolean(name="y"))

    with pytest.raises(Exception):
        named_hist(
            axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="x")
        )

    with pytest.raises(Exception):
        named_hist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="x"))

    with pytest.raises(Exception):
        named_hist(
            axis.IntCategory(range(-3, 3), name="x"),
            axis.IntCategory(range(-3, 3), name="x"),
        )

    with pytest.raises(Exception):
        named_hist(
            axis.StrCategory("TF", name="y"), axis.StrCategory(["T", "F"], name="y")
        )


def test_general_fill():
    """
    Test general fill -- whether Hist can be properly filled.
    """

    # Regular
    h = Hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(10, 0, 1, name="y"),
        axis.Regular(2, 0, 2, name="z"),
    ).fill(
        x=[0.35, 0.35, 0.35, 0.45, 0.55, 0.55, 0.55],
        y=[0.35, 0.35, 0.45, 0.45, 0.45, 0.45, 0.45],
        z=[0, 0, 1, 1, 1, 1, 1],
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
    h = Hist(
        axis.Boolean(name="x"),
        axis.Boolean(name="y"),
        axis.Boolean(name="z"),
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
    h = Hist(
        axis.Variable(range(11), name="x"),
        axis.Variable(range(11), name="y"),
        axis.Variable(range(3), name="z"),
    ).fill(
        x=[3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        y=[3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        z=[0, 0, 1, 1, 1, 1, 1],
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
    h = Hist(
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
    h = Hist(
        axis.IntCategory(range(10), name="x"),
        axis.IntCategory(range(10), name="y"),
        axis.IntCategory(range(2), name="z"),
    ).fill(
        x=[3.5, 3.5, 3.5, 4.5, 5.5, 5.5, 5.5],
        y=[3.5, 3.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        z=[0, 0, 1, 1, 1, 1, 1],
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
    h = Hist(
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
    assert Hist(
        axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert Hist(axis.Boolean(name="x"), axis.Boolean(name="y")).fill(
        x=[True, False, True], y=[True, False, True]
    )

    assert Hist(
        axis.Variable(range(-3, 3), name="x"), axis.Variable(range(-3, 3), name="y")
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert Hist(axis.Integer(-3, 3, name="x"), axis.Integer(-3, 3, name="y")).fill(
        x=np.random.randn(10), y=np.random.randn(10)
    )

    assert Hist(
        axis.IntCategory(range(-3, 3), name="x"),
        axis.IntCategory(range(-3, 3), name="y"),
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert Hist(
        axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
    ).fill(x=["T", "F", "T"], y=["T", "F", "T"])

    h = Hist(
        axis.Regular(
            50, -4, 4, name="X", label="s [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=10))


def test_general_access():
    """
    Test general access -- whether Hist bins can be accessed.
    """

    h = Hist(axis.Regular(10, -5, 5, name="X", label="x [units]")).fill(
        np.random.normal(size=1000)
    )

    assert h[6] == h[bh.loc(1)] == h[1j] == h[0j + 1] == h[-3j + 4] == h[bh.loc(1, 0)]
    h[6] = h[bh.loc(1)] = h[1j] = h[0j + 1] = h[-3j + 4] = h[bh.loc(1, 0)] = 0

    h = Hist(
        axis.Regular(50, -5, 5, name="Norm", label="normal distribution"),
        axis.Regular(50, -5, 5, name="Unif", label="uniform distribution"),
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

    # mis-match dimension
    with pytest.raises(Exception):
        h[0j, -0j + 2, "hi", True]


def test_general_project():
    """
    Test general project -- whether Hist can be projected properly.
    """
    h = Hist(
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
    assert h.project()
    assert h.project(0, 1)
    assert h.project(0, 1, 2, 3, 4, 5)

    # via names
    assert h.project()
    assert h.project("A", "B")
    assert h.project("A", "B", "C", "D", "E", "F")

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Boolean(name="B", label="b [units]"),
        axis.Variable(range(11), name="C", label="c [units]"),
        axis.Integer(0, 10, name="D", label="d [units]"),
        axis.IntCategory(range(10), name="E", label="e [units]"),
        axis.StrCategory("FT", name="F", label="f [units]"),
    )

    # duplicated
    with pytest.raises(Exception):
        h.project(0, 0)

    with pytest.raises(Exception):
        h.project("A", "A")

    with pytest.raises(Exception):
        h.project(0, "A")

    # mixed types
    assert h.project(2, "A")

    # cannot found
    with pytest.raises(Exception):
        h.project(-1, 9)

    with pytest.raises(Exception):
        h.project("G", "H")


def test_general_index_access():
    """
    Test general index access -- whether Hist can be accessed by index.
    """

    h = Hist(
        axis.Regular(10, -5, 5, name="Ones"),
        axis.Regular(10, -5, 5, name="Twos"),
        axis.StrCategory(["hi", "hello"], name="Greet"),
        axis.Boolean(name="Yes"),
        axis.Integer(0, 10, name="Int"),
    ).fill(
        np.ones(10),
        np.ones(10) * 2,
        ["hi"] * 8 + ["hello"] * 2,
        [True] * 6 + [False] * 4,
        np.ones(10),
    )

    assert h[1j, 2j, "hi", True, 1] == 6
    assert h[{0: 6, 1: 7, 2: bh.loc("hi"), 3: bh.loc(True), 4: bh.loc(1)}] == 6
    assert h[0j + 1, -2j + 4, "hi", True, 1] == 6
    assert (
        h[{"Greet": "hi", "Ones": bh.loc(1, 0), 1: bh.loc(3, -1), 3: True, "Int": 1}]
        == 6
    )

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


class TestGeneralStorageProxy:
    """
        Test general storage proxy suite -- whether Hist storage proxy \
        works properly.
    """

    def test_double(self):
        h = (
            Hist.new.Reg(10, 0, 1, name="x")
            .Reg(10, 0, 1, name="y")
            .Double()
            .fill(x=[0.5, 0.5], y=[0.2, 0.6])
        )

        assert h[0.5j, 0.2j] == 1
        assert h[bh.loc(0.5), bh.loc(0.6)] == 1
        assert isinstance(h[0.5j, 0.5j], float)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Double()

    def test_int64(self):
        h = Hist.new.Reg(10, 0, 1, name="x").Int64().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], int)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Int64()

    def test_atomic_int64(self):
        h = Hist.new.Reg(10, 0, 1, name="x").AtomicInt64().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], int)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.AtomicInt64()

    def test_weight(self):
        h = Hist.new.Reg(10, 0, 1, name="x").Weight().fill([0.5, 0.5])
        assert h[0.5j].variance == 2
        assert h[0.5j].value == 2

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Weight()

    def test_mean(self):
        h = (
            Hist.new.Reg(10, 0, 1, name="x")
            .Mean()
            .fill([0.5, 0.5], weight=[1, 1], sample=[1, 1])
        )
        assert h[0.5j].count == 2
        assert h[0.5j].value == 1
        assert h[0.5j].variance == 0

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Mean()

    def test_weighted_mean(self):
        h = (
            Hist.new.Reg(10, 0, 1, name="x")
            .WeightedMean()
            .fill([0.5, 0.5], weight=[1, 1], sample=[1, 1])
        )
        assert h[0.5j].sum_of_weights == 2
        assert h[0.5j].sum_of_weights_squared == 2
        assert h[0.5j].value == 1
        assert h[0.5j].variance == 0

        # add storage to existing storage
        with pytest.raises(Exception):
            h.WeightedMean()

    def test_unlimited(self):
        h = Hist.new.Reg(10, 0, 1, name="x").Unlimited().fill([0.5, 0.5])
        assert h[0.5j] == 2

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Unlimited()


def test_general_transform_proxy():
    """
    Test general transform proxy -- whether Hist transform proxy works properly.
    """

    h0 = Hist.new.Sqrt(3, 4, 25).Sqrt(4, 25, 81).Double()
    h0.fill([5, 10, 17, 17], [26, 37, 50, 65])
    assert h0[0, 0] == 1
    assert h0[1, 1] == 1
    assert h0[2, 2] == 1
    assert h0[2, 3] == 1

    # wrong value
    with pytest.raises(Exception):
        Hist.new.Sqrt(3, -4, 25)

    h1 = Hist.new.Log(4, 1, 10_000).Log(3, 1 / 1_000, 1).Double()
    h1.fill([2, 11, 101, 1_001], [1 / 999, 1 / 99, 1 / 9, 1 / 9])
    assert h1[0, 0] == 1
    assert h1[1, 1] == 1
    assert h1[2, 2] == 1
    assert h1[3, 2] == 1

    # Missing arguments
    with pytest.raises(Exception):
        Hist.new.Regular(4, 1, 10_000).Log()

    # wrong value
    with pytest.raises(Exception):
        Hist.new.Log(3, -1, 10_000)

    h2 = Hist.new.Pow(24, 1, 5, power=2).Pow(124, 1, 5, power=3).Int64()
    h2.fill([1, 2, 3, 4], [1, 2, 3, 4])
    assert h2[0, 0] == 1
    assert h2[3, 7] == 1
    assert h2[8, 26] == 1
    assert h2[15, 63] == 1

    # wrong value
    with pytest.raises(Exception):
        Hist.new.Regular(24, 1, 5).Pow(2)

    # wrong value
    with pytest.raises(Exception):
        Hist.new.Pow(24, -1, 5, power=0.5)

    # lack args
    with pytest.raises(Exception):
        Hist.new.Pow(24, 1, 5)

    ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    h3 = (
        Hist.new.Func(4, 1, 5, forward=ftype(math.log), inverse=ftype(math.exp)).Func(
            4, 1, 5, forward=ftype(np.log), inverse=ftype(np.exp)
        )
    ).Int64()
    h3.fill([1, 2, 3, 4], [1, 2, 3, 4])
    assert h3[0, 0] == 1
    assert h3[1, 1] == 1
    assert h3[2, 2] == 1
    assert h3[3, 3] == 1

    # wrong value
    with pytest.raises(Exception):
        Hist().Regular(24, 1, 5).Func(ftype(math.log), ftype(math.exp))

    # wrong value
    assert Hist.new.Func(
        4, -1, 5, forward=ftype(math.log), inverse=ftype(math.log)
    ).Double()
    with pytest.raises(Exception):
        Hist.new.Func(4, -1, 5, forward=ftype(np.log), inverse=ftype(np.log))

    # lack args
    with pytest.raises(Exception):
        Hist.new.Func(4, 1, 5)


def test_hist_proxy_matches(named_hist):
    h = named_hist.new.Reg(10, 0, 1, name="x").Double()
    assert type(h) == named_hist


def test_hist_proxy():
    """
    Test general hist proxy -- whether Hist hist proxy works properly.
    """

    h = Hist.new.Reg(10, 0, 1, name="x").Double().fill([0.5, 0.5])
    assert h[0.5j] == 2

    assert type(h) == Hist

    with pytest.raises(AttributeError):
        Hist().new

    h = (
        Hist.new.Reg(10, 0, 1, name="x")
        .Reg(10, 0, 1, name="y")
        .Double()
        .fill([0.5, 0.5], [0.2, 0.6])
    )

    assert h[0.5j, 0.2j] == 1
    assert h[bh.loc(0.5), bh.loc(0.6)] == 1

    h = Hist.new.Bool(name="x").Double().fill([True, True])
    assert h[bh.loc(True)] == 2

    h = Hist.new.Bool(name="x").Bool(name="y").Int64().fill([True, True], [True, False])

    assert h[True, True] == 1
    assert h[True, False] == 1

    h = Hist.new.Var(range(10), name="x").Double().fill([5, 5])
    assert h[5j] == 2

    h = (
        Hist.new.Var(range(10), name="x")
        .Var(range(10), name="y")
        .Double()
        .fill([5, 5], [2, 6])
    )

    assert h[5j, 2j] == 1
    assert h[bh.loc(5), bh.loc(6)] == 1

    h = Hist.new.Int(0, 10, name="x").Int64().fill([5, 5])
    assert h[5j] == 2

    h = Hist.new.Int(0, 10, name="x").Int(0, 10, name="y").Int64().fill([5, 5], [2, 6])

    assert h[5j, 2j] == 1
    assert h[bh.loc(5), bh.loc(6)] == 1

    h = Hist.new.IntCat(range(10), name="x").Double().fill([5, 5])
    assert h[5j] == 2

    h = (
        Hist.new.IntCat(range(10), name="x")
        .IntCat(range(10), name="y")
        .Double()
        .fill([5, 5], [2, 6])
    )

    assert h[5j, 2j] == 1
    assert h[bh.loc(5), bh.loc(6)] == 1

    h = Hist.new.StrCat("TF", name="x").Int64().fill(["T", "T"])
    assert h["T"] == 2

    h = (
        Hist.new.StrCat("TF", name="x")
        .StrCat("TF", name="y")
        .Int64()
        .fill(["T", "T"], ["T", "F"])
    )

    assert h["T", "T"] == 1
    assert h["T", "F"] == 1


def test_general_density():
    """
    Test general density -- whether Hist density work properly.
    """

    for data in range(10, 20, 10):
        h = Hist(axis.Regular(10, -3, 3, name="x")).fill(np.random.randn(data))
        assert pytest.approx(sum(h.density()), 2) == pytest.approx(10 / 6, 2)


def test_general_axestuple():
    """
    Test general axes tuple -- whether Hist axes tuple work properly.
    """

    h = Hist(
        axis.Regular(20, 0, 12, name="A", label="alpha"),
        axis.Regular(10, 1, 3, name="B"),
        axis.Regular(15, 3, 5, label="other"),
        axis.Regular(5, 3, 2),
    )

    assert h.axes.name == ("A", "B", "", "")
    assert h.axes.label == ("alpha", "B", "other", "Axis 3")

    assert h.axes[0].size == 20
    assert h.axes["A"].size == 20

    assert h.axes[1].size == 10
    assert h.axes["B"].size == 10

    assert h.axes[2].size == 15

    assert h.axes[:2].size == (20, 10)
    assert h.axes["A":"B"].size == (20,)
    assert h.axes[:"B"].size == (20,)
    assert h.axes["B":].size == (10, 15, 5)
