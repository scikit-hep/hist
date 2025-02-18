from __future__ import annotations

import ctypes
import math

import boost_histogram as bh
import numpy as np
import pytest
from pytest import approx

import hist
from hist import Hist, axis, storage

BHV = tuple(int(x) for x in bh.__version__.split(".")[:2])

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
    ).fill(np.random.randint(-3, 3, 10), np.random.randint(-3, 3, 10))

    assert unnamed_hist(
        axis.IntCategory(range(-3, 3), name="x"),
        axis.IntCategory(range(-3, 3), name="y"),
    ).fill(np.random.randint(-3, 3, 10), np.random.randint(-3, 3, 10))

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


def test_general_fill_regular():
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
    for idx_x in range(10):
        for idx_y in range(10):
            if (idx_x == 3 and idx_y == 4) or (idx_x == 4 and idx_y == 4):
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0


def test_general_fill_bool():
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


def test_general_fill_variable():
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
    for idx_x in range(10):
        for idx_y in range(10):
            if (idx_x == 3 and idx_y == 4) or (idx_x == 4 and idx_y == 4):
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0


def test_general_fill_integer():
    h = Hist(
        axis.Integer(0, 10, name="x"),
        axis.Integer(0, 10, name="y"),
        axis.Integer(0, 2, name="z"),
    ).fill(
        [3, 3, 3, 4, 5, 5, 5],
        [3, 3, 4, 4, 4, 4, 4],
        [0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{2: bh.loc(1)}]
    for idx_x in range(10):
        for idx_y in range(10):
            if (idx_x == 3 and idx_y == 4) or (idx_x == 4 and idx_y == 4):
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0


def test_general_fill_int_cat():
    h = Hist(
        axis.IntCategory(range(10), name="x", flow=BHV < (1, 4)),
        axis.IntCategory(range(10), name="y", overflow=BHV < (1, 4)),
        axis.IntCategory(range(2), name="z"),
    ).fill(
        x=[3, 3, 3, 4, 5, 5, 5],
        y=[3, 3, 4, 4, 4, 4, 4],
        z=[0, 0, 1, 1, 1, 1, 1],
    )

    if BHV < (1, 4):
        assert h.axes[0].traits.overflow
        assert h.axes[1].traits.overflow
    else:
        assert not h.axes[0].traits.overflow
        assert not h.axes[1].traits.overflow
    assert h.axes[2].traits.overflow

    z_one_only = h[{2: bh.loc(1)}]
    for idx_x in range(10):
        for idx_y in range(10):
            if (idx_x == 3 and idx_y == 4) or (idx_x == 4 and idx_y == 4):
                assert z_one_only[idx_x, idx_y] == 1
            elif idx_x == 5 and idx_y == 4:
                assert z_one_only[idx_x, idx_y] == 3
            else:
                assert z_one_only[idx_x, idx_y] == 0


def test_general_fill_str_cat():
    h = Hist(
        axis.StrCategory("FT", name="x", flow=BHV < (1, 4)),
        axis.StrCategory(list("FT"), name="y", overflow=BHV < (1, 4)),
        axis.StrCategory(["F", "T"], name="z"),
    ).fill(
        ["T", "T", "T", "T", "T", "F", "T"],
        ["F", "T", "T", "F", "F", "T", "F"],
        ["F", "F", "T", "T", "T", "T", "T"],
    )

    if BHV < (1, 4):
        assert h.axes[0].traits.overflow
        assert h.axes[1].traits.overflow
    else:
        assert not h.axes[0].traits.overflow
        assert not h.axes[1].traits.overflow
    assert h.axes[2].traits.overflow

    z_one_only = h[{2: bh.loc("T")}]
    assert z_one_only[bh.loc("F"), bh.loc("F")] == 0
    assert z_one_only[bh.loc("F"), bh.loc("T")] == 1
    assert z_one_only[bh.loc("T"), bh.loc("F")] == 3
    assert z_one_only[bh.loc("T"), bh.loc("T")] == 1


def test_general_fill_names():
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
        x=np.random.randint(-3, 3, 10), y=np.random.randint(-3, 3, 10)
    )

    assert Hist(
        axis.IntCategory(range(-3, 3), name="x"),
        axis.IntCategory(range(-3, 3), name="y"),
    ).fill(x=np.random.randint(-3, 3, 10), y=np.random.randint(-3, 3, 10))

    assert Hist(
        axis.StrCategory(["F", "T"], name="x"), axis.StrCategory("FT", name="y")
    ).fill(x=["T", "F", "T"], y=["T", "F", "T"])

    Hist(
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
        axis.StrCategory(["hi", "hello"], name="Greet", flow=BHV < (1, 4)),
        axis.Boolean(name="Yes"),
        axis.Integer(0, 1000, name="Int"),
    ).fill(
        np.random.normal(size=1000),
        np.random.uniform(size=1000),
        ["hi"] * 800 + ["hello"] * 200,
        [True] * 600 + [False] * 400,
        np.ones(1000, dtype=int),
    )

    assert h[0j, -0j + 2, "hi", True, 1]

    # mismatch dimension
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
        np.ones(10, dtype=int),
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

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "double").storage_type
            == storage.Double
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="DouBle").storage_type
            == storage.Double
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.Double()).storage_type
            == storage.Double
        )

    def test_int64(self):
        h = Hist.new.Reg(10, 0, 1, name="x").Int64().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], int)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Int64()

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "int64").storage_type
            == storage.Int64
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="INT64").storage_type
            == storage.Int64
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.Int64()).storage_type
            == storage.Int64
        )

    def test_atomic_int64(self):
        h = Hist.new.Reg(10, 0, 1, name="x").AtomicInt64().fill([0.5, 0.5])
        assert h[0.5j] == 2
        assert isinstance(h[0.5j], int)

        # add storage to existing storage
        with pytest.raises(Exception):
            h.AtomicInt64()

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "atomicint64").storage_type
            == storage.AtomicInt64
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="AtomicINT64").storage_type
            == storage.AtomicInt64
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.AtomicInt64()).storage_type
            == storage.AtomicInt64
        )

    def test_weight(self):
        h = Hist.new.Reg(10, 0, 1, name="x").Weight().fill([0.5, 0.5])
        assert h[0.5j].variance == 2
        assert h[0.5j].value == 2

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Weight()

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "WeighT").storage_type
            == storage.Weight
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="weight").storage_type
            == storage.Weight
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.Weight()).storage_type
            == storage.Weight
        )

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

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "MEAn").storage_type == storage.Mean
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="mean").storage_type
            == storage.Mean
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.Mean()).storage_type
            == storage.Mean
        )

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

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "WeighTEDMEAn").storage_type
            == storage.WeightedMean
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="weightedMean").storage_type
            == storage.WeightedMean
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.WeightedMean()).storage_type
            == storage.WeightedMean
        )

    def test_unlimited(self):
        h = Hist.new.Reg(10, 0, 1, name="x").Unlimited().fill([0.5, 0.5])
        assert h[0.5j] == 2

        # add storage to existing storage
        with pytest.raises(Exception):
            h.Unlimited()

        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), "unlimited").storage_type
            == storage.Unlimited
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage="UNLImited").storage_type
            == storage.Unlimited
        )
        assert (
            Hist(axis.Regular(10, 0, 1, name="x"), storage.Unlimited()).storage_type
            == storage.Unlimited
        )


def test_quick_construct_kwargs():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    h = Hist.new.Regular(10, 0, 1, name="x").Double(name="h", label="y", data=data)
    assert h.name == "h"
    assert h.label == "y"
    assert h.values() == approx(data)


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
    with pytest.raises(TypeError):
        Hist.new.Reg(4, 1, 10_000).Log()

    # Wrong value
    with pytest.raises(ValueError):
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

    # wrong args
    with pytest.raises(TypeError):
        Hist.new.Reg(24, 1, 5).Func(ftype(math.log), ftype(math.exp))

    # wrong value raises uncatchable warning
    # Hist.new.Func(
    #     4, -1, 5, forward=ftype(math.log), inverse=ftype(math.log)
    # ).Double()

    with pytest.raises(ValueError), pytest.warns(RuntimeWarning):
        Hist.new.Func(4, -1, 5, forward=ftype(np.log), inverse=ftype(np.log))

    # lack args
    with pytest.raises(TypeError):
        Hist.new.Func(4, 1, 5)


def test_hist_proxy_matches(named_hist):
    h = named_hist.new.Reg(10, 0, 1, name="x").Double()
    assert type(h) is named_hist


def test_hist_proxy():
    """
    Test general hist proxy -- whether Hist hist proxy works properly.
    """

    h = Hist.new.Reg(10, 0, 1, name="x").Double().fill([0.5, 0.5])
    assert h[0.5j] == 2

    assert type(h) is Hist

    assert not hasattr(Hist(), "new")

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


def test_hist_proxy_mistake():
    h = Hist(Hist.new.IntCat(range(10)))
    h2 = Hist.new.IntCategory(range(10)).Double()

    assert h == h2


def test_general_density():
    """
    Test general density -- whether Hist density work properly.
    """

    for data in range(10, 20, 10):
        h = Hist(axis.Regular(10, -3, 3, name="x")).fill(np.random.randn(data))
        assert pytest.approx(sum(h.density()), 2) == pytest.approx(10 / 6, 2)


def test_weighted_density():
    for data in range(10, 20, 10):
        h = Hist(axis.Regular(10, -3, 3, name="x"), storage="weight").fill(
            np.random.randn(data)
        )
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


def test_from_columns(named_hist):
    columns = {
        "x": [1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 1, 1],
        "y": ["a", "b", "c", "d"] * 3,
        "data": np.arange(12),
    }

    h = named_hist.from_columns(columns, ("x", "y"))
    assert h.values() == approx(np.array([[3, 0, 2, 1], [0, 2, 0, 2], [0, 1, 1, 0]]))

    h_w = named_hist.from_columns(columns, ("x", "y"), weight="data")
    assert h_w.values() == approx(
        np.array([[12, 0, 16, 11], [0, 6, 0, 10], [0, 9, 2, 0]])
    )

    h_w2 = named_hist.from_columns(
        columns, (axis.Integer(1, 5, name="x"), "y"), weight="data"
    )
    assert h_w2.values() == approx(
        np.array([[12, 0, 16, 11], [0, 6, 0, 10], [0, 9, 2, 0], [0, 0, 0, 0]])
    )

    with pytest.raises(TypeError):
        named_hist.from_columns(columns, (axis.Integer(1, 5), "y"), weight="data")


def test_from_array(named_hist):
    h = named_hist(
        axis.Regular(10, 1, 2, name="A"),
        axis.Regular(7, 1, 3, name="B"),
        data=np.ones((10, 7)),
    )
    assert h.values() == approx(np.ones((10, 7)))
    assert h.sum() == approx(70)
    assert h.sum(flow=True) == approx(70)

    h = named_hist(
        axis.Regular(10, 1, 2, name="A"),
        axis.Regular(7, 1, 3, name="B"),
        data=np.ones((12, 9)),
    )

    assert h.values(flow=False) == approx(np.ones((10, 7)))
    assert h.values(flow=True) == approx(np.ones((12, 9)))
    assert h.sum() == approx(70)
    assert h.sum(flow=True) == approx(12 * 9)

    with pytest.raises(ValueError):
        h = named_hist(
            axis.Regular(10, 1, 2, name="A"),
            axis.Regular(7, 1, 3, name="B"),
            data=np.ones((11, 9)),
        )


def test_sum_empty_axis():
    hist = bh.Histogram(
        bh.axis.StrCategory("", growth=True),
        bh.axis.Regular(10, 0, 1),
        storage=bh.storage.Weight(),
    )
    assert hist.sum().value == 0
    assert "Str" in repr(hist)


def test_sum_empty_axis_hist():
    h = Hist(
        axis.StrCategory("", growth=True),
        axis.Regular(10, 0, 1),
        storage=storage.Weight(),
    )
    assert h.sum().value == 0
    assert "Str" in repr(h)
    h._repr_html_()


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_select_by_index():
    h = Hist(
        axis.StrCategory(["a", "two", "3"]),
        storage=storage.Weight(),
    )

    assert tuple(h[["a", "3"]].axes[0]) == ("a", "3")
    assert tuple(h[["a"]].axes[0]) == ("a",)


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_select_by_index_imag():
    h = Hist(
        axis.IntCategory([7, 8, 9]),
        storage=storage.Int64(),
    )

    assert tuple(h[[2, 1]].axes[0]) == (9, 8)
    assert tuple(h[[8j, 7j]].axes[0]) == (8, 7)


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_select_by_index_wildcards():
    h = hist.new.Reg(10, 0, 10).StrCat(["ABC", "BCD", "CDE", "DEF"]).Weight()
    assert tuple(h[:, "*E*"].axes[1]) == ("CDE", "DEF")
    assert tuple(h[:, ["*B*", "CDE"]].axes[1]) == ("ABC", "BCD", "CDE")
    assert tuple(h[:, ["*B*", "?D?"]].axes[1]) == ("ABC", "BCD", "CDE")


def test_sorted_simple():
    h = Hist.new.IntCat([4, 1, 2]).StrCat(["AB", "BCC", "BC"]).Double()
    assert tuple(h.sort(0).axes[0]) == (1, 2, 4)
    assert tuple(h.sort(0, reverse=True).axes[0]) == (4, 2, 1)
    assert tuple(h.sort(0, key=lambda x: -x).axes[0]) == (4, 2, 1)
    assert tuple(h.sort(1).axes[1]) == ("AB", "BC", "BCC")
    assert tuple(h.sort(1, reverse=True).axes[1]) == ("BCC", "BC", "AB")


def test_quick_construct_direct():
    h = hist.new.IntCat([4, 1, 2]).StrCat(["AB", "BCC", "BC"]).Double()
    assert tuple(h.sort(0).axes[0]) == (1, 2, 4)
    assert tuple(h.sort(0, reverse=True).axes[0]) == (4, 2, 1)
    assert tuple(h.sort(0, key=lambda x: -x).axes[0]) == (4, 2, 1)
    assert tuple(h.sort(1).axes[1]) == ("AB", "BC", "BCC")
    assert tuple(h.sort(1, reverse=True).axes[1]) == ("BCC", "BC", "AB")


def test_integrate():
    h = (
        hist.new.IntCat([4, 1, 2], name="x")
        .StrCat(["AB", "BCC", "BC"], name="y")
        .Int(1, 10, name="z")
        .Int64()
    )
    h.fill(4, "AB", 1)
    h.fill(4, "BCC", 2)
    h.fill(4, "BC", 4)
    h.fill(4, "X", 8)

    h.fill(2, "aAB", 3)
    h.fill(2, "BCC", 5)
    h.fill(2, "AB", 2)
    h.fill(2, "X", 1)

    h.fill(1, "AB", 3)
    h.fill(1, "BCC", 1)
    h.fill(1, "BC", 5)
    h.fill(1, "X", 2)

    h1 = h.integrate("y", ["AB", "BC"]).integrate("z")
    h2 = h.integrate("y", ["AB", "BC", "BCC"]).integrate("z")

    assert h1[{"x": 4j}] == 2
    assert h1[{"x": 2j}] == 1
    assert h2[{"x": 1j}] == 3


def test_T_property():
    # Create a 2D histogram with some data
    hist_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    h = hist.Hist(
        hist.axis.Regular(3, 0, 1, flow=False),
        hist.axis.Regular(3, 5, 6, flow=False),
        data=hist_data,
    )

    assert h.T.values() == approx(h.values().T)
    assert h.T.axes[0] == h.axes[1]
    assert h.T.axes[1] == h.axes[0]


def test_T_empty():
    hist_empty = hist.Hist()
    hist_T_empty = hist_empty.T
    assert hist_empty == hist_T_empty


def test_T_1D():
    # Create a 1D histogram with some data
    hist_data_1D = np.array([1, 2, 3, 4, 5])
    h_1D = hist.Hist(hist.axis.Regular(5, 0, 1, flow=False), data=hist_data_1D)

    assert h_1D.T.values() == approx(h_1D.values().T)
    assert h_1D.T.axes[0] == h_1D.axes[0]


def test_T_3D():
    # Create a 3D histogram with some data
    hist_data_3D = np.random.rand(1, 3, 4)
    h_3D = hist.Hist(
        hist.axis.Regular(2, 0, 1, flow=False),
        hist.axis.Regular(3, 2, 3, flow=False),
        hist.axis.Regular(4, 5, 6, flow=False),
        data=hist_data_3D,
    )

    assert h_3D.T.values() == approx(h_3D.values().T)
    assert h_3D.T.axes[0] == h_3D.axes[2]
    assert h_3D.T.axes[1] == h_3D.axes[1]
    assert h_3D.T.axes[2] == h_3D.axes[0]
