from hist import axis, NamedHist
import boost_histogram as bh
import pytest
import numpy as np


def test_basic_usage():
    """
        Test basic usage -- whether NamedHist are properly derived from\
        boost-histogram and whether it can be filled by names.
    """

    # Basic
    h = NamedHist(axis.Regular(10, 0, 1, name="x")).fill(x=[0.35, 0.35, 0.45])

    for idx in range(10):
        if idx == 3:
            assert h[idx] == h[{0: idx}] == h[{"x": idx}] == 2
        elif idx == 4:
            assert h[idx] == h[{0: idx}] == h[{"x": idx}] == 1
        else:
            assert h[idx] == h[{0: idx}] == h[{"x": idx}] == 0

    # Regular
    h = h = NamedHist(
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

    # Bool
    h = NamedHist(axis.Bool(name="x"), axis.Bool(name="y"), axis.Bool(name="z"),).fill(
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


def test_errors():

    # right axis names
    assert NamedHist(axis.Regular(50, -3, 3, name="x0"))
    assert NamedHist(axis.Bool(name="x_"))
    assert NamedHist(axis.Variable(range(-3, 3), name="xx"))
    assert NamedHist(axis.Integer(-3, 3, name="x_x"))
    assert NamedHist(axis.IntCategory(range(-3, 3), name="X__X"))
    assert NamedHist(axis.StrCategory("FT", name="X00"))

    # wrong axis names: protected or private prefix
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3, name="_x"))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="__x"))

    with pytest.raises(Exception):
        NamedHist(axis.Variable(range(-3, 3), name="_x_"))

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name="_x0"))

    with pytest.raises(Exception):
        NamedHist(axis.IntCategory(range(-3, 3), name="_0x"))

    with pytest.raises(Exception):
        NamedHist(axis.StrCategory("FT", name="_xX"))

    # wrong axis names: number prefix
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3, name="0"))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="00"))

    with pytest.raises(Exception):
        NamedHist(axis.Variable(range(-3, 3), name="0x"))

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name="0_x"))

    with pytest.raises(Exception):
        NamedHist(axis.IntCategory(range(-3, 3), name="00x"))

    with pytest.raises(Exception):
        NamedHist(axis.StrCategory("FT", name="0xx"))

    # wrong axis names: unsupported chr
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3, name="-x"))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="x-"))

    with pytest.raises(Exception):
        NamedHist(axis.Variable(range(-3, 3), name="?x"))

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name="%x"))

    with pytest.raises(Exception):
        NamedHist(axis.IntCategory(range(-3, 3), name="#x"))

    with pytest.raises(Exception):
        NamedHist(axis.StrCategory("FT", name="*x"))

    # wrong axis names: empty names
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3, name=""))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name=""))

    with pytest.raises(Exception):
        NamedHist(axis.Variable(range(-3, 3)))

    with pytest.raises(Exception):
        NamedHist(axis.Integer(-3, 3, name=""))

    with pytest.raises(Exception):
        NamedHist(axis.IntCategory(range(-3, 3), name=""))

    with pytest.raises(Exception):
        NamedHist(axis.StrCategory("FT"))

    # right histogram axis names
    assert NamedHist(
        axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
    ).fill(x=np.random.randn(10), y=np.random.randn(10))

    assert NamedHist(axis.Bool(name="x"), axis.Bool(name="y")).fill(
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

    # wrong histogram axis names: with the same names
    with pytest.raises(Exception):
        NamedHist(axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="x"))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="y"), axis.Bool(name="y"))

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

    # wrong histogram axis names: fill without names
    with pytest.raises(Exception):
        NamedHist(
            axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
        ).fill(np.random.randn(10), np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="x"), axis.Bool(name="y")).fill(
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

    # wrong histogram axis names: fill by wrong names
    with pytest.raises(Exception):
        NamedHist(
            axis.Regular(50, -3, 3, name="x"), axis.Regular(50, -3, 3, name="y")
        ).fill(x=np.random.randn(10), z=np.random.randn(10))

    with pytest.raises(Exception):
        NamedHist(axis.Bool(name="x"), axis.Bool(name="y")).fill(
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
