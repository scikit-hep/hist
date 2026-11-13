from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import hist
from hist import axis


def test_structured(unnamed_hist):  # basic
    h = unnamed_hist(
        axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")
    ).fill_flattened(
        np.array([[0.1, 0.2], [0.5, 0.3], [0.8, 0.2], [0.5, 0.5]]).astype(
            [("x", np.float64), ("y", np.float64)]
        )
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(h.values(), expected)


def test_positional(unnamed_hist):
    x = np.array([[0.1], [0.2], [0.3], [0.4], [0.6], [0.7]], dtype=np.float64)
    y = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.9, 0.8], [0.6, 0.7], [0.4, 0.5]],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    h = unnamed_hist(
        axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")
    ).fill_flattened(x, y)
    assert np.allclose(h.values(), expected)


def test_positional_keyword(unnamed_hist):
    x = np.array([[0.1], [0.2], [0.3], [0.4], [0.6], [0.7]], dtype=np.float64)
    y = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.9, 0.8], [0.6, 0.7], [0.4, 0.5]],
        dtype=np.float64,
    )
    z = np.array([0.8])
    expected = np.array(
        [
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    h = unnamed_hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(10, 0, 1, name="y"),
        axis.Regular(2, 0, 1, name="z"),
    ).fill_flattened(x, y, z=z)
    assert np.allclose(h.values(), expected)


def test_named_only_structured():  # basic
    h = hist.NamedHist(
        axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")
    ).fill_flattened(
        np.array([[0.1, 0.2], [0.5, 0.3], [0.8, 0.2], [0.5, 0.5]]).astype(
            [("x", np.float64), ("y", np.float64)]
        )
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(h.values(), expected)


def test_named_only_positional():
    x = np.array([[0.1], [0.2], [0.3], [0.4], [0.6], [0.7]], dtype=np.float64)
    with pytest.raises(TypeError, match="could not be destructed"):
        hist.NamedHist(axis.Regular(10, 0, 1, name="x")).fill_flattened(x)


def test_named_only_structured_keyword():
    xy = np.array([[0.1, 0.2], [0.5, 0.3], [0.8, 0.2], [0.5, 0.5]]).astype(
        [("x", np.float64), ("y", np.float64)]
    )
    z = np.array([0.8])
    with pytest.raises(TypeError, match="but not both"):
        hist.NamedHist(
            axis.Regular(10, 0, 1, name="x"),
            axis.Regular(10, 0, 1, name="y"),
            axis.Regular(2, 0, 1, name="z"),
        ).fill_flattened(xy, z=z)


def test_named_keyword(unnamed_hist):
    x = np.array([[0.1], [0.2], [0.3], [0.4], [0.6], [0.7]], dtype=np.float64)
    y = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.9, 0.8], [0.6, 0.7], [0.4, 0.5]],
        dtype=np.float64,
    )
    z = np.array([0.8])
    expected = np.array(
        [
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    h = unnamed_hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(10, 0, 1, name="y"),
        axis.Regular(2, 0, 1, name="z"),
    ).fill_flattened(x=x, y=y, z=z)
    assert np.allclose(h.values(), expected)


def test_string_fill_flattened():
    ark = np.array([1, 2, 3, 4, 5])
    h = hist.new.Reg(10, 0, 10, name="x").StrCat([], growth=True, name="cat").Weight()
    h.fill_flattened(x=ark, cat="A")


def test_ak_fill_flattened():
    ak = pytest.importorskip("awkward")

    ark = ak.Array([[1, 2, 3], [4, 5, 6], [9]])
    h = hist.new.Reg(10, 0, 10, name="x").StrCat([], growth=True, name="cat").Weight()
    h.fill_flattened(x=ark, cat="A")


def test_pandas_fill_flattened(named_hist):
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({"x": [0.1, 0.5, 0.5], "y": [0.2, 0.2, 0.8]})
    h = named_hist(
        axis.Regular(2, 0, 1, name="x"), axis.Regular(2, 0, 1, name="y")
    ).fill_flattened(df, weight=pd.Series([1.0, 2.0, 3.0]))
    assert h.values().tolist() == [[1.0, 0.0], [2.0, 3.0]]

    h = hist.new.Reg(2, 0, 1).Double().fill_flattened(pd.Series([0.1, 0.6, 0.7]))
    assert h.values().tolist() == [1.0, 2.0]


def test_import_does_not_load_pandas():
    pytest.importorskip("pandas")
    code = "import hist, sys; assert 'pandas' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pandas_imported_after_hist():
    pytest.importorskip("pandas")
    code = """
import sys
import numpy as np
import hist

h = hist.new.Reg(2, 0, 1, name="x").Double().fill_flattened(np.array([0.1]))
assert "pandas" not in sys.modules

import pandas as pd

h.fill_flattened(pd.DataFrame({"x": [0.6]}))
assert h.values().tolist() == [1.0, 1.0], h.values()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
