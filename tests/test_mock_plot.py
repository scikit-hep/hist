import numpy as np
import pytest

from hist import Hist


@pytest.fixture(autouse=True)
def mock_test(monkeypatch):
    monkeypatch.setattr(Hist, "plot1d", plot1d_mock)
    monkeypatch.setattr(Hist, "plot2d", plot2d_mock)


def plot1d_mock(*args, **kwargs):
    return "called plot1d"


def plot2d_mock(*args, **kwargs):
    return "called plot2d"


def test_categorical_plot():
    testCat = (
        Hist.new.StrCat("", name="dataset", growth=True)
        .Reg(10, 0, 10, name="good", label="y-axis")
        .Int64()
    )

    testCat.fill(dataset="A", good=np.random.normal(5, 9, 27))

    assert testCat.plot() == "called plot1d"


def test_integer_plot():
    testInt = (
        Hist.new.Int(1, 10, name="nice", label="x-axis")
        .Reg(10, 0, 10, name="good", label="y-axis")
        .Int64()
    )
    testInt.fill(nice=np.random.normal(5, 1, 10), good=np.random.normal(5, 1, 10))

    assert testInt.plot() == "called plot2d"
