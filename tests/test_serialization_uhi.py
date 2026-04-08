from __future__ import annotations

import importlib.metadata

import numpy as np
import packaging.version
import pytest

import hist
from hist.serialization import from_uhi, to_uhi

bhs = pytest.importorskip("boost_histogram.serialization")

BHV = packaging.version.Version(importlib.metadata.version("boost_histogram"))
BHMETADATA = packaging.version.Version("1.6.1") <= BHV


@pytest.mark.parametrize(
    ("storage_type", "expected_type"),
    [
        pytest.param(hist.storage.AtomicInt64(), "int", id="atomic_int"),
        pytest.param(hist.storage.Int64(), "int", id="int"),
        pytest.param(
            hist.storage.Unlimited(), "double", id="unlimited"
        ),  # This always renders as double
        pytest.param(hist.storage.Double(), "double", id="double"),
    ],
)
def test_simple_to_dict(storage_type: hist.storage.Storage, expected_type: str) -> None:
    h = hist.Hist(
        hist.axis.Regular(10, 0, 1, name="x", label="X"),
        storage=storage_type,
    )
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "regular"
    assert data["axes"][0]["lower"] == 0
    assert data["axes"][0]["upper"] == 1
    assert data["axes"][0]["bins"] == 10
    assert data["axes"][0]["underflow"]
    assert data["axes"][0]["overflow"]
    assert not data["axes"][0]["circular"]
    if BHMETADATA:
        assert data["axes"][0]["metadata"]["name"] == "x"
        assert data["axes"][0]["metadata"]["label"] == "X"
    assert data["storage"]["type"] == expected_type
    assert data["storage"]["values"] == pytest.approx(np.zeros(12))
    assert data["uhi_schema"] == 1
    assert "version" in data["writer_info"]["boost-histogram"]
    assert "version" in data["writer_info"]["hist"]


def test_weighed_to_dict() -> None:
    h = hist.Hist(
        hist.axis.Integer(3, 15),
        storage=hist.storage.Weight(),
    )
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "regular"
    assert data["axes"][0]["lower"] == 3
    assert data["axes"][0]["upper"] == 15
    assert data["axes"][0]["bins"] == 12
    assert data["axes"][0]["underflow"]
    assert data["axes"][0]["overflow"]
    assert not data["axes"][0]["circular"]
    assert data["storage"]["type"] == "weighted"
    assert data["storage"]["values"] == pytest.approx(np.zeros(14))
    assert data["storage"]["variances"] == pytest.approx(np.zeros(14))


def test_mean_to_dict() -> None:
    h = hist.Hist(
        hist.axis.StrCategory(["one", "two", "three"]),
        storage=hist.storage.Mean(),
        name="hi",
    )
    data = to_uhi(h)

    if BHMETADATA:
        assert data["metadata"]["name"] == "hi"
    assert data["axes"][0]["type"] == "category_str"
    assert data["axes"][0]["categories"] == ["one", "two", "three"]
    assert data["axes"][0]["flow"]
    assert data["storage"]["type"] == "mean"
    assert data["storage"]["counts"] == pytest.approx(np.zeros(4))
    assert data["storage"]["values"] == pytest.approx(np.zeros(4))
    assert data["storage"]["variances"] == pytest.approx(np.zeros(4))


def test_weighted_mean_to_dict() -> None:
    h = hist.Hist(
        hist.axis.IntCategory([1, 2, 3]),
        storage=hist.storage.WeightedMean(),
    )
    h.fill([1, 2, 3, 50], weight=[10, 20, 30, 5], sample=[100, 200, 300, 1])
    h.fill([1, 2, 3, -3], weight=[10, 20, 30, 5], sample=[100, 200, 300, 1])
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "category_int"
    assert data["axes"][0]["categories"] == pytest.approx([1, 2, 3])
    assert data["axes"][0]["flow"]
    assert data["storage"]["type"] == "weighted_mean"
    assert data["storage"]["sum_of_weights"] == pytest.approx(
        np.array([20, 40, 60, 10])
    )
    assert data["storage"]["sum_of_weights_squared"] == pytest.approx(
        np.array([200, 800, 1800, 50])
    )
    assert data["storage"]["values"] == pytest.approx(np.array([100, 200, 300, 1]))
    assert data["storage"]["variances"] == pytest.approx(np.zeros(4))


def test_transform_log_axis_to_dict() -> None:
    h = hist.Hist(hist.axis.Regular(10, 1, 10, transform=hist.axis.transform.log))
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "variable"
    assert data["axes"][0]["edges"] == pytest.approx(
        np.exp(np.linspace(0, np.log(10), 11))
    )


def test_transform_sqrt_axis_to_dict() -> None:
    h = hist.Hist(hist.axis.Regular(10, 0, 10, transform=hist.axis.transform.sqrt))
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "variable"
    assert data["axes"][0]["edges"] == pytest.approx(
        (np.linspace(0, np.sqrt(10), 11)) ** 2
    )


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(hist.storage.AtomicInt64(), id="atomic_int"),
        pytest.param(hist.storage.Int64(), id="int"),
        pytest.param(hist.storage.Double(), id="double"),
        pytest.param(hist.storage.Unlimited(), id="unlimited"),
    ],
)
def test_round_trip_simple(storage_type: hist.storage.Storage) -> None:
    h = hist.Hist(
        hist.axis.Regular(10, 0, 10),
        storage=storage_type,
    )
    h.fill([-1, 0, 0, 1, 20, 20, 20])
    data = to_uhi(h)
    h2 = from_uhi(data)

    if BHMETADATA and isinstance(
        storage_type, (hist.storage.Int64, hist.storage.Double)
    ):
        assert h._hist == h2._hist
        assert h == h2

    assert h.view() == pytest.approx(h2.view())


def test_round_trip_weighted() -> None:
    h = hist.Hist(
        hist.axis.Variable([1, 2, 4, 5], circular=True),
        storage=hist.storage.Weight(),
    )
    h.fill(["1", "2", "3"], weight=[10, 20, 30])
    h.fill(["1", "2", "3"], weight=[10, 20, 30])
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_round_trip_mean() -> None:
    h = hist.Hist(
        hist.axis.StrCategory(["1", "2", "3"]),
        storage=hist.storage.Mean(),
    )
    h.fill(["1", "2", "3"], weight=[10, 20, 30], sample=[100, 200, 300])
    h.fill(["1", "2", "3"], weight=[10, 20, 30], sample=[100, 200, 300])
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_round_trip_weighted_mean() -> None:
    h = hist.Hist(
        hist.axis.IntCategory([1, 2, 3]),
        storage=hist.storage.WeightedMean(),
    )
    h.fill([1, 2, 3], weight=[10, 20, 30], sample=[100, 200, 300])
    h.fill([1, 2, 3], weight=[10, 20, 30], sample=[100, 200, 300])
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_uhi_wrapper():
    h = hist.Hist(
        hist.axis.IntCategory([1, 2, 3]),
        storage=hist.storage.WeightedMean(),
    )
    assert to_uhi(h).keys() == h._to_uhi_().keys()
    data = h._to_uhi_()
    assert repr(from_uhi(data)) == repr(hist.Hist._from_uhi_(data))

    assert "version" in data["writer_info"]["hist"]


def test_uhi_direct_conversion():
    h = hist.Hist(
        hist.axis.IntCategory([1, 2, 3]),
        storage=hist.storage.Int64(),
    )
    uhi_dict = h._to_uhi_()
    h2 = hist.Hist(uhi_dict)
    if BHMETADATA:
        assert h == h2


def test_round_trip_native() -> None:
    h = hist.Hist(
        hist.axis.Integer(0, 10),
        storage=hist.storage.AtomicInt64(),
    )
    h.fill([-1, 0, 0, 1, 20, 20, 20])
    data = to_uhi(h)
    h2 = from_uhi(data)

    if BHMETADATA:
        assert h == h2

    assert isinstance(h2.axes[0], hist.axis.Integer)
    assert h2.storage_type is hist.storage.AtomicInt64


def test_round_trip_clean() -> None:
    h = hist.Hist(
        hist.axis.Integer(0, 10),
        storage=hist.storage.AtomicInt64(),
    )
    h.fill([-1, 0, 0, 1, 20, 20, 20])

    data = to_uhi(h)
    data = bhs.remove_writer_info(data)
    h2 = from_uhi(data)

    assert isinstance(h2.axes[0], hist.axis.Regular)
    assert h2.storage_type is hist.storage.Int64
