from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest import approx

import hist
import hist.io.yoda

DIR = Path(__file__).parent.resolve()

SKIPPED = r"Skipped 1 unsupported YODA object\(s\): /s1 \(SCATTER2D\)"


@pytest.fixture(scope="module")
def yoda2():
    with pytest.warns(UserWarning, match=SKIPPED):
        return hist.io.yoda.read(DIR / "yoda_files/yoda2.yoda")


@pytest.fixture(scope="module")
def yoda1():
    return hist.io.yoda.read(DIR / "yoda_files/yoda1.yoda")


def test_read_keys(yoda2, yoda1):
    assert list(yoda2) == [
        "/h1",
        "/h2",
        "/p1",
        "/p2",
        "/c",
        "/hint",
        "/hstr",
        "/hmix",
        "/h3",
        "/e1",
        "/e2",
        "/e0",
    ]
    assert list(yoda1) == ["/h1", "/h2", "/p1", "/p2", "/c"]


def test_read_histo1d(yoda2):
    h = yoda2["/h1"]
    assert h.label == "title one"
    assert h.metadata == {"Custom": "hello"}
    assert h.storage_type is hist.storage.Weight
    assert isinstance(h.axes[0], hist.axis.Variable)
    assert h.axes[0].edges == approx([0, 1 / 3, 2 / 3, 1], abs=1e-6)
    assert h.values(flow=True) == approx([1, 1, 2, 0, 3])
    assert h.variances(flow=True) == approx([1, 1, 4, 0, 9])


def test_read_histo2d(yoda2):
    h = yoda2["/h2"]
    assert h.label is None
    assert h.metadata == {}
    assert h.axes[0].edges == approx([0, 0.5, 1])
    assert h.axes[1].edges == approx([0, 1, 2, 3])
    expected = np.zeros((4, 5))
    expected[1, 1] = 1
    expected[2, 3] = 2
    expected[0, 4] = 1
    assert h.values(flow=True) == approx(expected)
    assert h.variances(flow=True)[2, 3] == approx(4)


def test_read_histo3d(yoda2):
    h = yoda2["/h3"]
    assert h.shape == (2, 2, 2)
    assert h[0, 1, 0].value == approx(1)
    assert h.sum(flow=True).value == approx(1)


def test_read_profile(yoda2):
    h = yoda2["/p1"]
    assert h.storage_type is hist.storage.WeightedMean
    assert h.values() == approx([5, 10 / 3, 0])
    view = h.view()
    assert view["sum_of_weights"] == approx([1, 3, 0])
    assert view["sum_of_weights_squared"] == approx([1, 5, 0])

    # Same as filling directly
    direct = hist.Hist(h.axes[0], storage=hist.storage.WeightedMean())
    direct.fill([0.1, 0.5, 0.5], sample=[5, 3, 4], weight=[1, 2, 1])
    assert h.variances() == approx(direct.variances(), nan_ok=True)

    h = yoda2["/p2"]
    assert h.values(flow=True)[1, 1] == approx(5)
    assert h.sum(flow=True).sum_of_weights == approx(1)


def test_read_counter(yoda2):
    h = yoda2["/c"]
    assert h.ndim == 0
    assert h.sum().value == approx(5)
    assert h.sum().variance == approx(13)


def test_read_discrete(yoda2):
    h = yoda2["/hint"]
    assert isinstance(h.axes[0], hist.axis.IntCategory)
    assert list(h.axes[0]) == [1, 2, 5]
    assert h.values(flow=True) == approx([0, 1, 0, 1])

    h = yoda2["/hstr"]
    assert isinstance(h.axes[0], hist.axis.StrCategory)
    assert list(h.axes[0]) == ["a", "b"]
    assert h.values(flow=True) == approx([0, 2, 0])

    h = yoda2["/hmix"]
    assert h["a", 1].value == approx(1)
    assert h[hist.overflow, 0].value == approx(3)
    assert h.sum(flow=True).value == approx(4)


def test_read_estimate(yoda2):
    h = yoda2["/e1"]
    assert h.label == "title one"
    assert h.values() == approx([3, 6, 0])
    # A total error, when present, wins over the other sources
    assert h.variances() == approx([9, 36, 0])
    assert np.isnan(h.values(flow=True)[0])

    assert yoda2["/e2"].values()[1, 2] == approx(4)
    assert yoda2["/e0"].sum().value == approx(5)
    assert yoda2["/e0"].sum().variance == approx(13)


def test_read_estimate_sources():
    text = """\
BEGIN YODA_ESTIMATE1D_V3 /e
Path: /e
---
Edges(A1): [0.0, 1.0, 2.0]
ErrorLabels: ["stat", "sys"]
# value  errDn(1)  errUp(1)  errDn(2)  errUp(2)
nan  ---  ---  ---  ---
1.0  -3.0  3.0  -4.0  4.0
2.0  -1.0  3.0  ---  ---
nan  ---  ---  ---  ---
END YODA_ESTIMATE1D_V3
"""
    h = hist.io.yoda.loads(text)["/e"]
    assert h.values() == approx([1, 2])
    assert h.variances() == approx([25, 4])


def test_read_legacy(yoda2, yoda1):
    for path in ("/h1", "/p1", "/c"):
        assert yoda1[path].storage_type is yoda2[path].storage_type
        new = yoda2[path].view(flow=True)
        old = yoda1[path].view(flow=True)
        for field in new.dtype.names:
            assert old[field] == approx(new[field]), f"{path} {field}"

    # No flow bins in legacy 2D objects
    for path in ("/h2", "/p2"):
        new = yoda2[path].view()
        old = yoda1[path].view()
        for field in new.dtype.names:
            assert old[field] == approx(new[field]), f"{path} {field}"

    assert yoda1["/h1"].label == "title one"
    assert yoda1["/h2"].label is None


def test_read_legacy_gaps():
    text = """\
BEGIN YODA_HISTO1D_V2 /gap
Path: /gap
---
Total  Total  3 3 0 0 3
Underflow  Underflow  0 0 0 0 0
Overflow  Overflow  1 1 0 0 1
3.0  4.0  2 2 0 0 2
0.0  1.0  1 1 0 0 1
END YODA_HISTO1D_V2
"""
    h = hist.io.yoda.loads(text)["/gap"]
    assert h.axes[0].edges == approx([0, 1, 3, 4])
    assert h.values(flow=True) == approx([0, 1, 0, 2, 1])


@pytest.mark.parametrize("text", ["junk", "BEGIN YODA_COUNTER_V3 /c\n---\n1 1 1\n"])
def test_read_invalid(text):
    with pytest.raises(ValueError, match=r"BEGIN|END"):
        hist.io.yoda.loads(text)


def roundtrip(h, **kwargs):
    return hist.io.yoda.loads(hist.io.yoda.dumps({"/h": h}, precision=17, **kwargs))[
        "/h"
    ]


@pytest.mark.parametrize(
    "storage",
    [hist.storage.Double(), hist.storage.Int64(), hist.storage.Weight()],
)
@pytest.mark.parametrize(
    "axes",
    [
        (),
        (hist.axis.Regular(4, 0, 2),),
        (hist.axis.Variable([0, 0.5, 2]), hist.axis.IntCategory([3, 1, 7])),
        (hist.axis.StrCategory(["x", 'q"uo\\te']), hist.axis.Regular(3, 0, 2)),
        (
            hist.axis.Regular(2, 0, 2),
            hist.axis.Regular(2, 0, 2),
            hist.axis.Variable([0, 1, 2]),
        ),
    ],
)
def test_roundtrip(axes, storage):
    rng = np.random.default_rng(42)
    h = hist.Hist(*axes, storage=storage)
    view = h.view(flow=True)
    if view.dtype.names:
        view["value"] = rng.integers(0, 50, size=view.shape)
        view["variance"] = rng.integers(0, 50, size=view.shape)
    else:
        view[...] = rng.integers(0, 50, size=view.shape)

    h2 = roundtrip(h)
    assert [list(ax) for ax in h2.axes] == [list(ax) for ax in h.axes]
    assert h2.values(flow=True) == approx(h.values(flow=True))
    assert h2.variances(flow=True) == approx(h.variances(flow=True))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("ndim", [1, 2])
def test_roundtrip_profile(ndim, weighted):
    rng = np.random.default_rng(42)
    axes = [hist.axis.Regular(3, 0, 1), hist.axis.IntCategory([1, 2])][:ndim]
    storage = hist.storage.WeightedMean() if weighted else hist.storage.Mean()
    h = hist.Hist(*axes, storage=storage)
    data = [rng.uniform(-0.2, 1.2, 200), rng.integers(1, 4, 200)][:ndim]
    weight = rng.uniform(0.5, 2, 200) if weighted else None
    h.fill(*data, sample=rng.normal(size=200), weight=weight)

    h2 = roundtrip(h)
    assert h2.storage_type is hist.storage.WeightedMean
    assert h2.values(flow=True) == approx(h.values(flow=True))
    assert h2.counts(flow=True) == approx(h.counts(flow=True))
    assert h2.variances(flow=True) == approx(h.variances(flow=True), nan_ok=True)


def test_roundtrip_flow():
    # YODA always has flow bins, and one "otherflow" bin on discrete axes
    h = hist.Hist(
        hist.axis.Regular(2, 0, 1, flow=False),
        hist.axis.Integer(0, 2),
        storage=hist.storage.Double(),
    )
    h.fill([0.1, 0.1, 0.6, 0.6], [-1, 0, 1, 5])

    h2 = roundtrip(h)
    assert h2.values(flow=True) == approx(
        np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]])
    )
    assert isinstance(h2.axes[1], hist.axis.IntCategory)
    assert list(h2.axes[1]) == [0, 1]


def test_roundtrip_annotations():
    h = hist.Hist.new.Reg(2, 0, 1).Double()
    h.label = "A title"
    h.metadata = {"IsRef": 1, "ScaledBy": "2.0"}

    h2 = roundtrip(h)
    assert h2.label == "A title"
    assert h2.metadata == {"IsRef": "1", "ScaledBy": "2.0"}


def test_precision():
    h = hist.Hist.new.Reg(1, 0, 1).Double().fill([0.5])
    text = hist.io.yoda.dumps({"/h": h})
    assert "Edges(A1): [0.000000e+00, 1.000000e+00]" in text
    assert "Edges(A1): [0.00e+00, 1.00e+00]" in hist.io.yoda.dumps(
        {"/h": h}, precision=2
    )
    assert not any(line != line.rstrip() for line in text.splitlines())


@pytest.mark.parametrize(
    "h",
    [
        hist.Hist.new.Reg(2, 0, 1).Reg(2, 0, 1).Reg(2, 0, 1).Reg(2, 0, 1).Double(),
        hist.Hist.new.Reg(2, 0, 1).Reg(2, 0, 1).IntCat([1]).Double(),
        hist.Hist.new.Reg(2, 0, 1).Reg(2, 0, 1).Reg(2, 0, 1).Mean(),
        hist.Hist(storage=hist.storage.Mean()),
    ],
)
def test_write_unsupported(h):
    with pytest.raises(ValueError, match="YODA does not support"):
        hist.io.yoda.dumps({"/h": h})


@pytest.mark.parametrize("name", ["out.yoda", "out.yoda.gz"])
def test_files(tmp_path, name):
    h = hist.Hist.new.Reg(4, 0, 1).Weight().fill([0.1, 0.4], weight=[2, 3])
    path = tmp_path / name
    hist.io.yoda.write(path, {"/a": h, "/b": h})
    assert path.read_bytes().startswith(b"\x1f\x8b") == name.endswith(".gz")

    result = hist.io.yoda.read(path)
    assert list(result) == ["/a", "/b"]
    assert result["/b"].values() == approx(h.values())


# These tests need YODA, which is only on conda-forge


@pytest.fixture
def yoda():
    return pytest.importorskip("yoda")


def moments(bins, name):
    """
    sumW, sumW2, and the first and second moment of the sample for YODA bins.
    """
    return [
        np.array([getattr(b, method)() for b in bins])
        for method in ("sumW", "sumW2", f"sumW{name}", f"sumW{name}2")
    ]


def hist_moments(h):
    view = h.view(flow=True)
    if h.storage_type is hist.storage.Mean:
        sumw = sumw2 = view["count"]
        deltas = view["_sum_of_deltas_squared"]
    else:
        sumw, sumw2 = view["sum_of_weights"], view["sum_of_weights_squared"]
        deltas = view["_sum_of_weighted_deltas_squared"]
    return [sumw, sumw2, sumw * view["value"], deltas + sumw * view["value"] ** 2]


def test_yoda_reads_hist(yoda, tmp_path):
    rng = np.random.default_rng(42)
    hists = {
        "/h1": hist.Hist.new.Reg(5, 0, 1).Double().fill(rng.uniform(-0.2, 1.2, 50)),
        "/h2": hist.Hist.new.Reg(3, 0, 1)
        .Var([0, 0.2, 1])
        .Weight()
        .fill(
            rng.uniform(-0.2, 1.2, 50),
            rng.uniform(-0.2, 1.2, 50),
            weight=rng.uniform(1, 2, 50),
        ),
        "/h3": hist.Hist.new.Reg(2, 0, 1)
        .Reg(2, 0, 1)
        .Reg(2, 0, 1)
        .Double()
        .fill(*rng.uniform(-0.2, 1.2, (3, 50))),
        "/cat": hist.Hist.new.StrCat(["a", "b"])
        .IntCat([4, 2])
        .Double()
        .fill(["a", "b", "b", "c"], [2, 4, 4, 9]),
        "/p1": hist.Hist.new.Reg(4, 0, 1)
        .WeightedMean()
        .fill(
            rng.uniform(-0.2, 1.2, 50),
            sample=rng.normal(size=50),
            weight=rng.uniform(1, 2, 50),
        ),
        "/p2": hist.Hist.new.Reg(2, 0, 1)
        .Reg(2, 0, 1)
        .Mean()
        .fill(*rng.uniform(-0.2, 1.2, (2, 50)), sample=rng.normal(size=50)),
        "/c": hist.Hist(storage=hist.storage.Weight()),
    }
    hists["/c"].fill(weight=2.0)
    hists["/c"].fill(weight=3.0)
    hists["/h1"].label = "My title"
    hists["/h1"].metadata = {"Extra": "some text"}
    path = tmp_path / "hist.yoda"
    hist.io.yoda.write(path, hists, precision=12)

    aos = yoda.read(str(path))
    assert list(aos) == sorted(hists)
    assert {k: ao.type() for k, ao in aos.items()} == {
        "/h1": "Histo1D",
        "/h2": "Histo2D",
        "/h3": "Histo3D",
        "/cat": "BinnedHisto<s,i>",
        "/p1": "Profile1D",
        "/p2": "Profile2D",
        "/c": "Counter",
    }
    assert aos["/h1"].title() == "My title"
    assert aos["/h1"].annotation("Extra") == "some text"
    assert aos["/h2"].xEdges() == approx(hists["/h2"].axes[0].edges)
    assert aos["/h2"].yEdges() == approx(hists["/h2"].axes[1].edges)
    assert list(aos["/cat"].xEdges()) == ["a", "b"]
    assert list(aos["/cat"].yEdges()) == [4, 2]

    for name in ("/h1", "/h2", "/h3"):
        bins = aos[name].bins(True)
        h = hists[name]
        sumw = np.reshape([b.sumW() for b in bins], h.values(flow=True).shape, "F")
        sumw2 = np.reshape([b.sumW2() for b in bins], sumw.shape, "F")
        assert sumw == approx(h.values(flow=True)), name
        assert sumw2 == approx(h.variances(flow=True)), name
    assert aos["/h1"].numEntries() == approx(50)
    assert aos["/h1"].xMean() == approx(0.5, abs=0.1)

    # Otherflow is first in YODA
    sumw = np.reshape([b.sumW() for b in aos["/cat"].bins(True)], (3, 3), "F")
    assert sumw == approx(np.array([[1, 0, 0], [0, 0, 1], [0, 2, 0]]))

    for name in ("/p1", "/p2"):
        h = hists[name]
        got = moments(aos[name].bins(True), "XYZ"[h.ndim])
        expected = hist_moments(h)
        for i in range(4):
            assert np.reshape(got[i], expected[i].shape, "F") == approx(expected[i])

    assert aos["/c"].sumW() == approx(5)
    assert aos["/c"].sumW2() == approx(13)


def test_hist_reads_yoda(yoda, tmp_path):
    rng = np.random.default_rng(42)
    h1 = yoda.Histo1D(5, 0, 1, "/h1", "My title")
    h2 = yoda.BinnedHisto2D([1, 3], [0.0, 0.5, 2.0], "/h2")
    p1 = yoda.Profile1D(4, 0, 1, "/p1")
    for x, y, w in zip(*rng.uniform(-0.2, 1.2, (3, 50)), strict=True):
        h1.fill(x, w)
        h2.fill(int(4 * x), y, w)
        p1.fill(x, y, w)
    est = h1.mkEstimate("/e1")
    path = tmp_path / "yoda.yoda.gz"
    yoda.write([h1, h2, p1, est], str(path))

    hists = hist.io.yoda.read(path)
    assert sorted(hists) == ["/e1", "/h1", "/h2", "/p1"]
    assert hists["/h1"].label == "My title"

    for name, ao in (("/h1", h1), ("/h2", h2)):
        h = hists[name]
        shape = h.values(flow=True).shape
        sumw = np.reshape([b.sumW() for b in ao.bins(True)], shape, "F")
        sumw2 = np.reshape([b.sumW2() for b in ao.bins(True)], shape, "F")
        if name == "/h2":
            sumw, sumw2 = np.roll(sumw, -1, axis=0), np.roll(sumw2, -1, axis=0)
        assert h.values(flow=True) == approx(sumw, rel=1e-5), name
        assert h.variances(flow=True) == approx(sumw2, rel=1e-5), name

    # The weighted error on the mean is defined differently in YODA
    got = hist_moments(hists["/p1"])
    expected = moments(p1.bins(True), "Y")
    for i in range(4):
        assert got[i] == approx(expected[i], rel=1e-5, abs=1e-6)

    bins = est.bins()
    assert hists["/e1"].values() == approx([b.val() for b in bins], rel=1e-5)
    assert hists["/e1"].variances() == approx(
        [b.totalErrAvg() ** 2 for b in bins], rel=1e-5
    )

    # YODA can read a file again after a round trip through hist
    path2 = tmp_path / "again.yoda"
    hist.io.yoda.write(path2, {k: v for k, v in hists.items() if k != "/e1"})
    again = yoda.read(str(path2))
    assert again["/h1"].sumW() == approx(h1.sumW(), rel=1e-5)
    assert again["/p1"].sumW() == approx(p1.sumW(), rel=1e-5)
