# Tests for hist.graphed — deferred hist filling on graphed task graphs (HIST-1; P0.1 of the
# ADL-benchmarks port). QuickConstruct-built deferred histograms must equal their eager hist.Hist
# twins BIT FOR BIT over graphed-numpy AND graphed-awkward sources and over a real uproot TTree,
# with named-axis fills, weights, NamedHist, and the partition-wise efficiency witness (the
# source's whole-dataset loader never runs). Evaluation is graphed's idiom [freeze-HIST-2,
# user-directed]: plan() + an R7 executor; hist.Hist(value) wraps results back in-memory.
from __future__ import annotations

import numpy as np
import pytest

import hist

gh = pytest.importorskip("graphed_histogram")
graphed = pytest.importorskip("graphed")
hist_graphed = pytest.importorskip("hist.graphed")

from dataclasses import dataclass, field  # noqa: E402

from graphed import Session  # noqa: E402
from graphed_core.execution import SequentialRunner  # noqa: E402
from graphed_core import Partition  # noqa: E402

RNG = np.random.default_rng(7)
DATA = RNG.normal(5.0, 2.0, 800)
WEIGHTS = RNG.uniform(0.5, 1.5, 800)


@dataclass
class ChunkedNumpySource:
    data: np.ndarray
    whole_calls: list = field(default_factory=list)

    def __call__(self) -> np.ndarray:
        self.whole_calls.append(1)
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://x", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition, columns, resources):  # type: ignore[no-untyped-def]
        part = partition.resolve(len(self.data))
        return self.data[part.entry_start : part.entry_stop]


def _numpy_source():
    from graphed_numpy import NumpyBackend
    from graphed_numpy.forms import NumpyForm

    s = Session(NumpyBackend())
    src = ChunkedNumpySource(DATA)
    return s.source("x", form=NumpyForm(DATA.dtype, shape=(None,)), data=src), src


def test_quickconstruct_matches_the_eager_twin_bit_for_bit():
    pytest.importorskip("graphed_numpy")
    x, src = _numpy_source()
    h = hist_graphed.Hist.new.Reg(40, 0, 10, name="met", label="$E_T$").Int64().fill(met=x)
    # graphed idiom: the executor aggregates; hist.Hist(value) wraps back into the in-memory type
    out = hist.Hist(SequentialRunner().run(h.plan(steps_per_file=4)).value)
    assert isinstance(out, hist.Hist) and not isinstance(out, hist_graphed.Hist)
    eager = hist.Hist.new.Reg(40, 0, 10, name="met", label="$E_T$").Int64()
    eager.fill(met=DATA)
    assert np.array_equal(out.values(flow=True), eager.values(flow=True))
    assert out.axes[0].name == "met" and out.axes[0].label == "$E_T$"
    assert out[{"met": sum}] == eager[{"met": sum}]
    assert src.whole_calls == []  # partition-wise: the whole-dataset loader never ran


def test_weighted_2d_and_namedhist():
    pytest.importorskip("graphed_numpy")
    x, _ = _numpy_source()
    h = (
        hist_graphed.NamedHist.new.Reg(10, 0, 10, name="a").Reg(8, 0, 5, name="b").Weight()
        .fill(a=x, b=x * 0.5, weight=np.sqrt(abs(x)))
    )
    out = hist.NamedHist(SequentialRunner().run(h.plan(steps_per_file=3)).value)
    eager = hist.NamedHist.new.Reg(10, 0, 10, name="a").Reg(8, 0, 5, name="b").Weight()
    eager.fill(a=DATA, b=DATA * 0.5, weight=np.sqrt(np.abs(DATA)))
    assert np.allclose(out.values(flow=True), eager.values(flow=True))
    assert np.allclose(out.variances(flow=True), eager.variances(flow=True))


def test_awkward_ragged_fills_flatten():
    ak = pytest.importorskip("awkward")
    pytest.importorskip("graphed_awkward")
    from graphed_awkward import AwkwardBackend, AwkwardForm

    events = ak.Array({"Jet_pt": [[50.0, 30.0], [], [70.0, 20.0, 10.0]] * 50})

    @dataclass
    class ChunkedAkSource:
        data: object
        whole_calls: list = field(default_factory=list)

        def __call__(self):
            self.whole_calls.append(1)
            return self.data

        def partitions(self, steps_per_file: int = 1):
            return tuple(Partition.blind("toy://e", "", s, steps_per_file) for s in range(steps_per_file))

        def read_partition(self, partition, columns, resources):
            part = partition.resolve(len(self.data))
            return self.data[part.entry_start : part.entry_stop]

    s = Session(AwkwardBackend())
    src = ChunkedAkSource(events)
    tt = ak.Array(events.layout.to_typetracer(forget_length=True))
    g = s.source("events", form=AwkwardForm(tt), data=src)

    h = hist_graphed.Hist.new.Reg(20, 0, 100, name="pt").Int64().fill(pt=g.Jet_pt)
    out = hist.Hist(SequentialRunner().run(h.plan(steps_per_file=5)).value)
    eager = hist.Hist.new.Reg(20, 0, 100, name="pt").Int64()
    eager.fill(pt=ak.flatten(events.Jet_pt, axis=None))  # ragged fills flatten completely
    assert np.array_equal(out.values(flow=True), eager.values(flow=True))
    assert src.whole_calls == []


def test_uproot_ttree_fill_end_to_end():
    uproot = pytest.importorskip("uproot")
    pytest.importorskip("graphed_awkward")
    skhep_testdata = pytest.importorskip("skhep_testdata")

    where = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(where, library="ak", filter_name=["px1", "py1"])
    h = hist_graphed.Hist.new.Reg(50, 0, 100, name="pt1").Double().fill(
        pt1=np.hypot(g.px1, g.py1)
    )
    out = hist.Hist(SequentialRunner().run(h.plan(steps_per_file=3)).value)
    raw = uproot.open(where).arrays(["px1", "py1"])
    eager = hist.Hist.new.Reg(50, 0, 100, name="pt1").Double()
    eager.fill(pt1=np.hypot(np.asarray(raw.px1), np.asarray(raw.py1)))
    assert np.array_equal(out.values(flow=True), eager.values(flow=True))


def test_multiple_fills_and_process_executor():
    pytest.importorskip("graphed_numpy")
    pexec = pytest.importorskip("graphed_exec_local")

    x, _ = _numpy_source()
    h = hist_graphed.Hist.new.Reg(16, 0, 10, name="v").Int64()
    h.fill(v=x).fill(v=abs(x) * 0.5)
    direct = SequentialRunner().run(h.plan(steps_per_file=3)).value
    later = pexec.ProcessExecutor(max_workers=2).run(h.plan(steps_per_file=3)).value
    eager = hist.Hist.new.Reg(16, 0, 10, name="v").Int64()
    eager.fill(v=DATA)
    eager.fill(v=np.abs(DATA) * 0.5)
    assert np.array_equal(direct.values(flow=True), eager.values(flow=True))
    assert np.array_equal(np.asarray(later.values(flow=True)), eager.values(flow=True))
