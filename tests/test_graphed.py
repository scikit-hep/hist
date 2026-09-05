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
from graphed.core.execution import SequentialRunner  # noqa: E402
from graphed.core import Partition  # noqa: E402

RNG = np.random.default_rng(7)
DATA = RNG.normal(5.0, 2.0, 800)
WEIGHTS = RNG.uniform(0.5, 1.5, 800)
DATA2 = RNG.normal(5.0, 2.0, 800)  # a distinct second axis, so axis ORDER is observable


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
    from graphed.numpy import NumpyBackend
    from graphed.numpy.forms import NumpyForm

    s = Session(NumpyBackend())
    src = ChunkedNumpySource(DATA)
    return s.source("x", form=NumpyForm(DATA.dtype, shape=(None,)), data=src), src


def test_quickconstruct_matches_the_eager_twin_bit_for_bit():
    pytest.importorskip("graphed.numpy")
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
    pytest.importorskip("graphed.numpy")
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
    pytest.importorskip("graphed.awkward")
    from graphed.awkward import AwkwardBackend, AwkwardForm

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
    pytest.importorskip("graphed.awkward")
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
    pytest.importorskip("graphed.numpy")
    pexec = pytest.importorskip("graphed_executors.local")

    x, _ = _numpy_source()
    h = hist_graphed.Hist.new.Reg(16, 0, 10, name="v").Int64()
    h.fill(v=x).fill(v=abs(x) * 0.5)
    direct = SequentialRunner().run(h.plan(steps_per_file=3)).value
    later = pexec.ProcessPoolExecutor(max_workers=2).run(h.plan(steps_per_file=3)).value
    eager = hist.Hist.new.Reg(16, 0, 10, name="v").Int64()
    eager.fill(v=DATA)
    eager.fill(v=np.abs(DATA) * 0.5)
    assert np.array_equal(direct.values(flow=True), eager.values(flow=True))
    assert np.array_equal(np.asarray(later.values(flow=True)), eager.values(flow=True))


def test_variation_axis_flag_reaches_graphed_and_is_not_read_as_an_axis_name():
    """hist's named-axis fill maps every keyword to an axis name, which swallowed graphed's
    fill-MODE flags (`variation_axis=`) with `ValueError: axis name ... could not be found`."""
    pytest.importorskip("graphed.numpy")
    import boost_histogram as bh
    import graphed_histogram as ghist
    import graphed_histogram.boost as ghb

    def varied():
        x, _ = _numpy_source()
        w = x * 0.1
        return x, graphed.vary(w, "wgt", up=w * 1.2, down=w * 0.8)

    x, w = varied()
    h = hist_graphed.Hist.new.Reg(10, 0, 10, name="met").Weight()
    h.fill(met=x, weight=[w], variation_axis=True)
    (got,) = dict(SequentialRunner().run(ghist.plan({"h": h}, steps_per_file=4)).value).values()

    # axis mode: graphed declares the extra "variation" StrCategory itself
    assert [ax.__class__.__name__ for ax in got.axes] == ["Regular", "StrCategory"]
    assert list(got.axes[1]) == ["nominal", "wgt_down", "wgt_up"]

    # ... and it is the same histogram the low-level fill produces
    x2, w2 = varied()
    low = ghb.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())
    low.fill(x2, weight=[w2], variation_axis=True)
    (want,) = dict(SequentialRunner().run(ghist.plan({"h": low}, steps_per_file=4)).value).values()
    assert np.array_equal(got.view(flow=True), want.view(flow=True))

    # the other end of the class: NamedHist, and the sibling control kwarg `unweighted=`
    x3, w3 = varied()
    named = hist_graphed.NamedHist.new.Reg(10, 0, 10, name="met").Weight()
    named.fill(met=x3, weight=[w3], variation_axis=True)
    (n,) = dict(SequentialRunner().run(ghist.plan({"h": named}, steps_per_file=4)).value).values()
    assert np.array_equal(n.view(flow=True), want.view(flow=True))

    x4, _ = _numpy_source()
    bare = hist_graphed.Hist.new.Reg(10, 0, 10, name="met").Weight()
    bare.fill(met=x4, unweighted=True)  # read as an axis name before the passthrough
    assert SequentialRunner().run(bare.plan(steps_per_file=4)).value.sum().value == np.count_nonzero(
        (DATA >= 0) & (DATA < 10)
    )


def _named_numpy_source(name, data):  # type: ignore[no-untyped-def]
    from graphed.numpy import NumpyBackend
    from graphed.numpy.forms import NumpyForm

    s = Session(NumpyBackend())
    return s.source(name, form=NumpyForm(data.dtype, shape=(None,)), data=ChunkedNumpySource(data))


def test_flag_path_reorders_axes_and_engages_only_for_flags():
    """Two guards the 1-D flag tests cannot see: the flag path re-derives axis ORDER from
    `_name_to_index` (a 2-D out-of-order kwarg fill must equal the low-level POSITIONAL fill, not a
    swapped one), and it engages ONLY when a flag is set (a no-flag fill still takes hist's path,
    whose missing-axis message differs from the arity error the flag path would raise). Also: the
    `unweighted=` flag is forwarded, not defaulted away."""
    pytest.importorskip("graphed.numpy")
    import boost_histogram as bh
    import graphed_histogram as ghist
    import graphed_histogram.boost as ghb
    from graphed.errors import GraphedError

    def run(h):  # type: ignore[no-untyped-def]
        return next(iter(dict(SequentialRunner().run(ghist.plan({"h": h}, steps_per_file=4)).value).values()))

    # (reorder) axes are declared a, b; passing them as b=, a= must land on a, b — i.e. equal the
    # positional fill(a_data, b_data), and NOT the swapped fill(b_data, a_data).
    h = hist_graphed.Hist.new.Reg(10, 0, 10, name="a").Reg(10, 0, 10, name="b").Double()
    h.fill(b=_named_numpy_source("b", DATA2), a=_named_numpy_source("a", DATA), variation_axis=True)
    got = run(h).view(flow=True)
    low = ghb.Histogram(bh.axis.Regular(10, 0, 10), bh.axis.Regular(10, 0, 10))
    low.fill(_named_numpy_source("a", DATA), _named_numpy_source("b", DATA2), variation_axis=True)
    assert np.array_equal(got, run(low).view(flow=True))
    swapped = ghb.Histogram(bh.axis.Regular(10, 0, 10), bh.axis.Regular(10, 0, 10))
    swapped.fill(_named_numpy_source("b", DATA2), _named_numpy_source("a", DATA), variation_axis=True)
    assert not np.array_equal(got, run(swapped).view(flow=True))  # order genuinely matters

    # (routing) no flag -> hist's path, whose missing-axis TypeError names the axis ("Missing
    # values ... ['b']"); the flag path would instead raise graphed's arity message.
    miss = hist_graphed.Hist.new.Reg(10, 0, 10, name="a").Reg(10, 0, 10, name="b").Double()
    with pytest.raises(TypeError, match="Missing values"):
        miss.fill(a=_named_numpy_source("a", DATA))

    # (unweighted) forwarded, not silently set False: unweighted=True with a weight= factor is the
    # contradiction graphed refuses at fill time.
    x = _named_numpy_source("met", DATA)
    hbad = hist_graphed.Hist.new.Reg(10, 0, 10, name="met").Weight()
    with pytest.raises(GraphedError, match="unweighted=True suppresses"):
        hbad.fill(met=x, weight=[x], unweighted=True)


def test_sibling_mode_is_unchanged_by_the_flag_passthrough():
    pytest.importorskip("graphed.numpy")
    import graphed_histogram as ghist

    x, _ = _numpy_source()
    w = x * 0.1
    h = hist_graphed.Hist.new.Reg(10, 0, 10, name="met").Weight()
    h.fill(met=x, weight=[graphed.vary(w, "wgt", up=w * 1.2, down=w * 0.8)])  # default: siblings
    slots = dict(SequentialRunner().run(ghist.plan({"h": h}, steps_per_file=4)).value)
    assert {k[1] for k in slots} == {"nominal", "wgt_up", "wgt_down"}
    for got in slots.values():
        assert [ax.__class__.__name__ for ax in got.axes] == ["Regular"]  # no variation axis
