from __future__ import annotations

import boost_histogram as bh
import numpy as np
import pytest
from pytest import approx

import hist
from hist import axis
from hist.chunked import ChunkedHist


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_constructor_basic():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    assert h.chunk_axis_names == ("cat",)
    assert len(h) == 0
    assert h.axes[0].name == "x"
    assert h.axes[1].name == "cat"


def test_constructor_int_category():
    h = ChunkedHist(
        axis.Regular(5, 0, 1, name="x"),
        axis.IntCategory([], growth=True, name="icat"),
    )
    assert h.chunk_axis_names == ("icat",)
    assert h.storage_type is bh.storage.Double


def test_constructor_with_storage():
    h = ChunkedHist(
        axis.Regular(5, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="s"),
        storage=bh.storage.Weight(),
    )
    assert h.storage_type is bh.storage.Weight


def test_constructor_no_axes_raises():
    with pytest.raises(ValueError, match="at least one axis"):
        ChunkedHist()


def test_constructor_unnamed_axes_raises():
    with pytest.raises(ValueError, match="all axes must have names"):
        ChunkedHist(axis.Regular(10, 0, 1))


def test_constructor_transformed_axis_raises():
    with pytest.raises(ValueError, match="transformed Regular axes"):
        ChunkedHist(
            axis.Regular(10, 0, 1, name="x", transform=axis.transform.sqrt),
            axis.StrCategory([], growth=True, name="cat"),
        )


def test_constructor_unsupported_storage_raises():
    # Mean and WeightedMean are not supported yet (see implementation note)
    with pytest.raises(ValueError, match="Mean storage"):
        ChunkedHist(
            axis.Regular(10, 0, 1, name="x"),
            axis.StrCategory([], growth=True, name="cat"),
            storage=bh.storage.Mean(),
        )
    with pytest.raises(ValueError, match="WeightedMean storage"):
        ChunkedHist(
            axis.Regular(10, 0, 1, name="x"),
            axis.StrCategory([], growth=True, name="cat"),
            storage=bh.storage.WeightedMean(),
        )


def test_constructor_growing_dense_axis_raises():
    # Growth would reallocate the scratch buffer mid-fill, invalidating chunks
    with pytest.raises(ValueError, match="growing dense axes"):
        ChunkedHist(
            axis.Regular(10, 0, 1, name="x", growth=True),
            axis.StrCategory([], growth=True, name="cat"),
        )
    with pytest.raises(ValueError, match="growing dense axes"):
        ChunkedHist(
            axis.Integer(0, 5, name="i", growth=True),
            axis.StrCategory([], growth=True, name="cat"),
        )


def test_constructor_no_chunk_axes():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(5, 0, 1, name="y"),
    )
    assert h.chunk_axes == []
    assert h.chunk_axis_names == ()


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------
def test_fill_basic():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2, 0.4], cat="a")
    assert len(h) == 1
    assert ("a",) in h
    # Check counts via materialization
    dense = h.to_hist()
    assert dense[{"cat": "a", "x": bh.loc(0.2)}] == 1
    assert dense[{"cat": "a", "x": bh.loc(0.4)}] == 1


def test_fill_multiple_categories():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    h.fill(x=[0.3, 0.5], cat="b")
    h.fill(x=[0.4], cat="a")

    assert len(h) == 2
    dense = h.to_hist()
    assert dense[{"cat": "a", "x": bh.loc(0.2)}] == 1
    assert dense[{"cat": "a", "x": bh.loc(0.4)}] == 1
    assert dense[{"cat": "b", "x": bh.loc(0.3)}] == 1
    assert dense[{"cat": "b", "x": bh.loc(0.5)}] == 1


def test_fill_missing_chunk_axis_raises():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    with pytest.raises(ValueError, match="missing chunk axes"):
        h.fill(x=[0.2])


def test_fill_array_chunk_axis_raises():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    with pytest.raises(ValueError, match="only accepts scalar"):
        h.fill(x=[0.2, 0.4], cat=["a", "b"])


def test_fill_weight():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
        storage=bh.storage.Weight(),
    )
    h.fill(x=[0.2, 0.4], cat="a", weight=[1.0, 2.0])
    dense = h.to_hist()
    assert dense[{"cat": "a", "x": bh.loc(0.2)}].variance == approx(1.0)
    assert dense[{"cat": "a", "x": bh.loc(0.4)}].variance == approx(4.0)


def test_fill_int_category():
    h = ChunkedHist(
        axis.Regular(5, 0, 1, name="x"),
        axis.IntCategory([], growth=True, name="icat"),
    )
    h.fill(x=[0.1], icat=42)
    h.fill(x=[0.2], icat=7)
    assert len(h) == 2
    dense = h.to_hist()
    assert dense[{"icat": bh.loc(42), "x": bh.loc(0.1)}] == 1
    assert dense[{"icat": bh.loc(7), "x": bh.loc(0.2)}] == 1


def test_fill_unknown_key_non_growing_raises():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], name="cat"),  # growth=False
    )
    h.fill(x=[0.2], cat="a")
    with pytest.raises(ValueError, match="non-growing chunk axis"):
        h.fill(x=[0.4], cat="z")
    # The failed fill must not leave any trace
    assert list(h.keys()) == [("a",)]
    h.fill(x=[0.6], cat="a")
    dense = h.to_hist()
    assert list(dense.axes[1]) == ["a"]
    assert dense[{"cat": "a", "x": bh.loc(0.2)}] == 1
    assert dense[{"cat": "a", "x": bh.loc(0.4)}] == 0
    assert dense[{"cat": "a", "x": bh.loc(0.6)}] == 1


def test_fill_no_chunk_axes():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
    )
    h.fill(x=[0.2, 0.4])
    assert len(h) == 1
    dense = h.to_hist()
    assert dense[{"x": bh.loc(0.2)}] == 1
    assert dense[{"x": bh.loc(0.4)}] == 1


# ---------------------------------------------------------------------------
# to_hist / from_hist
# ---------------------------------------------------------------------------
def test_to_hist_empty():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    dense = h.to_hist()
    assert isinstance(dense, hist.Hist)
    assert dense.ndim == 2
    assert len(dense.axes[1]) == 0


def test_to_hist_preserves_declared_categories_without_chunks():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a", "b"], growth=False, name="cat"),
    )
    dense = h.to_hist()
    assert list(dense.axes[1]) == ["a", "b"]
    assert not dense.axes[1].traits.growth


def test_from_hist_basic():
    source = hist.Hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], growth=True, name="cat"),
    )
    source.fill(x=[0.2, 0.4], cat=["a", "a"])
    chunked = ChunkedHist.from_hist(source)
    assert len(chunked) == 1
    dense = chunked.to_hist()
    assert dense[{"x": bh.loc(0.2), "cat": "a"}] == 1
    assert dense[{"x": bh.loc(0.4), "cat": "a"}] == 1


def test_from_hist_multiple_categories():
    source = hist.Hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a", "b"], growth=True, name="cat"),
    )
    source.fill(x=[0.1, 0.2, 0.3], cat=["a", "a", "b"])
    chunked = ChunkedHist.from_hist(source)
    assert len(chunked) == 2
    dense = chunked.to_hist()
    assert dense[{"x": bh.loc(0.1), "cat": "a"}] == 1
    assert dense[{"x": bh.loc(0.2), "cat": "a"}] == 1
    assert dense[{"x": bh.loc(0.3), "cat": "b"}] == 1


def test_round_trip():
    chunked = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    chunked.fill(x=[0.2, 0.4], cat="a")
    chunked.fill(x=[0.3], cat="b")

    dense = chunked.to_hist()
    recovered = ChunkedHist.from_hist(dense)
    assert len(recovered) == len(chunked)
    for key in chunked:
        np.testing.assert_array_equal(
            recovered.chunk_view({"cat": key[0]}),
            chunked.chunk_view({"cat": key[0]}),
        )


def test_from_hist_categorical_flow_content_raises():
    source = hist.Hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], name="cat"),  # growth=False, has overflow
    )
    source.fill(x=[0.2, 0.4], cat=["a", "z"])  # "z" lands in the flow bin
    with pytest.raises(ValueError, match="flow bin of categorical axis 'cat'"):
        ChunkedHist.from_hist(source)


def test_from_hist_empty_flow_ok():
    source = hist.Hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], name="cat"),
    )
    source.fill(x=[0.2], cat=["a"])
    chunked = ChunkedHist.from_hist(source)
    assert chunked.to_hist()[{"cat": "a", "x": bh.loc(0.2)}] == 1


def test_iadd_unknown_key_non_growing_is_atomic():
    left = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], name="cat"),
    )
    right = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a", "z"], name="cat"),
    )
    left.fill(x=[0.2], cat="a")
    right.fill(x=[0.4], cat="a")
    right.fill(x=[0.5], cat="z")

    with pytest.raises(ValueError, match="non-growing chunk axis"):
        left += right
    # No partial merge: left is untouched
    assert list(left.keys()) == [("a",)]
    assert left.to_hist()[{"cat": "a", "x": bh.loc(0.4)}] == 0


def test_from_hist_no_chunk_axes():
    source = hist.Hist(axis.Regular(5, 0, 1, name="x"))
    source.fill(x=[0.1, 0.2, 0.3])
    chunked = ChunkedHist.from_hist(source)
    assert len(chunked) == 1
    dense = chunked.to_hist()
    assert np.sum(dense.values()) == 3


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def test_iadd_chunked():
    left = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    right = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    left.fill(x=[0.2], cat="a")
    right.fill(x=[0.4], cat="a")
    right.fill(x=[0.5], cat="b")

    left += right
    assert len(left) == 2
    dense = left.to_hist()
    assert dense[{"x": bh.loc(0.2), "cat": "a"}] == 1
    assert dense[{"x": bh.loc(0.4), "cat": "a"}] == 1
    assert dense[{"x": bh.loc(0.5), "cat": "b"}] == 1


def test_iadd_chunked_does_not_alias_source_chunk_arrays():
    left = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    right = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    right.fill(x=[0.5], cat="b")

    left += right
    right.chunk_view({"cat": "b"})[...] = 0

    assert left.to_hist()[{"x": bh.loc(0.5), "cat": "b"}] == 1


def test_add_chunked():
    left = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    right = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    left.fill(x=[0.2], cat="a")
    right.fill(x=[0.4], cat="a")

    merged = left + right
    assert len(merged) == 1
    dense = merged.to_hist()
    assert dense[{"x": bh.loc(0.2), "cat": "a"}] == 1
    assert dense[{"x": bh.loc(0.4), "cat": "a"}] == 1


def test_iadd_hist():
    chunked = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    dense = hist.Hist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], growth=True, name="cat"),
    )
    chunked.fill(x=[0.2], cat="a")
    dense.fill(x=[0.4], cat="a")

    chunked += dense
    result = chunked.to_hist()
    assert result[{"x": bh.loc(0.2), "cat": "a"}] == 1
    assert result[{"x": bh.loc(0.4), "cat": "a"}] == 1


def test_add_incompatible_raises():
    left = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    right = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="different"),
    )
    with pytest.raises(ValueError, match="incompatible chunk axes"):
        left + right


# ---------------------------------------------------------------------------
# Selection (__getitem__)
# ---------------------------------------------------------------------------
def test_getitem_single_value():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    h.fill(x=[0.4], cat="b")

    only_a = h[{"cat": "a"}]
    assert len(only_a) == 1
    assert ("a",) in only_a
    dense = only_a.to_hist()
    assert dense[{"x": bh.loc(0.2), "cat": "a"}] == 1


def test_getitem_multiple_values():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    h.fill(x=[0.4], cat="b")
    h.fill(x=[0.6], cat="c")

    ab = h[{"cat": ("a", "b")}]
    assert len(ab) == 2
    assert ("a",) in ab
    assert ("b",) in ab
    assert ("c",) not in ab


def test_getitem_wildcard():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="apple")
    h.fill(x=[0.4], cat="apricot")
    h.fill(x=[0.6], cat="banana")

    selected = h[{"cat": "ap*"}]
    assert len(selected) == 2


def test_getitem_wildcard_question():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a1")
    h.fill(x=[0.4], cat="a2")
    h.fill(x=[0.6], cat="b1")

    selected = h[{"cat": "a?"}]
    assert len(selected) == 2


def test_getitem_wildcard_no_match_returns_empty():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="apple")
    selected = h[{"cat": "z*"}]
    assert len(selected) == 0


def test_getitem_non_chunk_axis_raises():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    with pytest.raises(ValueError, match="dense"):
        h[{"x": 1}]


def test_getitem_unknown_axis_raises():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    with pytest.raises(KeyError, match="unknown axis"):
        h[{"foo": "a"}]


def test_getitem_empty_result():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    empty = h[{"cat": "nonexistent"}]
    assert len(empty) == 0


def test_getitem_multiple_chunk_axes():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="s"),
        axis.IntCategory([], growth=True, name="i"),
    )
    h.fill(x=[0.2], s="a", i=1)
    h.fill(x=[0.4], s="b", i=2)
    h.fill(x=[0.6], s="a", i=2)

    selected = h[{"s": "a"}]
    assert len(selected) == 2
    dense = selected.to_hist()
    assert dense[{"s": "a", "i": bh.loc(1), "x": bh.loc(0.2)}] == 1
    assert dense[{"s": "a", "i": bh.loc(2), "x": bh.loc(0.6)}] == 1


def test_getitem_numpy_scalar_selection():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.IntCategory([], growth=True, name="icat"),
    )
    h.fill(x=[0.2], icat=42)
    selected = h[{"icat": np.int64(42)}]
    assert len(selected) == 1
    assert (42,) in selected


# ---------------------------------------------------------------------------
# Chunk access
# ---------------------------------------------------------------------------
def test_chunk_view():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2, 0.4], cat="a")

    view = h.chunk_view({"cat": "a"})
    assert isinstance(view, np.ndarray)
    # The scratch dense hist does not include chunk axes
    assert view.ndim == 1
    assert view.shape[0] == 12  # 10 bins + 2 flow


def test_chunk_view_missing():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    with pytest.raises(KeyError, match="not found"):
        h.chunk_view({"cat": "missing"})


def test_keys_len_contains():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    h.fill(x=[0.4], cat="b")

    assert len(h) == 2
    keys_list = list(h.keys())
    assert ("a",) in keys_list
    assert ("b",) in keys_list
    assert ("a",) in h
    assert ("z",) not in h


def test_items():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    items_list = list(h.items())
    assert len(items_list) == 1
    key, view = items_list[0]
    assert key == ("a",)
    assert isinstance(view, np.ndarray)
    # items() yields live views, same as chunk_view()
    assert view is h.chunk_view({"cat": "a"})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def test_empty_like():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    empty = h.empty_like()
    assert len(empty) == 0
    assert empty.axes == h.axes


def test_empty_like_preserves_subclass():
    class SubHist(ChunkedHist):
        pass

    h = SubHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    assert type(h.empty_like()) is SubHist
    assert type(h[{"cat": "a"}]) is SubHist


def test_reset():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    assert len(h) == 1
    h.reset()
    assert len(h) == 0
    assert ("a",) not in h


def test_histogram_bytes():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    assert h.histogram_bytes() == 0
    h.fill(x=[0.2], cat="a")
    assert h.histogram_bytes() > 0


def test_repr():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    r = repr(h)
    assert "ChunkedHist" in r
    assert "cat" in r
    assert "Regular" in r
    assert "Chunks" in r


def test_repr_large_sizes():
    h = ChunkedHist(
        axis.Regular(300_000, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")  # ~2.4 MB chunk
    assert "MB" in repr(h)


def test_repr_respects_axis_growth():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory(["a"], growth=False, name="cat"),
    )
    assert "growth=True, name='cat'" not in repr(h)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
def test_dense_view_shape_dtype():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(5, 0, 1, name="y"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    assert h.dense_view_shape == (12, 7)  # +2 flow on each
    assert h.dense_view_dtype == np.dtype("float64")


def test_dense_axes():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    dense = h.dense_axes
    assert len(dense) == 1
    assert dense[0].name == "x"


# ---------------------------------------------------------------------------
# Exact chunk key helpers
# ---------------------------------------------------------------------------
def test_exact_chunk_key():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    key = h.exact_chunk_key({"cat": "a"})
    assert key == ("a",)


def test_selection_dict():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    d = h.selection_dict(("a",))
    assert d == {"cat": "a"}


def test_exact_chunk_key_multiple_values_raises():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    with pytest.raises(ValueError, match="exactly one value"):
        h.exact_chunk_key({"cat": ("a", "b")})


# ---------------------------------------------------------------------------
# Copy / pickle-like behavior
# ---------------------------------------------------------------------------
def test_to_hist_copy_independence():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    dense1 = h.to_hist()
    dense2 = h.to_hist()
    # Mutating one should not affect the other
    dense2[{"cat": "a", "x": 0}] = 99
    assert dense1[{"cat": "a", "x": 0}] == 0


def test_iadd_modifies_in_place():
    h = ChunkedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.StrCategory([], growth=True, name="cat"),
    )
    h.fill(x=[0.2], cat="a")
    original_id = id(h)
    h += h.empty_like()
    assert id(h) == original_id


# ---------------------------------------------------------------------------
# str/int normalization
# ---------------------------------------------------------------------------
def test_int_category_with_numpy_scalar():
    h = ChunkedHist(
        axis.Regular(5, 0, 1, name="x"),
        axis.IntCategory([], growth=True, name="icat"),
    )
    # numpy scalar int should be accepted as chunk key
    h.fill(x=[0.1], icat=np.int64(42))
    assert len(h) == 1
    assert (42,) in h


def test_bool_chunk_key_normalizes_to_int():
    h = ChunkedHist(
        axis.Regular(5, 0, 1, name="x"),
        axis.IntCategory([], growth=True, name="icat"),
    )
    h.fill(x=[0.1], icat=True)
    h.fill(x=[0.2], icat=np.True_)
    assert list(h.keys()) == [(1,)]
    key = next(iter(h.keys()))[0]
    assert type(key) is int
