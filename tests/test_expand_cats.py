from __future__ import annotations

__lazy_modules__ = {"hist", "itertools"}

import itertools

import pytest

from hist import Hist, axis, storage


def make_hist(first, second, counts):
    h = Hist(
        axis.StrCategory(first, name="first"),
        axis.StrCategory(second, name="second"),
        axis.Regular(1, 0, 1, name="x"),
        storage=storage.Int64(),
        name="source",
        label="original",
    )
    for (a, b), count in zip(itertools.product(first, second), counts, strict=True):
        h.fill(first=[a] * count, second=[b] * count, x=[0.5] * count)
    return h


@pytest.mark.parametrize("counts", [[2, 3, 5, 7], [0, 0, 0, 0], [0, 3, 5, 7]])
def test_default_name_collision(counts):
    h = make_hist(["a_b", "a"], ["c", "b_c"], counts)
    original = h.copy()

    with pytest.raises(ValueError, match=r"(?i)duplicate.*a_b_c"):
        h.expand_cats()

    assert h == original
    assert h.name == "source"
    assert h.label == "original"


@pytest.mark.parametrize("counts", [[2, 3, 5, 7], [0, 0, 0, 0], [0, 3, 5, 7]])
def test_custom_name_collision(counts):
    h = make_hist(["a_b", "a"], ["c", "b_c"], counts)
    original = h.copy()

    # Removing one category value from the key also loses whole combinations.
    with pytest.raises(ValueError, match=r"(?i)duplicate.*a_b"):
        h.expand_cats(name=lambda first, _second: first)

    assert h == original
    assert h.name == "source"
    assert h.label == "original"


@pytest.mark.parametrize("key", ["sample", ""])
def test_single_int_category_custom_collision(key):
    h = Hist(axis.IntCategory([4, 1], name="n"), axis.Regular(1, 0, 1))
    original = h.copy()

    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        h.expand_cats(name=lambda *cats: key)

    assert h == original


@pytest.mark.parametrize(
    ("first", "second", "keys"),
    [
        (["A", "B"], ["C", "D"], ["A_C", "A_D", "B_C", "B_D"]),
        (["a_b", "a"], ["c", "d"], ["a_b_c", "a_b_d", "a_c", "a_d"]),
    ],
)
def test_unique_default_names_with_underscores(first, second, keys):
    counts = [2, 3, 5, 7]
    h = make_hist(first, second, counts)
    original = h.copy()
    result = h.expand_cats()

    assert list(result) == keys
    for (key, group), count in zip(result.items(), counts, strict=True):
        assert group.name == key
        assert group.axes == (h.axes[2],)
        assert group.values().tolist() == [count]
    assert sum(group.sum() for group in result.values()) == 17
    assert h == original
    assert h.name == "source"
    assert h.label == "original"


def test_unique_empty_custom_name():
    h = Hist(axis.StrCategory(["a_b"], name="s"), axis.Regular(1, 0, 1))
    original = h.copy()
    result = h.expand_cats(name=lambda *cats: "")

    assert list(result) == [""]
    assert result[""].name == ""
    assert result[""].axes == (h.axes[1],)
    assert result[""].sum() == 0
    assert h == original


def test_unique_custom_names_preserve_all_groups():
    counts = [2, 3, 5, 7]
    h = make_hist(["a_b", "a"], ["c", "b_c"], counts)
    original = h.copy()
    result = h.expand_cats(name=lambda *cats: repr(cats))

    assert list(result) == [
        "('a_b', 'c')",
        "('a_b', 'b_c')",
        "('a', 'c')",
        "('a', 'b_c')",
    ]
    for (key, group), count in zip(result.items(), counts, strict=True):
        assert group.name == key
        assert group.axes == (h.axes[2],)
        assert group.values().tolist() == [count]
    assert sum(group.sum() for group in result.values()) == 17
    assert h == original
    assert h.name == "source"
    assert h.label == "original"


@pytest.mark.parametrize("int_first", [True, False])
def test_category_values_in_axis_order_and_empty_groups(int_first):
    integer = axis.IntCategory([4, 1], name="n")
    string = axis.StrCategory(["a_b", "c"], name="s")
    first, second = (integer, string) if int_first else (string, integer)
    h = Hist(
        axis.Boolean(name="flag"),
        first,
        axis.Regular(2, 0, 1, name="x"),
        second,
        storage=storage.Int64(),
    )
    h.fill(
        flag=[False, True, True], n=[4, 1, 1], s=["a_b", "c", "c"], x=[0.2, 0.7, 0.7]
    )
    original = h.copy()
    calls = []

    def name(*cats):
        calls.append(cats)
        return repr(cats)

    result = h.expand_cats(name=name)
    expected = (
        [(4, "a_b"), (4, "c"), (1, "a_b"), (1, "c")]
        if int_first
        else [("a_b", 4), ("a_b", 1), ("c", 4), ("c", 1)]
    )
    assert calls == expected
    assert list(result) == [repr(cats) for cats in expected]
    for key, group in result.items():
        assert group.name == key
        assert group.axes == (h.axes[0], h.axes[2])
    assert result[repr((4, "a_b") if int_first else ("a_b", 4))].values().tolist() == [
        [1, 0],
        [0, 0],
    ]
    assert result[repr((1, "c") if int_first else ("c", 1))].values().tolist() == [
        [0, 0],
        [0, 2],
    ]
    assert sorted(group.sum() for group in result.values()) == [0, 0, 1, 2]
    assert h == original


def test_empty_category_axis_has_no_combinations():
    h = Hist(axis.StrCategory([], name="s"), axis.IntCategory([4, 1], name="n"))
    original = h.copy()
    calls = []

    def name(*cats):
        calls.append(cats)
        return "sample"

    assert h.expand_cats() == {}
    assert h.expand_cats(name=name) == {}
    assert calls == []
    assert h == original


def test_no_category_axes_still_raise():
    h = Hist(axis.Regular(1, 0, 1), axis.Boolean())
    with pytest.raises(ValueError, match="without categorical axes"):
        h.expand_cats()
