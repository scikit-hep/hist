from __future__ import annotations

import numpy as np
import pytest

import hist

ak = pytest.importorskip("awkward")
da = pytest.importorskip("dask.array")
dah = pytest.importorskip("hist.dask")
dak = pytest.importorskip("dask_awkward")


@pytest.mark.parametrize("pass_kwargs", [False, True])
@pytest.mark.parametrize("unknown_divisions", [False, True])
@pytest.mark.parametrize("npartitions", [1, 2, 5])
@pytest.mark.parametrize("use_weights", [True, False])
def test_simple_1D_dask_awkward(
    unnamed_dask_hist,
    use_weights,
    npartitions,
    unknown_divisions,
    pass_kwargs,
):
    x = ak.Array(np.random.standard_normal(size=1000))
    x = dak.from_awkward(x, npartitions=npartitions)
    if unknown_divisions:
        x.clear_divisions()
    if use_weights:
        weights = ak.Array(np.random.uniform(0.5, 0.75, size=1000))
        weights = dak.from_awkward(weights, npartitions=npartitions)
        if unknown_divisions:
            weights.clear_divisions()
        storage = hist.storage.Weight()
    else:
        weights = None
        storage = hist.storage.Double()

    h = unnamed_dask_hist(
        hist.axis.Regular(10, -4.0, 4.0, name="x"),
        storage=storage,
    )
    if pass_kwargs:
        h.fill(x=x, weight=weights)
    else:
        h.fill(x, weight=weights)
    h = h.compute()

    control = h.__class__(*h.axes, storage=h.storage_type())
    xc = x.compute()
    if use_weights:
        wc = weights.compute()
        if pass_kwargs:
            control.fill(x=xc, weight=wc)
        else:
            control.fill(xc, weight=wc)
    else:
        if pass_kwargs:
            control.fill(x=xc)
        else:
            control.fill(xc)

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


@pytest.mark.parametrize("pass_kwargs", [False, True])
@pytest.mark.parametrize("use_weights", [True, False])
def test_unnamed_5D_strcat_intcat_rectangular(
    unnamed_dask_hist, use_weights, pass_kwargs
):
    x = da.random.standard_normal(size=(2000, 3), chunks=(400, 3))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape[0], chunks=x.chunksize[0])
        storage = hist.storage.Weight()
    else:
        weights = None
        storage = hist.storage.Double()

    h = unnamed_dask_hist(
        hist.axis.StrCategory([], growth=True, name="strcat"),
        hist.axis.IntCategory([], growth=True, name="intcat"),
        hist.axis.Regular(8, -3.5, 3.5, name="x"),
        hist.axis.Regular(7, -3.3, 3.3, name="y"),
        hist.axis.Regular(9, -3.2, 3.2, name="z"),
        storage=storage,
    )
    xT = x.T
    if pass_kwargs:
        h.fill(strcat="testcat1", intcat=1, x=xT[0], y=xT[1], z=xT[2], weight=weights)
        h.fill(strcat="testcat2", intcat=2, x=xT[0], y=xT[1], z=xT[2], weight=weights)
    else:
        h.fill("testcat1", 1, xT[0], xT[1], xT[2], weight=weights)
        h.fill("testcat2", 2, xT[0], xT[1], xT[2], weight=weights)
    h = h.compute()

    control = h.__class__(*h.axes, storage=h.storage_type())
    xTc = x.compute().T
    if use_weights:
        if pass_kwargs:
            control.fill(
                strcat="testcat1",
                intcat=1,
                x=xTc[0],
                y=xTc[1],
                z=xTc[2],
                weight=weights.compute(),
            )
            control.fill(
                strcat="testcat2",
                intcat=2,
                x=xTc[0],
                y=xTc[1],
                z=xTc[2],
                weight=weights.compute(),
            )
        else:
            control.fill(
                "testcat1",
                1,
                xTc[0],
                xTc[1],
                xTc[2],
                weight=weights.compute(),
            )
            control.fill(
                "testcat2",
                2,
                xTc[0],
                xTc[1],
                xTc[2],
                weight=weights.compute(),
            )
    else:
        if pass_kwargs:
            control.fill(strcat="testcat1", intcat=1, x=xTc[0], y=xTc[1], z=xTc[2])
            control.fill(strcat="testcat2", intcat=2, x=xTc[0], y=xTc[1], z=xTc[2])
        else:
            control.fill("testcat1", 1, xTc[0], xTc[1], xTc[2])
            control.fill("testcat2", 2, xTc[0], xTc[1], xTc[2])

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())

    assert len(h.axes[0]) == 2
    assert len(control.axes[0]) == 2
    assert all(cx == hx for cx, hx in zip(control.axes[0], h.axes[0]))

    assert len(h.axes[1]) == 2
    assert len(control.axes[1]) == 2
    assert all(cx == hx for cx, hx in zip(control.axes[1], h.axes[1]))


@pytest.mark.parametrize("use_weights", [True, False])
def test_named_5D_strcat_intcat_rectangular(named_dask_hist, use_weights):
    x = da.random.standard_normal(size=(2000, 3), chunks=(400, 3))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape[0], chunks=x.chunksize[0])
        storage = hist.storage.Weight()
    else:
        weights = None
        storage = hist.storage.Double()

    h = named_dask_hist(
        hist.axis.StrCategory([], growth=True, name="strcat"),
        hist.axis.IntCategory([], growth=True, name="intcat"),
        hist.axis.Regular(8, -3.5, 3.5, name="x"),
        hist.axis.Regular(7, -3.3, 3.3, name="y"),
        hist.axis.Regular(9, -3.2, 3.2, name="z"),
        storage=storage,
    )
    xT = x.T
    h.fill(strcat="testcat1", intcat=1, x=xT[0], y=xT[1], z=xT[2], weight=weights)
    h.fill(strcat="testcat2", intcat=2, x=xT[0], y=xT[1], z=xT[2], weight=weights)
    h = h.compute()

    control = h.__class__(*h.axes, storage=h.storage_type())
    xTc = x.compute().T
    if use_weights:
        control.fill(
            strcat="testcat1",
            intcat=1,
            x=xTc[0],
            y=xTc[1],
            z=xTc[2],
            weight=weights.compute(),
        )
        control.fill(
            strcat="testcat2",
            intcat=2,
            x=xTc[0],
            y=xTc[1],
            z=xTc[2],
            weight=weights.compute(),
        )
    else:
        control.fill(strcat="testcat1", intcat=1, x=xTc[0], y=xTc[1], z=xTc[2])
        control.fill(strcat="testcat2", intcat=2, x=xTc[0], y=xTc[1], z=xTc[2])

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())

    assert len(h.axes[0]) == 2
    assert len(control.axes[0]) == 2
    assert all(cx == hx for cx, hx in zip(control.axes[0], h.axes[0]))

    assert len(h.axes[1]) == 2
    assert len(control.axes[1]) == 2
    assert all(cx == hx for cx, hx in zip(control.axes[1], h.axes[1]))
