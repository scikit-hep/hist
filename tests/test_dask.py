from __future__ import annotations

import numpy as np
import pytest

import hist

da = pytest.importorskip("dask.array")
dah = pytest.importorskip("hist.dask")


@pytest.mark.parametrize("use_weights", [True, False])
def test_unnamed_5D_strcat_intcat_rectangular(unnamed_dask_hist, use_weights):
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
