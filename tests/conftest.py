from __future__ import annotations

import pytest

from hist import Hist, NamedHist
from hist.hist import BaseHist

try:
    import hist.dask as dah
except ImportError:
    dah = None


@pytest.fixture(params=[Hist, BaseHist, NamedHist])
def named_hist(request):
    return request.param


@pytest.fixture(params=[Hist, BaseHist])
def unnamed_hist(request):
    return request.param


dask_params_named = []
dask_params_unnamed = []
if dah is not None:
    dask_params_named = [dah.Hist, dah.NamedHist]
    dask_params_unnamed = [dah.Hist]


@pytest.fixture(params=dask_params_named)
def named_dask_hist(request):
    return request.param


@pytest.fixture(params=dask_params_unnamed)
def unnamed_dask_hist(request):
    return request.param
