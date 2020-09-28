# -*- coding: utf-8 -*-
import pytest
from hist import Hist, NamedHist
from hist.hist import BaseHist


@pytest.fixture(params=[Hist, BaseHist, NamedHist])
def named_hist(request):
    yield request.param


@pytest.fixture(params=[Hist, BaseHist])
def unnamed_hist(request):
    yield request.param
