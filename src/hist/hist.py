from __future__ import annotations

from typing import Generic, TypeVar

import boost_histogram as bh

import hist

from .basehist import BaseHist

S = TypeVar("S", bound=bh.storage.Storage)


class Hist(BaseHist[S], Generic[S], family=hist):
    pass
