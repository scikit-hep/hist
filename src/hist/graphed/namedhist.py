from __future__ import annotations

from typing import Generic, TypeVar

import boost_histogram as bh
import graphed_histogram.boost as ghb

import hist

from ..namedhist import NamedHist as NamedHistInMemory
from .hist import FillModeMixin

S = TypeVar("S", bound=bh.storage.Storage)


class NamedHist(  # type: ignore[misc]
    FillModeMixin,
    NamedHistInMemory[S],
    ghb.Histogram,  # type: ignore[misc]
    Generic[S],
    family=hist,
):
    pass
