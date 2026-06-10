from __future__ import annotations

from typing import Generic, TypeVar

import boost_histogram as bh
import graphed_histogram.boost as ghb

import hist

from ..namedhist import NamedHist as NamedHistInMemory

S = TypeVar("S", bound=bh.storage.Storage)


class NamedHist(NamedHistInMemory[S], ghb.Histogram, Generic[S], family=hist):  # type: ignore[misc]
    @property
    def _in_memory_type(self) -> type[NamedHistInMemory[S]]:
        return NamedHistInMemory
