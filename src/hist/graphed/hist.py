from __future__ import annotations

from typing import Generic, TypeVar

import boost_histogram as bh
import graphed_histogram.boost as ghb

import hist

from ..hist import Hist as HistInMemory

S = TypeVar("S", bound=bh.storage.Storage)


class Hist(HistInMemory[S], ghb.Histogram, Generic[S], family=hist):  # type: ignore[misc]
    """A `hist.Hist` whose fills are DEFERRED graphed computations (the `hist.dask` analogue):
    QuickConstruct (`Hist.new.Reg(...).Double()`) and named-axis fills record into the graphed
    IR; `.compute()` returns a concrete in-memory `hist.Hist`."""

    @property
    def _in_memory_type(self) -> type[HistInMemory[S]]:
        return HistInMemory
