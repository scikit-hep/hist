from __future__ import annotations

from typing import Generic, TypeVar

import boost_histogram as bh
import dask_histogram.boost as dhb

import hist

from ..hist import Hist as HistNoDask

S = TypeVar("S", bound=bh.storage.Storage)


class Hist(HistNoDask[S], dhb.Histogram, Generic[S], family=hist):  # type: ignore[misc]
    @property
    def _in_memory_type(self) -> type[HistNoDask[S]]:
        return HistNoDask
