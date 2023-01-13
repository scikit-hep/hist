from __future__ import annotations

import dask_histogram.boost as dhb

import hist

from ..hist import Hist as HistNoDask


class Hist(HistNoDask, dhb.Histogram, family=hist):  # type: ignore[misc]
    @property
    def _in_memory_type(self) -> type[HistNoDask]:
        return HistNoDask
