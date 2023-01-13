from __future__ import annotations

import dask_histogram.boost as dhb

import hist

from ..namedhist import NamedHist as NamedHistNoDask


class NamedHist(NamedHistNoDask, dhb.Histogram, family=hist):  # type: ignore[misc]
    @property
    def _in_memory_type(self) -> type[NamedHistNoDask]:
        return NamedHistNoDask
