from __future__ import annotations

from typing import Generic, TypeVar

import boost_histogram as bh
import graphed_histogram.boost as ghb

import hist

from ..hist import Hist as HistInMemory

S = TypeVar("S", bound=bh.storage.Storage)


class Hist(HistInMemory[S], ghb.Histogram, Generic[S], family=hist):  # type: ignore[misc]
    """A `hist.Hist` whose fills are DEFERRED graphed computations: QuickConstruct
    (`Hist.new.Reg(...).Double()`) and named-axis fills record into the graphed IR. Evaluation
    is graphed's own idiom — `plan()` + an R7 executor (whose result wraps back into an
    in-memory `hist.Hist` via `hist.Hist(value)`), or the reference `session.materialize` on a
    fill node."""
