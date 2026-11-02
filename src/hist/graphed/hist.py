from __future__ import annotations

from typing import Any, Generic, TypeVar

import boost_histogram as bh
import graphed_histogram.boost as ghb

import hist

from ..hist import Hist as HistInMemory

S = TypeVar("S", bound=bh.storage.Storage)


class FillModeMixin:
    """Routes graphed's fill-MODE flags past hist's named-axis fill.

    `BaseHist.fill` resolves every keyword as an axis name, so `variation_axis=`/`unweighted=`
    die at `_name_to_index` before reaching `graphed_histogram.boost.Histogram.fill`. When one is
    set, resolve the axis names here and call graphed's fill directly; otherwise defer to hist
    unchanged.
    """

    def fill(
        self,
        *args: Any,
        weight: Any = None,
        sample: Any = None,
        threads: int | None = None,
        unweighted: bool = False,
        variation_axis: bool = False,
        **kwargs: Any,
    ) -> Any:
        base: Any = self
        if not (unweighted or variation_axis):
            return super().fill(  # type: ignore[misc]
                *args, weight=weight, sample=sample, threads=threads, **kwargs
            )
        by_index = {base._name_to_index(k): v for k, v in kwargs.items()}
        return ghb.Histogram.fill(
            base,
            *args,
            *(by_index[i] for i in sorted(by_index)),
            weight=weight,
            sample=sample,
            threads=threads,
            unweighted=unweighted,
            variation_axis=variation_axis,
        )

    # BaseHist reads these under simplefilter("ignore"): reading first lets an un-run
    # histogram's warning, or its error filter, reach the caller.
    def sort(self, *args: Any, **kwargs: Any) -> Any:
        self.view()  # type: ignore[attr-defined]
        return super().sort(*args, **kwargs)  # type: ignore[misc]

    def expand_cats(self, *args: Any, **kwargs: Any) -> Any:
        self.view()  # type: ignore[attr-defined]
        return super().expand_cats(*args, **kwargs)  # type: ignore[misc]

    def integrate(self, name: Any, i_or_list: Any = None, j: Any = None) -> Any:
        if isinstance(i_or_list, list):
            self.view()  # type: ignore[attr-defined]
        return super().integrate(name, i_or_list, j)  # type: ignore[misc]


class Hist(FillModeMixin, HistInMemory[S], ghb.Histogram, Generic[S], family=hist):  # type: ignore[misc]
    """A `hist.Hist` whose fills are DEFERRED graphed computations: QuickConstruct
    (`Hist.new.Reg(...).Double()`) and named-axis fills record into the graphed IR. Evaluation
    is graphed's own idiom — `plan()` + a graphed executor (whose result wraps back into an
    in-memory `hist.Hist` via `hist.Hist(value)`), or the reference `session.materialize` on a
    fill node."""
