from __future__ import annotations

__lazy_modules__ = {"boost_histogram.accumulators"}

from boost_histogram.accumulators import Mean, Sum, WeightedMean, WeightedSum

__all__ = ("Mean", "Sum", "WeightedMean", "WeightedSum")
