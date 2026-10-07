from __future__ import annotations

__lazy_modules__ = {"boost_histogram.storage"}

from boost_histogram.storage import (
    AtomicInt64,
    Double,
    Int64,
    Mean,
    MultiCell,
    Storage,
    Unlimited,
    Weight,
    WeightedMean,
)

__all__ = [
    "AtomicInt64",
    "Double",
    "Int64",
    "Mean",
    "MultiCell",
    "Storage",
    "Unlimited",
    "Weight",
    "WeightedMean",
]
