from __future__ import annotations

__lazy_modules__ = {"boost_histogram.numpy"}

from boost_histogram.numpy import histogram, histogram2d, histogramdd

__all__ = ("histogram", "histogram2d", "histogramdd")
