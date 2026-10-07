from __future__ import annotations

__lazy_modules__ = {"boost_histogram.tag"}

from boost_histogram.tag import (
    Locator,
    Slicer,
    at,
    loc,
    overflow,
    rebin,
    sum,
    underflow,
)

__all__ = ("Locator", "Slicer", "at", "loc", "overflow", "rebin", "sum", "underflow")
