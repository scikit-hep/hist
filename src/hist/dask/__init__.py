from __future__ import annotations

import importlib.util

if not importlib.util.find_spec("dask_histogram"):
    raise ModuleNotFoundError(
        """for hist.dask, install the 'dask_histogram' package with:
        pip install dask_histogram
        or
        conda install dask_histogram"""
    )

from .hist import Hist
from .namedhist import NamedHist

new = Hist.new

__all__ = ["Hist", "NamedHist", "new"]
