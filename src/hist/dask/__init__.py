from __future__ import annotations

try:
    import dask_histogram  # noqa: F401
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        """for hist.dask, install the 'dask_histogram' package with:
        pip install dask_histogram
        or
        conda install dask_histogram"""
    ) from err

from .hist import Hist
from .namedhist import NamedHist

__all__ = ["Hist", "NamedHist"]
