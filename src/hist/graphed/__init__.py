from __future__ import annotations

import importlib.util

if not importlib.util.find_spec("graphed_histogram"):
    msg = """for hist.graphed, install the 'graphed_histogram' package with:
        pip install graphed_histogram"""
    raise ModuleNotFoundError(msg)

from .hist import Hist
from .namedhist import NamedHist

new = Hist.new

__all__ = ["Hist", "NamedHist", "new"]
