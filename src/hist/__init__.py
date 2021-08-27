# Copyright (c) 2020, Henry Schreiner
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hist for details.

from __future__ import annotations

import warnings
from types import ModuleType

from . import accumulators, axis, numpy, storage, tag
from .basehist import BaseHist
from .hist import Hist
from .namedhist import NamedHist
from .stack import Stack
from .tag import loc, overflow, rebin, sum, underflow

# Convenient access to the version number
from .version import version as __version__

__all__ = (
    "__version__",
    "Hist",
    "BaseHist",
    "NamedHist",
    "Stack",
    "accumulators",
    "axis",
    "loc",
    "numpy",
    "overflow",
    "rebin",
    "storage",
    "sum",
    "tag",
    "underflow",
    "new",
)


new = Hist.new


def __dir__() -> tuple[str, ...]:
    return __all__


def __getattr__(name: str) -> ModuleType:

    if name == "axes":
        msg = f"Misspelling error, '{name}' should be 'axis'"
        warnings.warn(msg)
        return axis
    raise AttributeError(f"module {__name__} has no attribute {name}")
