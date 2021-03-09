# Copyright (c) 2020, Henry Schreiner
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hist for details.

from types import ModuleType

from . import accumulators, axis, numpy, storage, tag
from .basehist import BaseHist
from .hist import Hist
from .namedhist import NamedHist
from .tag import loc, overflow, rebin, sum, underflow

# Convenient access to the version number
# Convenient access to the version number
from .version import version as __version__

__all__ = (
    "__version__",
    "Hist",
    "BaseHist",
    "NamedHist",
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
)


# Python 3.7 only
def __getattr__(name: str) -> ModuleType:
    import warnings

    if name == "axes":
        msg = f"Misspelling error, '{name}' should be 'axis'"
        warnings.warn(msg)
        return axis
    raise AttributeError(f"module {__name__} has no attribute {name}")
