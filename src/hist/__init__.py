# -*- coding: utf-8 -*-
# Copyright (c) 2020, Henry Schreiner
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hist for details.

# Convenient access to the version number
from .version import version as __version__

from . import axis, numpy, tag, accumulators, utils, storage

from .hist import Hist
from .namedhist import NamedHist
from .basehist import BaseHist

__all__ = (
    "__version__",
    "axis",
    "Hist",
    "NamedHist",
    "BaseHist",
    "numpy",
    "tag",
    "accumulators",
    "utils",
    "storage",
)
