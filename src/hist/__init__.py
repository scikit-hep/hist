# Copyright (c) 2020, Henry Schreiner
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hist for details.

# Convenient access to the version number
from .version import version as __version__

from .general import Hist
from .named import NamedHist

from . import axis
