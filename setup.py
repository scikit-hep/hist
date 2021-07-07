#!/usr/bin/env python
# Copyright (c) 2019-2020, Eduardo Rodrigues and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hist for details.

from setuptools import setup

extras_require = {
    "plot": [
        "matplotlib >=3.0",
        "scipy >=1.4",
        "iminuit >=2",
        "mplhep >=0.2.16",
    ]
}


extras_require["test"] = [
    *extras_require["plot"],
    "pytest >=6",
    "pytest-mpl >=0.12",
]

extras_require["dev"] = [*extras_require["test"], *extras_require["plot"], "ipykernel"]

extras_require["docs"] = [
    *extras_require["plot"],
    "nbsphinx",
    "Sphinx >=3.0.0",
    "sphinx_copybutton",
    "sphinx_rtd_theme >=0.5.0",
    "sphinx_book_theme >=0.0.38",
    "ipython",
    "ipykernel",
    "pillow",
    "uncertainties>=3",
    "myst_parser>=0.14",
]

extras_require["all"] = sorted(set(sum(extras_require.values(), [])))

setup(extras_require=extras_require)
