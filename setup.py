#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Eduardo Rodrigues and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hist for details.

from setuptools import setup

extras_require = {}

extras_require["plot"] = [
    "matplotlib >=3.0",
    "scipy >=1.4",
    "uncertainties >=3",
    "mplhep >=0.2.2",
]

extras_require["test"] = [
    "pytest >=4.6",
]

extras_require["dev"] = [*extras_require["test"], *extras_require["plot"], "ipykernel"]

extras_require["docs"] = [
    *extras_require["plot"],
    "nbsphinx",
    "recommonmark >=0.5.0",
    "Sphinx >=3.0.0",
    "sphinx_copybutton",
    "sphinx_rtd_theme >=0.5.0",
    "ipython",
]

extras_require["all"] = sorted(set(sum(extras_require.values(), [])))

setup(extras_require=extras_require)
