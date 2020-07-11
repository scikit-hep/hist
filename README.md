# Hist

[![Github Actions badge](https://github.com/scikit-hep/hist/workflows/CI/badge.svg)](https://github.com/scikit-hep/hist/actions)
[![Join the chat at https://gitter.im/Scikit-HEP/hist](https://badges.gitter.im/HSF/PyHEP-histogramming.svg)](https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Development for Hist. See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for information on setting up a development environment.

Hist is a analyst friendly front-end for [boost-histogram][].

Hist augments metadata by adding names to axes; these are *highly* recommend and will help you track axes. There is also a special `NamedHist`, which will enforce all hist axes have names, and all axes will require named access.

* `name=` is a unique label describing each axis
* `title=` is an optional string that is used in plotting (defaults to name if not provided)
* Indexing, projection, and more support named axes.

The `Hist` class augments the `bh.Histogram` class with the following shortcuts, designed for interactive exploration without extensive imports:

* Optional import-free construction system
* Quick import-free data-coordinates and rebin syntax (use a j suffix for numbers, or strings directly in indexing expressions)

Quick plotting routines encourage exploration:

* `.plot()` provides 1D and 2D plots
* `.plot2d_full()` shows 1D projects around a 2D plot
* `.plot_pull(...)` performs a pull plot



[boost-histogram]: https://github.com/scikit-hep/boost-histogram

---

Support for this work was provided by the National Science Foundation cooperative agreement OAC-1836650 (IRIS-HEP) and OAC-1450377 (DIANA/HEP). Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
