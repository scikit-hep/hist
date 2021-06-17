<img alt="histogram" width="200" src="https://raw.githubusercontent.com/scikit-hep/hist/main/docs/_images/histlogo.png"/>

# Hist

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![Code style: black][black-badge]][black-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![DOI](https://zenodo.org/badge/239605861.svg)](https://zenodo.org/badge/latestdoi/239605861)

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link]
[![Scikit-HEP][sk-badge]](https://scikit-hep.org/)

Hist is an analyst-friendly front-end for
[boost-histogram](https://github.com/scikit-hep/boost-histogram), designed for
Python 3.7+ (3.6 users get version 2.3). See [what's new](https://hist.readthedocs.io/en/latest/changelog.html).

## Installation

You can install this library from [PyPI](https://pypi.org/project/hist/) with pip:

```bash
python3 -m pip install "hist[plot]"
```

If you do not need the plotting features, you can skip the `[plot]` extra.

## Features

Hist currently provides everything boost-histogram provides, and the following enhancements:

- Hist augments axes with names:
  - `name=` is a unique label describing each axis
  - `label=` is an optional string that is used in plotting (defaults to `name`
    if not provided)
  - Indexing, projection, and more support named axes
  - Experimental `NamedHist` is a `Hist` that disables most forms of positional access

- The `Hist` class augments `bh.Histogram` with reduced typing construction:
  - Optional import-free construction system
  - `flow=False` is a fast way to turn off flow for the axes on construction
  - Storages can be given by string
  - `storage=` can be omitted
  - `data=` can initialize a histogram with existing data
  - `Hist.from_columns` can be used to initialize with a DataFrame or dict

- Hist implements UHI+; an extension to the UHI (Unified Histogram Indexing) system designed for import-free interactivity:
  - Uses `j` suffix to switch to data coordinates in access or slices
  - Uses `j` suffix on slices to rebin
  - Strings can be used directly to index into string category axes

- Quick plotting routines encourage exploration:
  - `.plot()` provides 1D and 2D plots (or use `plot1d()`, `plot2d()`)
  - `.plot2d_full()` shows 1D projects around a 2D plot
  - `.plot_ratio(...)` make a ratio plot between the histogram and another histogram or callable
  - `.plot_pull(...)` performs a pull plot
  - `.plot_pie()` makes a pie plot
  - `.show()` provides a nice str printout using Histoprint

- Extended Histogram features:
  - `.density()` computes the density as an array
  - `.profile(remove_ax)` can convert a ND COUNT histogram into a (N-1)D MEAN histogram

- New modules
  - `intervals` supports frequentist coverage intervals

- Notebook ready: Hist has gorgeous in-notebook representation.
  - No dependencies required

## Usage

```python
from hist import Hist

# Quick construction, no other imports needed:
h = (
  Hist.new
  .Reg(10, 0 ,1, name="x", label="x-axis")
  .Var(range(10), name="y", label="y-axis")
  .Int64()
)

# Filling by names is allowed:
h.fill(y=[1, 4, 6], x=[3, 5, 2])

# Names can be used to manipulate the histogram:
h.project("x")
h[{"y": 0.5j + 3, "x": 5j}]

# You can access data coordinates or rebin with a `j` suffix:
h[.3j:, ::2j] # x from .3 to the end, y is rebinned by 2

# Elegant plotting functions:
h.plot()
h.plot2d_full()
h.plot_pull(Callable)
```

## Development

From a git checkout, run:

```bash
python -m pip install -e .[dev]
```

See [CONTRIBUTING.md](https://hist.readthedocs.io/en/latest/contributing.html) for information on setting up a development environment.

## Contributors

We would like to acknowledge the contributors that made this project possible ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/henryiii"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="#maintenance-henryiii" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/hist/commits?author=henryiii" title="Code">💻</a> <a href="https://github.com/scikit-hep/hist/commits?author=henryiii" title="Documentation">📖</a></td>
    <td align="center"><a href="http://lovelybuggies.com.cn/"><img src="https://avatars3.githubusercontent.com/u/29083689?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nino Lau</b></sub></a><br /><a href="#maintenance-LovelyBuggies" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/hist/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/scikit-hep/hist/commits?author=LovelyBuggies" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/chrisburr"><img src="https://avatars3.githubusercontent.com/u/5220533?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Chris Burr</b></sub></a><br /><a href="https://github.com/scikit-hep/hist/commits?author=chrisburr" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/aminnj"><img src="https://avatars.githubusercontent.com/u/5760027?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nick Amin</b></sub></a><br /><a href="https://github.com/scikit-hep/hist/commits?author=aminnj" title="Code">💻</a></td>
    <td align="center"><a href="http://cern.ch/eduardo.rodrigues"><img src="https://avatars.githubusercontent.com/u/5013581?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Eduardo Rodrigues</b></sub></a><br /><a href="https://github.com/scikit-hep/hist/commits?author=eduardo-rodrigues" title="Code">💻</a></td>
    <td align="center"><a href="http://andrzejnovak.github.io/"><img src="https://avatars.githubusercontent.com/u/13226500?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrzej Novak</b></sub></a><br /><a href="https://github.com/scikit-hep/hist/commits?author=andrzejnovak" title="Code">💻</a></td>
    <td align="center"><a href="http://www.matthewfeickert.com/"><img src="https://avatars.githubusercontent.com/u/5142394?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Matthew Feickert</b></sub></a><br /><a href="https://github.com/scikit-hep/hist/commits?author=matthewfeickert" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://theoryandpractice.org"><img src="https://avatars.githubusercontent.com/u/4458890?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kyle Cranmer</b></sub></a><br /><a href="https://github.com/scikit-hep/hist/commits?author=cranmer" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.

## Talks

- [2020-07-07 SciPy Proceedings](https://www.youtube.com/watch?v=ERraTfHkPd0&list=PLYx7XA2nY5GfY4WWJjG5cQZDc7DIUmn6Z&index=4)
- [2020-07-17 PyHEP2020](https://indico.cern.ch/event/882824/contributions/3931299/)

---

## Acknowledgements

This library was primarily developed by Henry Schreiner and Nino Lau.

Support for this work was provided by the National Science Foundation cooperative agreement OAC-1836650 (IRIS-HEP) and OAC-1450377 (DIANA/HEP). Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

[actions-badge]:            https://github.com/scikit-hep/hist/workflows/CI/badge.svg
[actions-link]:             https://github.com/scikit-hep/hist/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/hist
[conda-link]:               https://github.com/conda-forge/hist-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/scikit-hep/hist/discussions
[gitter-badge]:             https://badges.gitter.im/HSF/PyHEP-histogramming.svg
[gitter-link]:              https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pre-commit-badge]:         https://results.pre-commit.ci/badge/github/scikit-hep/hist/main.svg
[pre-commit-link]:          https://results.pre-commit.ci/repo/github/scikit-hep/hist
[pypi-link]:                https://pypi.org/project/hist/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/hist
[pypi-version]:             https://badge.fury.io/py/hist.svg
[rtd-badge]:                https://readthedocs.org/projects/hist/badge/?version=latest
[rtd-link]:                 https://hist.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
