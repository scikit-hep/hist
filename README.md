# Hist

[![Github Actions badge](https://github.com/scikit-hep/hist/workflows/CI/badge.svg)](https://github.com/scikit-hep/hist/actions)
[![Join the chat at https://gitter.im/Scikit-HEP/hist](https://badges.gitter.im/HSF/PyHEP-histogramming.svg)](https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Hist is a analyst friendly front-end for [boost-histogram](https://github.com/scikit-hep/boost-histogram).

## Installation

You can install this library from [PyPI](https://pypi.org/project/hist/) with pip:

```bash
python -m pip install hist
```

## Usage

```python
import hist

# You can create a histogram like this.
h = (
  hist.Hist()
  .Reg(10, 0 ,1, name="x", label="x-axis")
  .Variable(range(10), name="y", label="y-axis")
  .Int64()
)

# Filling by names is allowed in hist.
hist.fill(y=[1, 4, 6], x=[3, 5, 2])

# New ways to manipulate the histogram.
h.project("x")
h[{"y": 1j + 3, "x": 5j}]
...

# Elegant plotting functions.
h.plot()
h.plot2d_full()
h.plot_pull(Callable)
...
```

## Features

- Hist augments metadata by adding names to axes; these are *highly* recommend and will help you track axes. There is also a special `NamedHist`, which will enforce all hist axes have names, and all axes will require named access.
  - `   name=` is a unique label describing each axis
  - `label=` is an optional string that is used in plotting (defaults to name if not provided)
  - Indexing, projection, and more support named axes.

- The `Hist` class augments the `bh.Histogram` class with the following shortcuts, designed for interactive exploration without extensive imports:
  - Optional import-free construction system
  - Quick import-free data-coordinates and rebin syntax (use a j suffix for numbers, or strings directly in indexing expressions)

- Quick plotting routines encourage exploration:

  - `.plot()` provides 1D and 2D plots
  - `.plot2d_full()` shows 1D projects around a 2D plot
  - `.plot_pull(...)` performs a pull plot

## Development

```bash
python -m pip install hist
```

See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for information on setting up a development environment.

## Contributors

We would like to acknowledge the contributors that made this project possible ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="#maintenance-henryiii" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/hist/commits?author=henryiii" title="Code">💻</a> <a href="https://github.com/scikit-hep/hist/commits?author=henryiii" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/lovelybuggies"><img src="https://avatars3.githubusercontent.com/u/29083689?v=4" width="100px;" alt=""/><br /><sub><b>Nino Lau</b></sub></a><br /><a href="#maintenance-lovelybuggies" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/hist/commits?author=lovelybuggies" title="Code">💻</a><a href="https://github.com/scikit-hep/hist/commits?author=lovelybuggies" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.

## Talks

* [2020-07-07 SciPy Proceedings](https://www.youtube.com/watch?v=ERraTfHkPd0&list=PLYx7XA2nY5GfY4WWJjG5cQZDc7DIUmn6Z&index=4)
* [2020-07-17 PyHEP2020](https://indico.cern.ch/event/882824/contributions/3931299/)

---

## Acknowledgements

This library was primarily developed by Henry Schreiner and Nino Lau.

Support for this work was provided by the National Science Foundation cooperative agreement OAC-1836650 (IRIS-HEP) and OAC-1450377 (DIANA/HEP). Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
