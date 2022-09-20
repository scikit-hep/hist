# Changelog

## Version 2.6.2

* Nicer stacks repr
  [#449](https://github.com/scikit-hep/hist/pull/449)
* Backport `storage_type` if boost-histogram < 1.3.2
  [#447](https://github.com/scikit-hep/hist/pull/447)
* Allow overwriting labels for plot/overlay
  [#414](https://github.com/scikit-hep/hist/pull/414)
* Use Hatching to build the package
  [#418](https://github.com/scikit-hep/hist/pull/418)
* Support git archival version numbers
  [#441](https://github.com/scikit-hep/hist/pull/441)


## Version 2.6.1

* Fall back on normal repr when histogram is too large
  [#388](https://github.com/scikit-hep/hist/pull/388)
* Fix issue with no-axis histogram
  [#388](https://github.com/scikit-hep/hist/pull/388)
* Fix issue with empty axis causing segfault until fixed upstream
  [#387](https://github.com/scikit-hep/hist/pull/387)
* Only require SciPy if using SciPy
  [#386](https://github.com/scikit-hep/hist/pull/386)

## Version 2.6.0

* Using [boost-histogram 1.3](https://boost-histogram.readthedocs.io/en/latest/CHANGELOG.html#version-1-3)
* Fix runtime dependency on matplotlib when not plotting
  [#353](https://github.com/scikit-hep/hist/pull/353)
* Fix `.plot` shortcut failure
  [#368](https://github.com/scikit-hep/hist/pull/368)
* New nox sessions: `regenerate` and `pylint`
* Update tests for latest matplotlib

## Version 2.5.2

* Remove more-itertools requirement
  [#347](https://github.com/scikit-hep/hist/pull/347)
* Fix missing pass-through for stack plot
  [#339](https://github.com/scikit-hep/hist/pull/339)

## Version 2.5.1

* Support named stack indexing
  [#325](https://github.com/scikit-hep/hist/pull/325)
* Fix histoprint error with stacks
  [#325](https://github.com/scikit-hep/hist/pull/325)
* Better README

## Version 2.5.0

* Stacks support axes, math operations, projection, setting items, and
  iter/dict construction. They also support histogram titles in
  legends. Added histoprint support for Stacks.
  [#291](https://github.com/scikit-hep/hist/pull/291)
  [#315](https://github.com/scikit-hep/hist/pull/315)
  [#317](https://github.com/scikit-hep/hist/pull/317)
  [#318](https://github.com/scikit-hep/hist/pull/318)

* Added `name=` and `label=` to histograms, include Hist arguments in
  QuickConstruct. [#297](https://github.com/scikit-hep/hist/pull/297)

* AxesTuple now supports bulk name setting,
  `h.axes.name = ("a", "b", ...)`.
  [#288](https://github.com/scikit-hep/hist/pull/288)

* Added `hist.new` alias for `hist.Hist.new`.
  [#296](https://github.com/scikit-hep/hist/pull/296)

* Added `"efficiency"` `uncertainty_type` option for `ratio_plot` API.
  [#266](https://github.com/scikit-hep/hist/pull/266)
  [#278](https://github.com/scikit-hep/hist/pull/278)

Smaller features or fixes:

* Dropped Python 3.6 support.
  [#194](https://github.com/scikit-hep/hist/pull/194)
* Uses boost-histogram 1.2.x series, includes all features and fixes,
  and Python 3.10 support.
* No longer require scipy or iminuit unless actually needed.
  [#316](https://github.com/scikit-hep/hist/pull/316)
* Improve and clarify treatment of confidence intervals in `intervals`
  submodule.
  [#281](https://github.com/scikit-hep/hist/pull/281)
* Use NumPy 1.21 for static typing.
  [#285](https://github.com/scikit-hep/hist/pull/285)
* Support running tests without plotting requirements.
  [#321](https://github.com/scikit-hep/hist/pull/321)

## Version 2.4.0

* Support `.stack(axis)` and stacked histograms.
  [#244](https://github.com/scikit-hep/hist/pull/244)
  [#257](https://github.com/scikit-hep/hist/pull/257)
  [#258](https://github.com/scikit-hep/hist/pull/258)
* Support selection lists (experimental with boost-histogram 1.1.0).
  [#255](https://github.com/scikit-hep/hist/pull/255)
* Support full names for QuickConstruct, and support mistaken usage in
  constructor. [#256](https://github.com/scikit-hep/hist/pull/256)
* Add `.sort(axis)` for quickly sorting a categorical axis.
  [#243](https://github.com/scikit-hep/hist/pull/243)

Smaller features or fixes:

* Support nox for easier contributor setup.
  [#228](https://github.com/scikit-hep/hist/pull/228)
* Better name axis error.
  [#232](https://github.com/scikit-hep/hist/pull/232)
* Fix for issue plotting size 0 axes.
  [#238](https://github.com/scikit-hep/hist/pull/238)
* Fix issues with repr information missing.
  [#241](https://github.com/scikit-hep/hist/pull/241)
* Fix issues with wrong plot shortcut being triggered by Integer axes.
  [#247](https://github.com/scikit-hep/hist/pull/247)
* Warn and better error if overlapping keyword used as axis name.
  [#250](https://github.com/scikit-hep/hist/pull/250)

Along with lots of smaller docs updates.

## Version 2.3.0

* Add `plot_ratio` to the public API, which allows for making ratio
  plots between the histogram and either another histogram or a
  callable. [#161](https://github.com/scikit-hep/hist/pull/161)
* Add `.profile` to compute a (N-1)D profile histogram.
  [#160](https://github.com/scikit-hep/hist/pull/160)
* Support plot1d / plot on Histograms with a categorical axis.
  [#174](https://github.com/scikit-hep/hist/pull/174)
* Add frequentist coverage interval support in the `intervals` module.
  [#176](https://github.com/scikit-hep/hist/pull/176)
* Allow `plot_pull` to take a more generic callable or a string as a
  fitting function. Introduce an option to perform a likelihood fit.
  Write fit parameters' values and uncertainties in the legend.
  [#149](https://github.com/scikit-hep/hist/pull/149)
* Add `fit_fmt=` to `plot_pull` to control display of fit params.
  [#168](https://github.com/scikit-hep/hist/pull/168)
* Support `<prefix>_kw` arguments for setting each axis in plotting.
  [#193](https://github.com/scikit-hep/hist/pull/193)
* Cleaner IPython completion for Python 3.7+.
  [#179](https://github.com/scikit-hep/hist/pull/179)

## Version 2.2.1

* Fix bug with `plot_pull` missing a sqrt.
  [#150](https://github.com/scikit-hep/hist/pull/150)
* Fix static typing with ellipses.
  [#145](https://github.com/scikit-hep/hist/pull/145)
* Require boost-histogram 1.0.1+, fixing typing related issues,
  allowing subclassing Hist without a family and including a important
  Mean/WeighedMean summing fix.
  [#151](https://github.com/scikit-hep/hist/pull/151)

## Version 2.2.0

* Support boost-histogram 1.0. Better plain reprs. Full Static Typing.
  [#137](https://github.com/scikit-hep/hist/pull/137)
* Support `data=` when construction a histogram to copy in initial
  data. [#142](https://github.com/scikit-hep/hist/pull/142)
* Support `Hist.from_columns`, for simple conversion of DataFrames and
  similar structures
  [#140](https://github.com/scikit-hep/hist/pull/140)
* Support `.plot_pie` for quick pie plots
  [#140](https://github.com/scikit-hep/hist/pull/140)

## Version 2.1.1

* Fix density (and density based previews)
  [#134](https://github.com/scikit-hep/hist/pull/134)

## Version 2.1.0

* Support shortcuts for setting storages by string or position
  [#129](https://github.com/scikit-hep/hist/pull/129)

Updated dependencies:

* `boost-histogram` 0.11.0 to 0.13.0.
  * Major new features, including PlottableProtocol

* `histoprint` >=1.4 to >=1.6.

* `mplhep` >=0.2.16 when `[plot]` given

## Version 2.0.1

* Make sum of bins explicit in notebook representations.
  [#106](https://github.com/scikit-hep/hist/pull/106)
* Fixed `plot2d_full` incorrectly mirroring the y-axis.
  [#105](https://github.com/scikit-hep/hist/pull/105)
* `Hist.plot_pull`: more suitable bands in the pull bands 1sigma, 2
  sigma, etc. [#102](https://github.com/scikit-hep/hist/pull/102)
* Fixed classichist's usage of `get_terminal_size` to support not running in
  a terminal [#99](https://github.com/scikit-hep/hist/pull/99)

## Version 2.0.0

First release of Hist.
