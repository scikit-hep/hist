Changelog
====================

Version 2.4.0
--------------------

* Support ``.stack(axis)`` and stacked histograms.
  `#244 <https://github.com/scikit-hep/hist/pull/244>`_
  `#257 <https://github.com/scikit-hep/hist/pull/257>`_
  `#258 <https://github.com/scikit-hep/hist/pull/258>`_

* Support selection lists (experimental with boost-histogram 1.1.0).
  `#255 <https://github.com/scikit-hep/hist/pull/255>`_

* Support full names for QuickConstruct, and support mistaken usage in constructor.
  `#256 <https://github.com/scikit-hep/hist/pull/256>`_

* Add ``.sort(axis)`` for quickly sorting a categorical axis.
  `#243 <https://github.com/scikit-hep/hist/pull/243>`_


Smaller features or fixes:

* Support nox for easier contributor setup.
  `#228 <https://github.com/scikit-hep/hist/pull/228>`_

* Better name axis error.
  `#232 <https://github.com/scikit-hep/hist/pull/232>`_

* Fix for issue plotting size 0 axes.
  `#238 <https://github.com/scikit-hep/hist/pull/238>`_

* Fix issues with repr information missing.
  `#241 <https://github.com/scikit-hep/hist/pull/241>`_

* Fix issues with wrong plot shortcut being triggered by Integer axes.
  `#247 <https://github.com/scikit-hep/hist/pull/247>`_

* Warn and better error if overlapping keyword used as axis name.
  `#250 <https://github.com/scikit-hep/hist/pull/250>`_

Along with lots of smaller docs updates.






Version 2.3.0
--------------------

* Add ``plot_ratio`` to the public API, which allows for making ratio plots between the
  histogram and either another histogram or a callable.
  `#161 <https://github.com/scikit-hep/hist/pull/161>`_

* Add ``.profile`` to compute a (N-1)D profile histogram.
  `#160 <https://github.com/scikit-hep/hist/pull/160>`_

* Support plot1d / plot on Histograms with a categorical axis.
  `#174 <https://github.com/scikit-hep/hist/pull/174>`_

* Add frequentist coverage interval support in the ``intervals`` module.
  `#176 <https://github.com/scikit-hep/hist/pull/176>`_

* Allow ``plot_pull`` to take a more generic callable or a string as a fitting function.
  Introduce an option to perform a likelihood fit. Write fit parameters' values
  and uncertainties in the legend.
  `#149 <https://github.com/scikit-hep/hist/pull/149>`_

* Add ``fit_fmt=`` to ``plot_pull`` to control display of fit params.
  `#168 <https://github.com/scikit-hep/hist/pull/168>`_

* Support ``<prefix>_kw`` arguments for setting each axis in plotting.
  `#193 <https://github.com/scikit-hep/hist/pull/193>`_

* Cleaner IPython completion for Python 3.7+.
  `#179 <https://github.com/scikit-hep/hist/pull/179>`_


Version 2.2.1
--------------------

* Fix bug with ``plot_pull`` missing a sqrt.
  `#150 <https://github.com/scikit-hep/hist/pull/150>`_

* Fix static typing with ellipses.
  `#145 <https://github.com/scikit-hep/hist/pull/145>`_

* Require boost-histogram 1.0.1+, fixing typing related issues, allowing
  subclassing Hist without a family and including a important Mean/WeighedMean
  summing fix.
  `#151 <https://github.com/scikit-hep/hist/pull/151>`_

Version 2.2.0
--------------------

* Support boost-histogram 1.0. Better plain reprs. Full Static Typing.
  `#137 <https://github.com/scikit-hep/hist/pull/137>`_

* Support ``data=`` when construction a histogram to copy in initial data.
  `#142 <https://github.com/scikit-hep/hist/pull/142>`_

* Support ``Hist.from_columns``, for simple conversion of DataFrames and similar structures
  `#140 <https://github.com/scikit-hep/hist/pull/140>`_

* Support ``.plot_pie`` for quick pie plots
  `#140 <https://github.com/scikit-hep/hist/pull/140>`_

Version 2.1.1
--------------------

* Fix density (and density based previews)
  `#134 <https://github.com/scikit-hep/hist/pull/134>`_


Version 2.1.0
--------------------

* Support shortcuts for setting storages by string or position
  `#129 <https://github.com/scikit-hep/hist/pull/129>`_

Updated dependencies:

* ``boost-histogram`` 0.11.0 to 0.13.0.
    * Major new features, including PlottableProtocol

* ``histoprint`` >=1.4 to >=1.6.

* ``mplhep`` >=0.2.16 when ``[plot]`` given


Version 2.0.1
--------------------

* Make sum of bins explicit in notebook representations.
  `#106 <https://github.com/scikit-hep/hist/pull/106>`_

* Fixed ``plot2d_full`` incorrectly mirroring the y-axis.
  `#105 <https://github.com/scikit-hep/hist/pull/105>`_

* ``Hist.plot_pull``: more suitable bands in the pull bands 1sigma, 2 sigma, etc.
  `#102 <https://github.com/scikit-hep/hist/pull/102>`_

* Fixed classichist's usage of `get_terminal_size` to support not running in a terminal
  `#99 <https://github.com/scikit-hep/hist/pull/99>`_


Version 2.0.0
--------------------

First release of Hist.
