Changelog
====================


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
