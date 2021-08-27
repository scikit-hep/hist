.. image:: _images/histlogo.png
   :width: 60%
   :alt: Hist logo
   :align: center

Welcome to Hist's documentation!
================================

|Actions Status| |Documentation Status| |pre-commit.ci Status| |Code style: black| |PyPI version|
|Conda-Forge| |PyPI platforms| |DOI| |GitHub Discussion| |Gitter| |Scikit-HEP|

Introduction
------------

`Hist <https://github.com/scikit-hep/hist>`_ is a powerful Histogramming tool for analysis based on `boost-histogram <https://boost-histogram.readthedocs.io/en/latest/index.html>`_ (the Python binding of the Histogram library in Boost). It is a friendly analysis-focused project that uses `boost-histogram <https://boost-histogram.readthedocs.io/en/latest/index.html>`_ as a backend to do the work, but provides plotting tools, shortcuts, and new ideas.

To get an idea of creating histograms in Hist looks like, you can take a look at the :doc:`Examples <examples/index>`. Once you have a feel for what is involved in using Hist, we recommend you start by following the instructions in :doc:`Installation <installation>`. Then, go through the :doc:`User Guide </user-guide/index>`, and read the :doc:`Reference </reference/modules>` documentation. We value your contributions and you can follow the instructions in :doc:`Contributing <contributing>`. Finally, if you’re having problems, please do let us know at our :doc:`Support <support>` page.


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: User Guide
   :glob:

   installation
   user-guide/quickstart
   user-guide/axes
   user-guide/storages
   user-guide/accumulators
   user-guide/notebooks/Transform
   user-guide/notebooks/Reprs
   user-guide/notebooks/Plots
   user-guide/analyses
   user-guide/notebooks/Histogram
   user-guide/notebooks/Stack
   user-guide/notebooks/Interpolation

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Developers
   :glob:

   contributing
   support
   changelog

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Examples
   :glob:

   examples/HistDemo

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: API Reference
   :glob:

   reference/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Actions Status| image:: https://github.com/scikit-hep/hist/workflows/CI/badge.svg
   :target: https://github.com/scikit-hep/hist/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/hist/badge/?version=latest
   :target: https://hist.readthedocs.io/en/latest/?badge=latest
.. |pre-commit.ci Status| image:: https://results.pre-commit.ci/badge/github/scikit-hep/hist/main.svg
   :target: https://results.pre-commit.ci/repo/github/scikit-hep/hist
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |PyPI version| image:: https://badge.fury.io/py/hist.svg
   :target: https://pypi.org/project/hist/
.. |Conda-Forge| image:: https://img.shields.io/conda/vn/conda-forge/hist
   :target: https://github.com/conda-forge/hist-feedstock
.. |PyPI platforms| image:: https://img.shields.io/pypi/pyversions/hist
   :target: https://pypi.org/project/hist/
.. |DOI| image:: https://zenodo.org/badge/239605861.svg
   :target: https://zenodo.org/badge/latestdoi/239605861
.. |GitHub Discussion| image:: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
   :target: https://github.com/scikit-hep/hist/discussions
.. |Gitter| image:: https://badges.gitter.im/HSF/PyHEP-histogramming.svg
   :target: https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |Scikit-HEP| image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org/
