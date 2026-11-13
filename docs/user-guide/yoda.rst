.. _usage-yoda:

YODA
====

The ``hist.io.yoda`` module reads and writes the text format of
`YODA <https://yoda.hepforge.org>`_. It does not need YODA.

.. code-block:: python

    import hist
    import hist.io.yoda

    h = hist.Hist.new.Reg(10, 0, 1, name="x").Weight()
    h.fill([0.1, 0.2, 0.3])

    hist.io.yoda.write("out.yoda", {"/analysis/h": h})
    hists = hist.io.yoda.read("out.yoda")

``read`` and ``write`` use a dict of YODA path to histogram, and support
compressed ``.yoda.gz`` files. ``loads`` and ``dumps`` do the same with a
string.

Reading
-------

The reader supports the YODA 2 format and the legacy YODA 1 format.

.. list-table::
   :header-rows: 1

   * - YODA object
     - Result
   * - ``Histo1D``, ``Histo2D``, ``Histo3D``, ``BinnedHisto``
     - ``Weight`` storage
   * - ``Profile1D``, ``Profile2D``, ``BinnedProfile``
     - ``WeightedMean`` storage
   * - ``Counter``
     - ``Weight`` storage with no axes
   * - ``Estimate0D``, ``Estimate1D``, ``Estimate2D``, ``Estimate3D``,
       ``BinnedEstimate``
     - ``Weight`` storage. The variance is the square of the average total
       error, so asymmetric errors and the error sources are lost.
   * - ``Scatter1D``, ``Scatter2D``, ``Scatter3D``
     - Skipped with a warning, a scatter is not a histogram.

A continuous axis becomes a ``Variable`` axis. Integer and string axes become
``IntCategory`` and ``StrCategory`` axes, and the YODA "otherflow" bin is the
overflow bin. The title is stored in ``.label``. All other annotations are
stored as a dict of strings in ``.metadata``. YODA does not have axis names.

Writing
-------

The writer produces the YODA 2 format. It supports the objects that YODA can
read: histograms with one or two axes, histograms with three continuous axes,
profiles (``Mean`` and ``WeightedMean`` storages) with one or two axes, and
counters (no axes).

* YODA always has flow bins. Missing flow bins are written as empty bins.
* ``Integer``, ``Boolean``, and category axes are written as discrete axes.
  The underflow and overflow bins are added together.
* YODA stores per-bin moments of the fill coordinates and the number of
  entries. These are computed from the bin centers and the effective number of
  entries.
* The default precision is the YODA default, 6 digits. Use ``precision=17``
  for a lossless file.
