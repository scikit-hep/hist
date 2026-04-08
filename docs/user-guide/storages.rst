.. _usage-storage:

Storages
========

There are several storages to choose from. Based on
`boost-histogram <https://github.com/scikit-hep/boost-histogram>`_’s Storage,
hist supports seven storage types, ``Double``, ``Unlimited``, ``Int64``,
``AutomicInt64``, ``Weight``, ``Mean``, and ``WeightedMean``.

There are two methods to select a Storage in hist:

* **Method 1:** You can use `boost-histogram <https://github.com/scikit-hep/boost-histogram>`_’s Storage in hist by passing the ``storage=hist.storage.`` argument when making a histogram.

* **Method 2:** Keeping the original features of `boost-histogram <https://github.com/scikit-hep/boost-histogram>`_’s Storage, hist gives dynamic shortcuts of Storage Proxy. You can also add Storage types after adding the axes.

Simple Storages
---------------

These storages hold a single value that keeps track of a count, possibly a
weighed count.

Double
^^^^^^

By default, hist selects the ``Double()`` storage. For most uses,
this should be ideal. It is just as fast as the ``Int64()`` storage, it can fill
up to 53 bits of information (9 quadrillion) counts per cell, and supports
weighted fills. It can also be scaled by a floating point values without making
a copy.

.. code-block:: python3

    # Method 1
    h = Hist(hist.axis.Regular(10, -5, 5, name="x"), storage=hist.storage.Double())
    h.fill([1.5, 2.5], weight=[0.5, 1.5])

    print(h[1.5j])
    print(h[2.5j])

.. code-block:: text

    0.5
    1.5

.. code-block:: python3

    # Method 2
    h = (
        Hist.new.Reg(10, 0, 1, name="x")
        .Reg(10, 0, 1, name="y")
        .Double()
        .fill(x=[0.5, 0.5], y=[0.2, 0.6])
    )

    print(h[0.5j, 0.2j])
    print(h[0.5j, 0.6j])

.. code-block:: text

    1.0
    1.0


Unlimited
^^^^^^^^^

The Unlimited storage starts as an 8-bit integer and grows, and converts to a
double if weights are used (or, currently, if a view is requested). This allows
you to keep the memory usage minimal, at the expense of occasionally making an
internal copy.

Int64
^^^^^

A true integer storage is provided, as well; this storage has the ``np.uint64``
datatype.  This eventually should provide type safety by not accepting
non-integer fills for data that should represent raw, unweighed counts.

.. code-block:: python3

    # Method 1
    h = Hist(hist.axis.Regular(10, -5, 5, name="x"), storage=hist.storage.Int64())
    h.fill([1.5, 2.5, 2.5, 2.5])

    print(h[1.5j])
    print(h[2.5j])

.. code-block:: text

    1
    3

.. code-block:: python3

    # Method 2
    h = Hist.new.Reg(10, 0, 1, name="x").Int64().fill([0.5, 0.5])

    print(h[0.5j])

.. code-block:: text

    2


AtomicInt64
^^^^^^^^^^^

This storage is like ``Int64()``, but also provides a thread safety guarantee.
You can fill a single histogram from multiple threads.


Accumulator storages
--------------------

These storages hold more than one number internally. They return a smart view when queried
with ``.view()``; see :ref:`usage-accumulators` for information on each accumulator and view.

Weight
^^^^^^

This storage keeps a sum of weights as well (in CERN ROOT, this is like calling
``.Sumw2()`` before filling a histogram). It uses the ``WeightedSum`` accumulator.


MultiCell
^^^^^^^^^

This storage is like the ``Double`` storage but supports storing multiple entries per bin independently. This is useful if one has to deal with many independent weights per event that all correspond to the same parameter (they will a be binned into the same bin) and have to be summed independently (e.g. to track the effect of systematic variations). It is supposed to be filled much faster compared to filling many histograms with a ``Weight`` storage type in a loop if one deals with a large number of different weights.

The number of entries per bin has to be fixed and is provided to the storage at its construction through

.. code-block:: python3

    bh.Histogram(…, storage=MultiCell(nelem))

where ``nelem`` is the number of entries per bin.
The entries have to provided as a 2-dimensional array ``(n, nelem)`` with the first dimension being the events to histogram and the second axis being the entries per event.
To fill a histogram ``h`` one has to provide the entries via the ``weight`` keyword:

.. code-block:: python3

    h.fill(..., weight=weights)

Any slicing or projection operation works for ``MultiCell`` histograms identical to any other histogram with different storage type, the entries are here not considered an additional axis for the histogram.
Calling ``h.view()`` returns an array where the entries are indexed as the first axis (e.g. ``h.view()[0]`` is the histogram content for the first entry per bin).
Contrary to the ``Weight`` storage the ``MultiCell`` storage does not track variances (it does not track the sum of weights squared) because this might not be necessary for every entry index. Instead, the user is supposed to track the variances themselves if required. This could be achieved by providing the variances as a separate entry to the ``MultiCell`` histogram by increasing the number of entries that are stored per bin (e.g. one could provide the square of ``weights[:, 0]`` as ``weights[:, 1]`` to track the sum of weight squared of the first entry index in the second entry index).
Note: If you should ever need to use the lowlevel ``h._hist.fill()`` function with a ``MultiCell`` storage you will have to use the ``sample`` keyword to pass the weights instead of the ``weight`` keyword because that is used on the C++ side, but the highlevel python boost_histogram API hides this from the user.


Mean
^^^^

This storage tracks a "Profile", that is, the mean value of the accumulation instead of the sum.
It stores the count (as a double), the mean, and a term that is used to compute the variance. When
filling, you must add a ``sample=`` term.


WeightedMean
^^^^^^^^^^^^

This is similar to Mean, but also keeps track a sum of weights like term as well.
