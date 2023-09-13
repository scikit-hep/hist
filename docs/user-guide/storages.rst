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


Mean
^^^^

This storage tracks a "Profile", that is, the mean value of the accumulation instead of the sum.
It stores the count (as a double), the mean, and a term that is used to compute the variance. When
filling, you must add a ``sample=`` term.


WeightedMean
^^^^^^^^^^^^

This is similar to Mean, but also keeps track a sum of weights like term as well.
