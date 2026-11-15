.. _usage-indexing:

Indexing
========

Hist implements the UHI and UHI+ indexing protocols. You can read more about them on the `Indexing <https://uhi.readthedocs.io/en/latest/indexing.html>`_ and `Indexing+ <https://uhi.readthedocs.io/en/latest/indexing%2B.html>`_ pages.

See the UHI `slicing guide <https://uhi.readthedocs.io/en/latest/indexing.html#slicing>`_ for selecting bin ranges, rebinning, and projecting axes.


Reusable slices
---------------

``hist.tag.Slicer()`` lets you write a slice outside a histogram's indexing brackets and store it for reuse. The resulting slice also works in named-axis selections:

.. code-block:: python3

    import hist

    h = hist.Hist(hist.axis.Regular(4, 0, 4, name="x"))
    h[...] = [1, 2, 3, 4]

    s = hist.tag.Slicer()
    middle = s[1:3]
    selected = h[middle]  # Equivalent to h[1:3]
    selected_by_name = h[{"x": middle}]

Both selections contain the bin values ``[2, 3]``.


Hist specific details
--------------------------------

Hist implements ``hist.loc``, ``builtins.sum``, ``hist.rebin``, ``hist.underflow``, and ``hist.overflow`` from the UHI spec. A ``hist.tag.at`` locator is provided as well, which simulates the Boost.Histogram C++ ``.at()`` indexing using the UHI locator protocol.

Hist allows "picking" using lists, similar to NumPy. If you select
with multiple lists, hist instead selects per-axis, rather than
group-selecting and reducing to a single axis, like NumPy does. You can use ``hist.loc(...)`` inside these lists.

Example::

    h = Hist(
        hist.axis.Regular(10, 0, 1),
        hist.axis.StrCategory(["a", "b", "c"]),
        hist.axis.IntCategory([5, 6, 7]),
    )

    minihist = h[:, [hist.loc("a"), hist.loc("c")], [0, 2]]

    # Produces a 3D histgoram with Regular(10, 0, 1) x StrCategory(["a", "c"]) x IntCategory([5, 7])


This feature is considered experimental. Removed bins are not added to the overflow bin currently.
