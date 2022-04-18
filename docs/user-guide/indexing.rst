.. _usage-indexing:

Indexing
========

Hist implements the UHI and UHI+ indexing protocols. You can read more about them on the `Indexing <https://uhi.readthedocs.io/en/latest/indexing.html>`_ and `Indexing+ <https://uhi.readthedocs.io/en/latest/indexing%2B.html>`_ pages.


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
