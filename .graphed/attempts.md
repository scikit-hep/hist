# attempts — hist-graphed-mvp (branch graphed-mvp)

## HIST-1 — hist.graphed: deferred hist filling on graphed task graphs — 2026-06-10 (freeze-HIST-1)

P0.1 of the ADL-benchmarks port (user-confirmed plan). src/hist/graphed mirrors src/hist/dask:
Hist/NamedHist as the MRO sandwich (hist's QuickConstruct + named-axis handling over
graphed_histogram.boost.Histogram's deferred fill), with `_in_memory_type` so compute() returns
a REAL hist.Hist (named indexing works on results).

- tests/test_graphed.py (5 tests, test-first against the integration surface): QuickConstruct
  deferred == eager twins BIT FOR BIT (counts incl. flow) over graphed-numpy AND graphed-awkward
  sources (ragged fills flatten completely) and a real uproot TTree (np.hypot of branches);
  weighted 2D NamedHist (values + variances); multi-fill accumulation; ProcessExecutor plan ==
  compute; the efficiency witness (the source's whole-dataset loader never runs).
- Findings folded back into graphed-histogram iteration 1 (M23): the `_in_memory_type` wrapping
  hook, and hist's name/label living in the axis `__dict__` (boost's metadata mechanism) — now
  captured/restored by the canonical spec.
- Full hist suite green alongside: 200 passed, 8 skipped (mplhep + pytest-mpl are test deps).
