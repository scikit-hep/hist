from __future__ import annotations

import hist


def test_has_multi_cell():
    assert "MultiCell" in repr(hist.storage.MultiCell)


def test_has_multi_cell_quick_construct():
    h = hist.new.Regular(10, 0, 1).MultiCell(2)
    assert h.storage_type.__name__ == "MultiCell"
    assert h.values().shape == (2, 10)
