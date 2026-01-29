from __future__ import annotations

import importlib.metadata

import packaging.version

import hist


def test_has_multi_cell():
    bh_version = packaging.version.Version(
        importlib.metadata.version("boost_histogram")
    )
    if bh_version >= packaging.version.Version("1.7"):
        assert "MultiCell" in repr(hist.storage.MultiCell)

def test_has_multi_cell_quick_construct():
    bh_version = packaging.version.Version(
        importlib.metadata.version("boost_histogram")
    )
    if bh_version >= packaging.version.Version("1.7"):
        h = hist.new.Regular(10, 0, 1).MultiCell(2)
        assert h.storage_type.__name__ == "MultiCell"
        assert h.values().shape == (2, 10)