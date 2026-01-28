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
