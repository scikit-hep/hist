from __future__ import annotations

__lazy_modules__ = {"hist"}

import hist


def test_version():
    assert hist.__version__ is not None
