from __future__ import annotations

import copy
from typing import Any, TypeVar

from boost_histogram import Histogram

from .. import Hist, __version__

__all__ = ["from_uhi", "remove_writer_info", "to_uhi"]


def from_uhi(data: dict[str, Any], /) -> Hist:
    import boost_histogram.serialization as bhs

    return Hist(bhs.from_uhi(data))


def to_uhi(h: Histogram, /) -> dict[str, Any]:
    import boost_histogram.serialization as bhs

    d = bhs.to_uhi(h)
    d["writer_info"]["hist"] = {"version": __version__}
    return d


T = TypeVar("T", bound="dict[str, Any]")


def remove_writer_info(obj: T) -> T:
    """Removes all hist writer_info from a histogram dict, axes dict, or storage dict. Makes copies where required, and the outer dictionary is always copied."""

    obj = copy.copy(obj)
    if "hist" in obj.get("writer_info", {}):
        obj["writer_info"] = copy.copy(obj["writer_info"])
        del obj["writer_info"]["hist"]

    if "axes" in obj:
        obj["axes"] = [remove_writer_info(ax) for ax in obj["axes"]]
    if "storage" in obj:
        obj["storage"] = remove_writer_info(obj["storage"])

    return obj
