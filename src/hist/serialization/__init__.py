from __future__ import annotations

import copy
import json
import typing
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import boost_histogram.serialization as bhs
from boost_histogram import Histogram

from .. import Hist, __version__

if TYPE_CHECKING:
    import os

    from uhi.typing.serialization import AnyHistogramIR

__all__ = ["from_uhi", "read", "remove_writer_info", "to_uhi", "write"]


def __dir__() -> list[str]:
    return __all__


def from_uhi(data: dict[str, Any], /) -> Hist[Any]:
    return Hist(bhs.from_uhi(data))


def to_uhi(h: Histogram[Any], /) -> dict[str, Any]:
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


_HDF5_SUFFIXES = {".h5", ".hdf5", ".hdf"}
_DEFAULT_NAME = "histogram"


def _backend(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".zip":
        return "zip"
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    msg = f"Unsupported file extension {suffix!r} for {path}; use .json, .zip, or .h5/.hdf5"
    raise ValueError(msg)


def _select_name(names: list[str], path: Path) -> str:
    if len(names) == 1:
        return names[0]
    if not names:
        msg = f"No histograms found in {path}"
    else:
        msg = f"Multiple histograms in {path}; pass name= (one of {names})"
    raise ValueError(msg)


def write(
    filename: str | os.PathLike[str],
    h: Histogram[Any],
    /,
    *,
    name: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Write a histogram to a UHI file. The backend is chosen by extension:
    ``.json``, ``.zip``, or ``.h5``/``.hdf5`` (requires h5py).

    JSON files hold a single histogram and are overwritten. Zip and HDF5
    files are opened in append mode, and ``name`` selects the entry; it
    defaults to the histogram's ``name`` or ``"histogram"``. Extra keyword
    arguments are passed to the backend writer.
    """
    path = Path(filename)
    backend = _backend(path)
    data = typing.cast("AnyHistogramIR", to_uhi(h))

    if backend == "json":
        import uhi.io.json

        if name is not None:
            msg = "name= is not supported for JSON files"
            raise TypeError(msg)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, default=uhi.io.json.default, **kwargs)
        return

    if name is None:
        name = getattr(h, "name", None) or _DEFAULT_NAME

    if backend == "zip":
        import uhi.io.zip

        with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            uhi.io.zip.write(zf, name, data, **kwargs)
        return

    import h5py
    import uhi.io.hdf5

    with h5py.File(path, "a") as f:
        uhi.io.hdf5.write(f.create_group(name), data, **kwargs)


def read(
    filename: str | os.PathLike[str],
    /,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Read a UHI histogram dictionary from a file. The backend is chosen by
    extension: ``.json``, ``.zip``, or ``.h5``/``.hdf5`` (requires h5py).

    For zip and HDF5 files, ``name`` selects the entry; if not given, the
    file must contain exactly one histogram. Use :meth:`hist.Hist.read` to
    get a histogram object directly.
    """
    path = Path(filename)
    backend = _backend(path)

    if backend == "json":
        import uhi.io.json

        if name is not None:
            msg = "name= is not supported for JSON files"
            raise TypeError(msg)
        with path.open(encoding="utf-8") as f:
            output: dict[str, Any] = json.load(f, object_hook=uhi.io.json.object_hook)
        return output

    if backend == "zip":
        import uhi.io.zip

        with zipfile.ZipFile(path) as zf:
            if name is None:
                names = [n[:-5] for n in zf.namelist() if n.endswith(".json")]
                name = _select_name(names, path)
            return uhi.io.zip.read(zf, name)

    import h5py
    import uhi.io.hdf5

    with h5py.File(path) as f:
        if name is None:
            names = [k for k, v in f.items() if "uhi_schema" in v.attrs]
            name = _select_name(names, path)
        return typing.cast("dict[str, Any]", uhi.io.hdf5.read(f[name]))
