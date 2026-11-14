"""
Read and write the YODA text format.

Reading supports the YODA 2 format (``V3``) and the legacy YODA 1 format
(``V2``). Writing always produces the YODA 2 format.

==================  ==============================  ==========================
YODA object         Read as                         Written from
==================  ==============================  ==========================
Histo / BinnedHisto ``Weight`` storage              all storages except means
Profile             ``WeightedMean`` storage        ``Mean``, ``WeightedMean``
Counter             0-D ``Weight`` storage          0-D histograms
Estimate            ``Weight`` storage (lossy)      not written
Scatter             skipped with a warning          not written
==================  ==============================  ==========================

YODA keeps per-bin moments of the fill coordinates and the number of entries.
A histogram does not have these, so the writer computes them from the bin
centers and the effective number of entries.
"""

from __future__ import annotations

__lazy_modules__ = {
    "boost_histogram",
    "collections.abc",
    "gzip",
    "pathlib",
    "warnings",
    f"{(__spec__.parent or '').rsplit('.', 1)[0]}",
}

import dataclasses
import gzip
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boost_histogram as bh
import numpy as np
import numpy.typing as npt

from .. import Hist, axis, storage

if TYPE_CHECKING:
    import os

__all__ = ["dumps", "loads", "read", "write"]

Array = npt.NDArray[np.float64]

_BEGIN = re.compile(
    r"BEGIN\s+(?:YODA_)?(?P<type>\S+?)(?:_V(?P<version>\d+))?(?:\s+(?P<path>\S+))?$"
)
_TYPE = re.compile(
    r"(?:BINNED)?(?P<kind>HISTO|PROFILE|ESTIMATE)(?:(?P<ndim>\d)D|<(?P<axes>[DIS,]+)>)$"
)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
_RESERVED = {"Path", "Title", "Type"}


@dataclasses.dataclass
class _Block:
    type: str
    version: int
    path: str
    annotations: dict[str, str] = dataclasses.field(default_factory=dict)
    lines: list[str] = dataclasses.field(default_factory=list)


def _blocks(text: str) -> Iterator[_Block]:
    block: _Block | None = None
    in_annotations = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            # YODA 1 allowed "# BEGIN" and "# END"
            marker = line.lstrip("# ")
            if not marker.startswith(("BEGIN ", "END ")):
                continue
            line = marker
        if not line:
            continue

        if block is None:
            match = _BEGIN.match(line)
            if match is None:
                msg = f"Expected a BEGIN line, got {line!r}"
                raise ValueError(msg)
            block = _Block(
                type=match["type"],
                version=int(match["version"] or 1),
                path=match["path"] or "",
            )
            in_annotations = True
        elif line.startswith("BEGIN "):
            msg = f"Unexpected BEGIN line inside {block.path!r}"
            raise ValueError(msg)
        elif line.startswith("END "):
            yield block
            block = None
        elif in_annotations:
            if line == "---":
                in_annotations = False
            elif ":" in line:
                key, _, value = line.partition(":")
                block.annotations[key.strip()] = value.strip()
        else:
            block.lines.append(line)

    if block is not None:
        msg = f"Missing END line for {block.path!r}"
        raise ValueError(msg)


def _unquote(text: str) -> list[str]:
    return [re.sub(r"\\(.)", r"\1", m) for m in _QUOTED.findall(text)]


def _make_axis(kind: str, text: str) -> Any:
    if kind == "s":
        return axis.StrCategory(_unquote(text))
    items = [item for item in text.strip(" []").split(",") if item.strip()]
    if kind == "i":
        return axis.IntCategory([int(item) for item in items])
    return axis.Variable([float(item) for item in items])


def _parse_v3(
    kind: str, axis_types: str, lines: Sequence[str]
) -> tuple[list[Any], Array]:
    """
    Return the axes and the columns in the hist layout, ``(*flow_shape, ncols)``.
    """
    edges: list[str] = []
    labels: list[str] = []
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("Edges(A"):
            edges.append(line.partition(":")[2])
        elif line.startswith("ErrorLabels:"):
            labels = _unquote(line)
        elif not line.startswith("MaskedBins:"):
            # "---" marks a missing estimate error
            rows.append(line.replace("---", "nan").split())

    axes = [_make_axis(t, e) for t, e in zip(axis_types, edges, strict=True)]

    table = np.array(rows, dtype=np.float64)
    if kind == "ESTIMATE":
        errors = table[:, 1:].reshape((len(rows), len(labels), 2))
        missing = np.isnan(errors[..., 0])
        errors[np.isnan(errors)] = 0.0
        # The total is the "" source if a bin has it, else the quadrature sum
        if "" in labels:
            k = labels.index("")
            quad = np.sqrt(np.sum(np.delete(errors, k, axis=1) ** 2, axis=1))
            total = np.where(missing[:, [k]], quad, np.abs(errors[:, k]))
        else:
            total = np.sqrt(np.sum(errors**2, axis=1))
        table = np.stack([table[:, 0], total.mean(axis=1) ** 2], axis=1)

    # Rows are in global index order: the first axis changes fastest.
    shape = [len(ax) + (2 if ax.traits.continuous else 1) for ax in axes]
    ndim = len(axes)
    data = table.reshape((*reversed(shape), -1)).transpose(*reversed(range(ndim)), ndim)
    for i, ax in enumerate(axes):
        if not ax.traits.continuous:
            # YODA puts the "otherflow" bin first, a category axis puts it last
            data = np.roll(data, -1, axis=i)
    return axes, data


def _parse_v2(ndim: int, lines: Sequence[str]) -> tuple[list[Any], Array]:
    bins: list[list[float]] = []
    flows: dict[str, list[float]] = {}
    for line in lines:
        tokens = line.split()
        if tokens[0] in {"Underflow", "Overflow"}:
            flows[tokens[0]] = [float(tok) for tok in tokens[2:]]
        elif tokens[0] != "Total":
            bins.append([float(tok) for tok in tokens])

    table = np.array(bins)
    if ndim == 0:
        return [], table[0]

    # Bins can have gaps and any order. Gaps become empty bins.
    edges = [np.unique(table[:, 2 * i : 2 * i + 2]) for i in range(ndim)]
    data = np.zeros((*(len(e) + 1 for e in edges), table.shape[1] - 2 * ndim))
    index = tuple(np.searchsorted(e, table[:, 2 * i]) + 1 for i, e in enumerate(edges))
    data[index] = table[:, 2 * ndim :]
    if ndim == 1:
        data[0] = flows.get("Underflow", 0.0)
        data[-1] = flows.get("Overflow", 0.0)
    return [axis.Variable(e) for e in edges], data


def _to_hist(block: _Block) -> Hist[Any] | None:
    if block.type == "COUNTER":
        kind, axis_types = "HISTO", ""
    else:
        match = _TYPE.match(block.type)
        if match is None:
            return None
        kind = match["kind"]
        if match["axes"] is None:
            axis_types = "d" * int(match["ndim"])
        else:
            axis_types = match["axes"].replace(",", "").lower()

    if block.version == 3:
        axes, data = _parse_v3(kind, axis_types, block.lines)
    elif block.version == 2 and kind != "ESTIMATE" and set(axis_types) <= {"d"}:
        axes, data = _parse_v2(len(axis_types), block.lines)
    else:
        return None

    title = block.annotations.get("Title", "")
    metadata = {k: v for k, v in block.annotations.items() if k not in _RESERVED}
    h: Hist[Any]
    if kind == "PROFILE":
        h = Hist(*axes, storage=storage.WeightedMean())
        sumw, sumw2 = data[..., 0], data[..., 1]
        sumwy, sumwy2 = data[..., 2 + 2 * len(axes)], data[..., 3 + 2 * len(axes)]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(sumw != 0, sumwy / sumw, 0.0)
            deltas = np.where(sumw != 0, sumwy2 - sumwy * mean, 0.0)
        view: Any = np.asarray(h.view(flow=True))
        view["sum_of_weights"] = sumw
        view["sum_of_weights_squared"] = sumw2
        view["value"] = mean
        view["_sum_of_weighted_deltas_squared"] = np.maximum(deltas, 0.0)
    else:
        h = Hist(*axes, storage=storage.Weight())
        view = np.asarray(h.view(flow=True))
        view["value"] = data[..., 0]
        view["variance"] = data[..., 1]

    # "~" is the YAML null that YODA 1 writes for an empty title
    h.label = title if title not in {"", "~"} else None
    h.metadata = metadata
    return h


def loads(text: str, /) -> dict[str, Hist[Any]]:
    """
    Read histograms from a string in the YODA text format. Returns a dict of
    path to histogram. Extra annotations are stored as a dict of strings in
    ``.metadata``, and the title is stored in ``.label``.

    Estimates keep the value and the square of the average total error.
    Objects that are not histograms, like scatters, are skipped with a warning.
    """
    result: dict[str, Hist[Any]] = {}
    skipped: list[str] = []
    for block in _blocks(text):
        h = _to_hist(block)
        if h is None:
            skipped.append(f"{block.path} ({block.type})")
        else:
            result[block.path] = h
    if skipped:
        msg = f"Skipped {len(skipped)} unsupported YODA object(s): {', '.join(skipped)}"
        warnings.warn(msg, stacklevel=2)
    return result


def read(file: str | os.PathLike[str], /) -> dict[str, Hist[Any]]:
    """
    Read histograms from a YODA text file (``.yoda`` or ``.yoda.gz``). See
    :func:`loads`.
    """
    data = Path(file).read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return loads(data.decode())


def _effective_entries(sumw: Array, sumw2: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sumw2 > 0, sumw**2 / sumw2, 0.0)


def _columns(h: bh.Histogram[Any]) -> tuple[Array, bool]:
    """
    Return the ``flow=True`` columns sumW, sumW2, numEntries, and for profiles
    the sample moments sumWY and sumWY2, stacked on the last axis. The bool is
    True for profiles.
    """
    # Plain array: field access on 0-D views fails in old boost-histogram
    view: Any = np.asarray(h.view(flow=True))

    def field(name: str) -> Array:
        return np.asarray(view[name], dtype=np.float64)

    if issubclass(h.storage_type, bh.storage.Mean):
        sumw = sumw2 = entries = field("count")
        mean, deltas = field("value"), field("_sum_of_deltas_squared")
    elif issubclass(h.storage_type, bh.storage.WeightedMean):
        sumw, sumw2 = field("sum_of_weights"), field("sum_of_weights_squared")
        # The number of entries is not stored, use the effective number
        entries = _effective_entries(sumw, sumw2)
        mean, deltas = field("value"), field("_sum_of_weighted_deltas_squared")
    elif issubclass(h.storage_type, bh.storage.Weight):
        sumw, sumw2 = field("value"), field("variance")
        entries = _effective_entries(sumw, sumw2)
        return np.stack([sumw, sumw2, entries], axis=-1), False
    else:
        sumw = np.asarray(view, dtype=np.float64)
        return np.stack([sumw, sumw, sumw], axis=-1), False

    sample = [sumw * mean, deltas + sumw * mean**2]
    return np.stack([sumw, sumw2, entries, *sample], axis=-1), True


def _yoda_layout(data: Array, axes: Sequence[Any]) -> Array:
    """
    Convert a ``flow=True`` array to the YODA bin layout. Trailing dimensions
    past the axes are kept.
    """
    for i, ax in enumerate(axes):
        pad = [(0, 0)] * data.ndim
        pad[i] = (int(not ax.traits.underflow), int(not ax.traits.overflow))
        data = np.pad(data, pad)
        if not ax.traits.continuous:
            # A discrete YODA axis has one "otherflow" bin, and it is first
            data = np.moveaxis(data, i, 0)
            data = np.concatenate([data[:1] + data[-1:], data[1:-1]])
            data = np.moveaxis(data, 0, i)
    return data


def _axis_info(ax: Any, precision: int) -> tuple[str, str, Array]:
    """
    Return the YODA axis type, the edges text, and the coordinate of each bin.
    """
    if ax.traits.continuous:
        edges = ", ".join(f"{e:.{precision}e}" for e in ax.edges)
        coords = [ax.edges[0], *ax.centers, ax.edges[-1]]
        return "d", edges, np.asarray(coords, dtype=np.float64)
    values = list(ax)
    if all(isinstance(v, str) for v in values):
        quoted = (v.replace("\\", "\\\\").replace('"', '\\"') for v in values)
        edges = ", ".join(f'"{v}"' for v in quoted)
        return "s", edges, np.arange(len(values) + 1, dtype=np.float64)
    edges = ", ".join(str(int(v)) for v in values)
    return "i", edges, np.asarray([0, *values], dtype=np.float64)


def _lines(path: str, h: bh.Histogram[Any], precision: int) -> Iterator[str]:
    axes = list(h.axes)
    ndim = len(axes)
    data, profile = _columns(h)
    info = [_axis_info(ax, precision) for ax in axes]
    types = "".join(kind for kind, _, _ in info)

    # These are the types that YODA can read
    base = "Profile" if profile else "Histo"
    if not profile and ndim == 0:
        name = "Counter"
    elif ndim in {1, 2} or (not profile and types == "ddd"):
        continuous = set(types) <= {"d"}
        name = f"{base}{ndim}D" if continuous else f"Binned{base}<{','.join(types)}>"
    else:
        msg = (
            f"YODA does not support a {ndim}D {base.lower()} with axis types {types!r}"
        )
        raise ValueError(msg)

    annotations = {}
    metadata = h.__dict__.get("metadata")
    if isinstance(metadata, Mapping):
        annotations = {str(k): str(v).replace("\n", " ") for k, v in metadata.items()}
    annotations["Path"] = path
    annotations["Title"] = getattr(h, "label", None) or ""
    annotations["Type"] = name

    tag = f"YODA_{name.upper()}_V3"
    yield f"BEGIN {tag} {path}"
    for key in sorted(annotations):
        yield f"{key}: {annotations[key]}".rstrip()
    yield "---"

    data = _yoda_layout(data, axes)
    sumw = data[..., 0]
    # Moments are computed from the bin coordinates
    coords: list[Array] = []
    firsts: list[Array] = []
    seconds: list[Array] = []
    for i, (_, edges, coord) in enumerate(info):
        yield f"Edges(A{i + 1}): [{edges}]"
        x = coord.reshape([-1 if j == i else 1 for j in range(ndim)])
        coords.append(x)
        firsts.append(sumw * x)
        seconds.append(sumw * x**2)
    if profile:
        firsts.append(data[..., 3])
        seconds.append(data[..., 4])

    header = ["# sumW", "sumW2"]
    columns = [sumw, data[..., 1]]
    for i, (first, second) in enumerate(zip(firsts, seconds, strict=True)):
        header += [f"sumW(A{i + 1})", f"sumW2(A{i + 1})"]
        columns += [first, second]
    for i, x in enumerate(coords):
        for j in range(i + 1, len(firsts)):
            header.append(f"sumW(A{i + 1},A{j + 1})")
            columns.append(firsts[j] * x)
    header.append("numEntries")
    columns.append(data[..., 2])

    width = precision + 7
    yield "\t".join(item.ljust(width) for item in header).rstrip()
    # The first axis changes fastest
    table = np.stack([np.ravel(c, order="F") for c in columns], axis=1)
    row_format = "\t".join([f"{{:<{width}.{precision}e}}"] * len(columns))
    for row in table:
        yield row_format.format(*row).rstrip()
    yield f"END {tag}"
    yield ""


def dumps(hists: Mapping[str, bh.Histogram[Any]], /, *, precision: int = 6) -> str:
    """
    Write a dict of path to histogram to a string in the YODA 2 text format.

    ``precision`` is the number of digits after the decimal point. The default
    is the YODA default; use 17 for a lossless round trip. A dict in
    ``.metadata`` is written as annotations, and ``.label`` is the title.
    """
    return "".join(
        f"{line}\n" for path, h in hists.items() for line in _lines(path, h, precision)
    )


def write(
    file: str | os.PathLike[str],
    hists: Mapping[str, bh.Histogram[Any]],
    /,
    *,
    precision: int = 6,
) -> None:
    """
    Write histograms to a YODA text file. The file is compressed if the name
    ends with ``.gz``. See :func:`dumps`.
    """
    data = dumps(hists, precision=precision).encode()
    if str(file).endswith(".gz"):
        data = gzip.compress(data)
    Path(file).write_bytes(data)
