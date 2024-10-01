from __future__ import annotations

from typing import Generator

import numpy as np

from .. import Hist, overflow, underflow

__all__ = ["to_yoda_lines", "from_yoda_str"]


def to_yoda_lines(input: dict[str, Hist]) -> Generator[str, None, None]:
    for path, h in input.items():
        if not isinstance(h, Hist):
            raise TypeError("Only Hist objects are supported")

        if h.ndim == 1:
            yield from _to_single_yoda_1d(path, h)
        elif h.ndim == 2:
            yield from _to_single_yoda_2d(path, h)
        else:
            raise TypeError("Only 1D and 2D histograms are supported")

        yield ""
        yield ""


def print_line_1d(
    lower: str | float,
    upper: str | float,
    sumw: float,
    sumw2: float,
    sumwx: float,
    sumwx2: float,
    num_entries: float,
) -> str:
    if isinstance(lower, float):
        lower = format(lower, ".6e")
    if isinstance(upper, float):
        upper = format(upper, ".6e")
    return f"{lower:8}\t{upper:8}\t{sumw:.6e}\t{sumw2:.6e}\t{sumwx:.6e}\t{sumwx2:.6e}\t{num_entries:.6e}"


def print_line_2d(
    lower: str | float,
    upper: str | float,
    sumw: float,
    sumw2: float,
    sumwx: float,
    sumwx2: float,
    sumwy: float,
    sumwy2: float,
    sumwxy: float,
    num_entries: float,
) -> str:
    if isinstance(lower, float):
        lower = format(lower, ".6e")
    if isinstance(upper, float):
        upper = format(upper, ".6e")
    return f"{lower:8}\t{upper:8}\t{sumw:.6e}\t{sumw2:.6e}\t{sumwx:.6e}\t{sumwx2:.6e}\t{sumwy:.6e}\t{sumwy2:.6e}\t{sumwxy:.6e}\t{num_entries:.6e}"


def _to_single_yoda_1d(path: str, h: Hist) -> Generator[str, None, None]:
    # Unpack single axis & values from histogram
    (axis,) = h.axes
    data = h.values()

    yield f"BEGIN YODA_HISTO1D_V2 {path}"
    yield f"Path: {path}"
    yield f"Title: {h.name}"
    yield "Type: Histo1D"

    # Add histogram data
    yield "---"

    # Calculate area and mean
    area = h.sum(flow=True)
    mean = np.sum(axis.centers * data) / np.sum(data)
    h_underflow = h[underflow]
    h_overflow = h[overflow]

    assert isinstance(area, float)
    assert isinstance(mean, float)
    assert isinstance(h_underflow, float)
    assert isinstance(h_overflow, float)

    # Add area and mean to YODA string
    yield f"# Mean: {mean:.6e}"
    yield f"# Area: {area:.6e}"

    yield "# ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries"
    yield print_line_1d("Total", "Total", area, area, mean, mean, area)
    yield print_line_1d(
        "Underflow",
        "Underflow",
        h_underflow,
        h_underflow,
        h_underflow,
        h_underflow,
        h_underflow,
    )
    yield print_line_1d(
        "Overflow",
        "Overflow",
        h_overflow,
        h_overflow,
        h_overflow,
        h_overflow,
        h_overflow,
    )

    yield "# xlow\txhigh\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries"

    # Add histogram bins
    for xlow, xhigh, value in zip(axis.edges[:-1], axis.edges[1:], data):
        yield print_line_1d(
            xlow,
            xhigh,
            value,
            value,
            (xlow + xhigh) * 0.5 * value,
            (xlow + xhigh) * 0.5 * value**2,
            value,
        )

    yield "END YODA_HISTO1D_V2"


def _to_single_yoda_2d(path: str, h: Hist) -> Generator[str, None, None]:
    yield f"BEGIN YODA_HISTO2D_V2 {path}"
    yield f"Path: {path}"
    if h.name:
        yield f"Title: {h.name}"
    yield "Type: Histo2D"

    # Add histogram data
    yield "---"

    (x_axis, y_axis) = h.axes
    data = h.values()

    # Calculate mean and volume
    mean_x = np.sum(x_axis.centers * data) / np.sum(data)
    mean_y = np.sum(y_axis.centers * data) / np.sum(data)
    volume = np.sum(data) * x_axis.widths[0] * y_axis.widths[0]

    # Add mean, volume, and ID to YODA string
    yield f"# Mean: ({mean_x:.6e}, {mean_y:.6e})"
    yield f"# Volume: {volume:.6e}"
    yield "# ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tsumwy\tsumwy2\tsumwxy\tnumEntries"
    yield print_line_2d(
        "Total",
        "Total",
        volume,
        volume,
        mean_x * volume,
        mean_x * volume,
        mean_y * volume,
        mean_y * volume,
        volume,
        volume,
    )
    yield "# 2D outflow persistency not currently supported until API is stable"
    yield "# xlow\txhigh\tylow\tyhigh\tsumw\tsumw2\tsumwx\tsumwx2\tsumwy\tsumwy2\tsumwxy\tnumEntries"

    # Add histogram bins
    x_bin_edges = h.axes[0].edges
    y_bin_edges = h.axes[1].edges

    for i in range(len(x_bin_edges) - 1):
        for j in range(len(y_bin_edges) - 1):
            xlow, xhigh = x_bin_edges[i], x_bin_edges[i + 1]
            ylow, yhigh = y_bin_edges[j], y_bin_edges[j + 1]
            sumw = data[i, j]
            sumw2 = sumw * sumw
            sumwx = sumw * (xlow + xhigh) * 0.5
            sumwx2 = sumw * (xlow + xhigh) * 0.5 * (xlow + xhigh) * 0.5
            sumwy = sumw * (ylow + yhigh) * 0.5
            sumwy2 = sumw * (ylow + yhigh) * 0.5 * (ylow + yhigh) * 0.5
            sumwxy = sumw * (xlow + xhigh) * 0.5 * (ylow + yhigh) * 0.5
            numEntries = sumw
            yield f"{xlow:.6e}\t{xhigh:.6e}\t{ylow:.6e}\t{yhigh:.6e}\t{sumw:.6e}\t{sumw2:.6e}\t{sumwx:.6e}\t{sumwx2:.6e}\t{sumwy:.6e}\t{sumwy2:.6e}\t{sumwxy:.6e}\t{numEntries:.6e}"

    yield "END YODA_HISTO2D_V2"


def from_yoda_str(input: str) -> dict[str, tuple[str, str, str]]:
    yoda_dict = {}
    lines = input.strip().split("\n")
    num_lines = len(lines)
    i = 0
    while i < num_lines:
        line = lines[i]

        if line.startswith("BEGIN"):
            path = line.split()[2]
            class_name = line.split()[1][:-3]

            body = ""
            header = ""

            while i < num_lines and lines[i].strip() == "---":
                header += lines[i] + "\n"
                i += 1

            while i < num_lines and not lines[i].startswith("END"):
                body += lines[i] + "\n"
                i += 1

            body += lines[i]
            yoda_dict[path] = (class_name, header, body)

        i += 1
    return yoda_dict
