# -*- coding: utf-8 -*-
import numpy as np

from .svgutils import svg, html, rect, line, text, div, polyline, polygon, circle


def _desc_hist(h):
    main_sum = h.sum()
    flow_too_sum = h.sum(flow=True)

    output = "<br/>\n".join(str(h) for h in h.axes)
    output += '<br/>\n<hr style="margin-top:.2em; margin-bottom:.2em;"/>\n'
    output += f"{h._storage_type()} "
    output += f"Σ={main_sum}"
    if main_sum != flow_too_sum:
        output += f" <em>({flow_too_sum} with flow)</em>"
    output += "\n"
    return output


def html_hist(h, function):
    left_column = div(function(h), style="width:290px;")
    right_column = div(_desc_hist(h), style="flex=grow:1;")

    container = div(
        left_column, right_column, style="display:flex; align-items:center;"
    )

    return html(container)


def make_text(txt, **kwargs):
    style = "fill:currentColor;"
    kwargs["style"] = style + kwargs.get("style", "")
    if isinstance(txt, float):
        txt = format(txt, ".3g")
    return text(txt, text_anchor="middle", **kwargs)


def make_ax_text(ax, **kwargs):
    style = "" if ax.label else "font-family: monospace;"
    kwargs["style"] = style + kwargs.get("style", "")
    return make_text(ax.label or ax.name, **kwargs)


def svg_hist_1d(h):
    width = 250
    height = 100

    assert h.ndim == 1, "Must be 1D"
    assert not h.axes[0].options.circular, "Must not be circular"

    (edges,) = h.axes.edges
    norm_edges = (edges - edges[0]) / (edges[-1] - edges[0])
    density = h.density()
    max_dens = np.max(density) or 1
    norm_vals = density / max_dens

    arr = np.empty((2, len(norm_vals) * 2 + 2), dtype=float)
    arr[0, 0:-1:2] = arr[0, 1::2] = width * norm_edges
    arr[1, 1:-2:2] = arr[1, 2:-1:2] = -height * norm_vals
    arr[1, 0] = arr[1, -1] = 0

    points = " ".join(f"{x:3g},{y:.3g}" for x, y in arr.T)
    bins = polyline(points=points, style="fill:none; stroke:currentColor;")

    ax_line = line(
        x1=-5,
        y1=0,
        x2=width + 5,
        y2=0,
        style="fill:none;stroke-width:2;stroke:currentColor",
    )

    lower = make_text(edges[0], x=0, y=15)
    upper = make_text(edges[-1], x=width, y=15)
    label = make_ax_text(
        h.axes[0],
        x=width / 2,
        y=15,
    )

    return svg(
        ax_line,
        lower,
        upper,
        label,
        bins,
        viewBox=f"-10 {-height-5} {width+20} {height+20}",
    )


def svg_hist_1d_c(h):
    width = 250
    height = 250
    radius = 100
    inner_radius = 20

    assert h.ndim == 1, "Must be 1D"
    assert h.axes[0].options.circular, "Must be circular"

    (edges,) = h.axes.edges
    norm_edges = (edges - edges[0]) / (edges[-1] - edges[0]) * np.pi * 2
    density = h.density()
    max_dens = np.max(density) or 1
    norm_vals = density / max_dens

    arr = np.empty((2, len(norm_vals) * 2), dtype=float)
    arr[0, :-1:2] = arr[0, 1::2] = norm_edges[:-1]
    arr[1, :-1:2] = arr[1, 1::2] = inner_radius + norm_vals * (radius - inner_radius)
    arr[1] = np.roll(arr[1], shift=1)

    xs = arr[1] * np.cos(arr[0])
    ys = arr[1] * np.sin(arr[0])

    points = " ".join(f"{x:3g},{y:.3g}" for x, y in zip(xs, ys))
    bins = polygon(points=points, style="fill:none; stroke:currentColor;")

    center = circle(
        cx="0",
        cy="0",
        r=f"{inner_radius}",
        style="fill:none;stroke-width:2;stroke:currentColor",
    )

    return svg(bins, center, viewBox=f"{-width/2} {-height/2} {width} {height}")


def svg_hist_2d(h):
    width = 250
    height = 250
    assert h.ndim == 2, "Must be 2D"

    e0, e1 = (h.axes[0].edges, h.axes[1].edges)
    ex = (e0 - e0[0]) / (e0[-1] - e0[0]) * width
    ey = -(e1 - e1[0]) / (e1[-1] - e1[0]) * height

    density = h.density()
    max_dens = np.max(density) or 1
    norm_vals = density / max_dens

    boxes = []
    for r, (up_edge, bottom_edge) in enumerate(zip(ey[:-1], ey[1:])):
        ht = up_edge - bottom_edge
        for c, (left_edge, right_edge) in enumerate(zip(ex[:-1], ex[1:])):
            opacity = norm_vals[c, r]
            wt = left_edge - right_edge
            boxes.append(
                rect(
                    x=left_edge,
                    y=bottom_edge,
                    width=round(abs(wt), 3),
                    height=round(abs(ht), 3),
                    opacity=opacity,
                    fill="currentColor",
                    stroke_width=0.1,
                )
            )

    texts = [
        make_text(e0[0], x=0, y=13),
        make_text(e0[-1], x=width, y=13),
        make_text(e1[0], x=-10, y=0),
        make_text(e1[-1], x=-10, y=-height),
        make_ax_text(
            h.axes[0],
            x=width / 2,
            y=13,
        ),
        make_ax_text(
            h.axes[1],
            x=-10,
            y=-height / 2,
            transform=f"rotate(-90,{-10},{-height/2})",
        ),
    ]

    return svg(*texts, *boxes, viewBox=f"{-20} {-height - 20} {width+40} {height+40}")


def svg_hist_nd(h):
    assert h.ndim > 2, "Must be more than 2D"

    width = 200
    height = 200

    boxes = [
        rect(
            x=20 * i,
            y=20 * i,
            width=width - 40,
            height=height - 40,
            style="fill:white;opacity:.5;stroke-width:2;stroke:currentColor;",
        )
        for i in range(3)
    ]

    nd = text(
        f"{h.ndim}D",
        x=height / 2 + 20,
        y=width / 2 + 20,
        style="font-size: 26pt; font-family: verdana; font-style: bold; fill: black;",
        text_anchor="middle",
        alignment_baseline="middle",
    )

    return svg(*boxes, nd, viewBox=f"-10 -10 {height + 20} {width + 20}")
