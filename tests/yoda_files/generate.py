"""
Regenerate the reference files with YODA 2 (``conda install -c conda-forge yoda``).
"""

from __future__ import annotations

from pathlib import Path

import yoda

DIR = Path(__file__).parent.resolve()

objs = []

h = yoda.Histo1D(3, 0, 1, "/h1", "title one")
h.setAnnotation("Custom", "hello")
h.fill(0.1)
h.fill(0.5, 2.0)
h.fill(-1)
h.fill(7, 3)
objs.append(h)

h = yoda.Histo2D(2, 0, 1, 3, 0, 3, "/h2")
h.fill(0.1, 0.5)
h.fill(0.7, 2.5, 2.0)
h.fill(-1, 9)
objs.append(h)

p = yoda.Profile1D(3, 0, 1, "/p1")
p.fill(0.1, 5)
p.fill(0.5, 3, 2.0)
p.fill(0.5, 4, 1.0)
objs.append(p)

p = yoda.Profile2D(2, 0, 1, 2, 0, 2, "/p2")
p.fill(0.1, 0.5, 5)
objs.append(p)

c = yoda.Counter("/c")
c.fill(2.0)
c.fill(3.0)
objs.append(c)

legacy = list(objs)

h = yoda.BinnedHisto1D([1, 2, 5], "/hint")
h.fill(2)
h.fill(3)
objs.append(h)

h = yoda.BinnedHisto1D(["a", "b"], "/hstr")
h.fill("b", 2.0)
objs.append(h)

h = yoda.BinnedHisto2D(["a", "b"], [0.0, 1.0, 2.0], "/hmix")
h.fill("a", 1.5)
h.fill("c", 0.5, 3.0)
objs.append(h)

h = yoda.Histo3D(2, 0, 1, 2, 0, 1, 2, 0, 1, "/h3")
h.fill(0.1, 0.7, 0.2)
objs.append(h)

e = objs[0].mkEstimate("/e1")
e.bin(2).setErr((-0.5, 1.5), "stat")
e.bin(2).setErr((-2.0, 2.0), "sys")
objs += [e, objs[1].mkEstimate("/e2"), c.mkEstimate("/e0"), objs[0].mkScatter("/s1")]


def clean(path: Path) -> None:
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    path.write_text("\n".join(lines).strip() + "\n")


yoda.writeYODA(objs, str(DIR / "yoda2.yoda"))
yoda.writeYODA1(legacy, str(DIR / "yoda1.yoda"))
clean(DIR / "yoda2.yoda")
clean(DIR / "yoda1.yoda")
