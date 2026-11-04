from __future__ import annotations

from pathlib import Path

import hist
import hist.io.yoda

DIR = Path(__file__).parent.resolve()


# python -c "import hist.io.yoda; print('\n'.join(hist.io.yoda.to_yoda_lines({'h': hist.Hist.new.Reg(10, 0, 1, name='x').Double().fill([0.1, 0.2, 0.3])})), end='')" > tests/yoda/simple.yoda
def test_simple_yoda():
    h = hist.Hist.new.Reg(10, 0, 1, name="x").Double()
    h.fill([0.1, 0.2, 0.3])

    txt = "\n".join(hist.io.yoda.to_yoda_lines({"h": h})).strip()
    simple_yoda = DIR.joinpath("yoda/simple.yoda").read_text().strip()
    assert txt == simple_yoda
