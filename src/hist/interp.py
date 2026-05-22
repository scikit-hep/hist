from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from hist import Hist


@dataclasses.dataclass
class Interp:
    x: np.typing.ArrayLike
    _hist: Hist

    def __call__(self, x: np.typing.ArrayLike, _hist: Hist) -> Any:
        raise NotImplementedError()


class Linear(Interp):
    def __call__(self, x: np.typing.ArrayLike, _hist: Hist) -> Any:
        return np.interp(x, _hist.axes[0].centers, _hist.values())  # type: ignore
