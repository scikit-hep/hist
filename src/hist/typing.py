from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if sys.version_info < (3, 8):
    from typing_extensions import Literal, Protocol, SupportsIndex
else:
    from typing import Literal, Protocol, SupportsIndex

if TYPE_CHECKING:
    from numpy import ufunc as Ufunc
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any
    Ufunc = Any


__all__ = ("Literal", "Protocol", "SupportsIndex", "Ufunc", "ArrayLike")


def __dir__() -> tuple[str, ...]:
    return __all__
