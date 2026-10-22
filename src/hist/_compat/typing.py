from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from numpy import ufunc as Ufunc
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any
    Ufunc = Any


__all__ = ["ArrayLike", "Self", "Ufunc"]


def __dir__() -> list[str]:
    return __all__
