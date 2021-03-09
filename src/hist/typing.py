import sys
from typing import TYPE_CHECKING, Any

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, SupportsIndex
else:
    from typing import Protocol, SupportsIndex

if TYPE_CHECKING:
    from numpy import ufunc as Ufunc
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any
    Ufunc = Any


__all__ = ("Protocol", "SupportsIndex", "Ufunc", "ArrayLike")
