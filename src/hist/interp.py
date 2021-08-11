from __future__ import annotations

from typing import Any

from .basehist import BaseHist

try:
    from scipy import interpolate
except ModuleNotFoundError:
    from sys import stderr

    print(
        "hist.interp requires scipy. Please install hist[plot] or manually install scipy.",
        file=stderr,
    )
    raise

__all__ = ("Linear", "Cubic")


def __dir__() -> tuple[str, ...]:
    return __all__


def Linear(
    h: BaseHist, **kwargs: Any
) -> Any:  # Callable[[Any], np.ndarray] doesn't work
    r"""
    Linear interpolator using scipy.

    Args:
        h: the histogram whose values are going to be interpolated.

    Returns:
        The callable interpolator.
    """

    if "kind" in kwargs:
        raise ValueError("Kind is set default as linear")

    return interpolate.interp1d(h.axes[0].centers, h.values(), kind="linear", **kwargs)


def Cubic(
    h: BaseHist, **kwargs: Any
) -> Any:  # Callable[[Any], np.ndarray] doesn't work
    r"""
    Cubic interpolator using scipy.

    Args:
        h: the histogram whose values are going to be interpolated.

    Returns:
        The callable interpolator.
    """
    if "kind" in kwargs:
        raise ValueError("Kind is set default as cubic")

    return interpolate.interp1d(h.axes[0].centers, h.values(), kind="cubic", **kwargs)
