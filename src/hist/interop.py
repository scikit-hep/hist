from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from _compat.typing import ArrayLike, Protocol
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class HistogramModuleProtocol(Protocol[T, U]):
    def unpack(self, obj: T) -> dict[str, U] | None:
        ...

    def broadcast_and_flatten(self, objects: Sequence[U]) -> tuple[ArrayLike]:
        ...


_histogram_modules: dict[type, HistogramModuleProtocol] = {}


M = TypeVar("M", bound=HistogramModuleProtocol)


def histogram_module_for(cls: type) -> Callable[[M], M]:
    """
    Register a histogram-module object for the given class.
    """

    def wrapper(obj: M) -> M:
        _histogram_modules[cls] = obj
        return obj

    return wrapper


def find_histogram_modules(*objects: Any) -> Iterator[HistogramModuleProtocol]:
    """
    Yield histogram-module objects that are known to support any of the given objects.
    """
    for arg in objects:
        try:
            yield arg._histogram_module_
        except AttributeError:
            # Find class exactly, or check subclasses
            for cls in type(arg).__mro__:
                try:
                    yield _histogram_modules[cls]
                except KeyError:
                    continue


def destructure(obj: Any) -> dict[str, Any] | None:
    """
    Pull out named histogram-module arrays from the given structured object as a mapping.
    The returned arrays should be compatible with `broadcast_and_flatten`.
    If the argument is not understood as a structured object by any histogram modules,
    return `None`.
    """
    for module in find_histogram_modules(obj):
        return module.unpack(obj)
    raise TypeError(f"No histogram module found for {obj!r}")


def broadcast_and_flatten(args: Sequence[Any]) -> tuple[ArrayLike]:
    """
    Convert the given histogram-module arrays into a set of consistent 1D NumPy arrays
    for histogram filling. For NumPy this entails broadcasting and flattening.
    """
    for module in find_histogram_modules(*args):
        result = module.broadcast_and_flatten(args)
        if result is not NotImplemented:
            return result

    raise TypeError(f"No histogram module found for {args!r}")


@histogram_module_for(np.ndarray)
class NumpyHistogramModule:
    @staticmethod
    def unpack(obj: np.ndarray) -> dict[str, np.ndarray] | None:
        if obj.dtype.names is None:
            return None
        else:
            return {k: obj[k] for k in obj.dtype.fields}

    @staticmethod
    def broadcast_and_flatten(args: Sequence[ArrayLike]) -> tuple[np.ndarray]:
        arrays = []
        for arg in args:
            # If we can't interpret this argument, it's not NumPy-friendly!
            try:
                arrays.append(np.asarray(arg))
            except (TypeError, ValueError):
                return NotImplemented

        return tuple([np.ravel(x) for x in np.broadcast_arrays(*arrays)])


try:
    import pandas as pd
except ImportError:
    ...
else:

    @histogram_module_for(pd.DataFrame)
    class PandasHistogramModule:
        @staticmethod
        def unpack(obj: pd.DataFrame) -> dict[str, pd.Series] | None:
            return obj.to_dict("series")

        @staticmethod
        def broadcast_and_flatten(
            args: Sequence[pd.Series | ArrayLike],
        ) -> tuple[np.ndarray]:
            arrays = []
            for arg in args:
                # If we can't interpret this argument, it's not NumPy-friendly!
                try:
                    arrays.append(np.asarray(arg))
                except (TypeError, ValueError):
                    return NotImplemented

            return tuple([np.ravel(x) for x in np.broadcast_arrays(*arrays)])
