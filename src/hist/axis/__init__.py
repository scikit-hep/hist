import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boost_histogram.axis as bha

import hist
from hist.axestuple import ArrayTuple, NamedAxesTuple

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

from . import transform

__all__ = (
    "AxisProtocol",
    "AxesMixin",
    "Regular",
    "Variable",
    "Integer",
    "IntCategory",
    "StrCategory",
    "Boolean",
    "transform",
    "NamedAxesTuple",
    "ArrayTuple",
)


def __dir__() -> Tuple[str, ...]:
    return __all__


class CoreAxisProtocol(Protocol):
    metadata: Dict[str, Any]


class AxisProtocol(Protocol):
    metadata: Any

    @property
    def name(self) -> str:
        ...

    label: str
    _ax: CoreAxisProtocol


class AxesMixin:
    __slots__ = ()

    # Support mixing before or after a bh class
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore

    @property
    def name(self: AxisProtocol) -> str:
        """
        Get the name for the Regular axis
        """
        return self._ax.metadata.get("name", "")

    @property
    def label(self: AxisProtocol) -> str:
        """
        Get or set the label for the Regular axis
        """
        return self._ax.metadata.get("label", "") or self.name

    @label.setter
    def label(self: AxisProtocol, value: str) -> None:
        self._ax.metadata["label"] = value

    def _repr_args_(self: AxisProtocol) -> List[str]:
        """
        Return options for use in repr.
        """
        ret: List[str] = super()._repr_args_()  # type: ignore

        if self.name:
            ret.append(f"name={self.name!r}")
        if self.label:
            ret.append(f"label={self.label!r}")

        return ret


class Regular(AxesMixin, bha.Regular, family=hist):
    __slots__ = ()

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        flow: bool = True,
        underflow: Optional[bool] = None,
        overflow: Optional[bool] = None,
        growth: bool = False,
        circular: bool = False,
        transform: Optional[bha.transform.AxisTransform] = None,
        __dict__: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            bins,
            start,
            stop,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
            circular=circular,
            transform=transform,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label


class Boolean(AxesMixin, bha.Boolean, family=hist):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        __dict__: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            metadata=metadata,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label


class Variable(AxesMixin, bha.Variable, family=hist):
    __slots__ = ()

    def __init__(
        self,
        edges: Iterable[float],
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        flow: bool = True,
        underflow: Optional[bool] = None,
        overflow: Optional[bool] = None,
        growth: bool = False,
        circular: bool = False,
        __dict__: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            edges,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
            circular=circular,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label


class Integer(AxesMixin, bha.Integer, family=hist):
    __slots__ = ()

    def __init__(
        self,
        start: int,
        stop: int,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        flow: bool = True,
        underflow: Optional[bool] = None,
        overflow: Optional[bool] = None,
        growth: bool = False,
        circular: bool = False,
        __dict__: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            start,
            stop,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
            circular=circular,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label


class IntCategory(AxesMixin, bha.IntCategory, family=hist):
    __slots__ = ()

    def __init__(
        self,
        categories: Iterable[int],
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        growth: bool = False,
        __dict__: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            categories,
            metadata=metadata,
            growth=growth,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label


class StrCategory(AxesMixin, bha.StrCategory, family=hist):
    __slots__ = ()

    def __init__(
        self,
        categories: Iterable[str],
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        growth: bool = False,
        __dict__: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            categories,
            metadata=metadata,
            growth=growth,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label
