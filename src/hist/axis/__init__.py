import sys
from typing import Any, Dict, List, Optional, Union

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


class CoreAxisProtocol(Protocol):
    metadata: Dict[str, Any]


class AxisProtocol(Protocol):
    metadata: Any
    name: str
    label: str
    _ax: CoreAxisProtocol


class AxesMixin:
    __slots__ = ()

    # Support mixing before or after a bh class
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # type: ignore

    @property
    def name(self: AxisProtocol) -> str:
        """
        Get or set the name for the Regular axis
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

    def _repr_kwargs(self: AxisProtocol):
        """
        Return options for use in repr.
        """
        ret = super()._repr_kwargs()  # type: ignore

        if self.name:
            ret += f", name={self.name!r}"
        if self.label:
            ret += f", label={self.label!r}"

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
        transform: bha.transform.Function = None,
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
        )
        self._ax.metadata["name"] = name
        self.label = label


class Boolean(AxesMixin, bha.Boolean, family=hist):
    __slots__ = ()

    def __init__(
        self, *, name: str = "", label: str = "", metadata: Any = None
    ) -> None:
        super().__init__(metadata=metadata)
        self._ax.metadata["name"] = name
        self.label = label


class Variable(bha.Variable, AxesMixin, family=hist):
    __slots__ = ()

    def __init__(
        self,
        edges: Union[range, List[float]],
        *,
        name: str = "",
        label: str = "",
        flow: bool = True,
        underflow: Optional[bool] = None,
        overflow: Optional[bool] = None,
        growth: bool = False,
        metadata: Any = None,
    ) -> None:
        super().__init__(
            edges,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
        )
        self._ax.metadata["name"] = name
        self.label = label


class Integer(bha.Integer, AxesMixin, family=hist):
    __slots__ = ()

    def __init__(
        self,
        start: int,
        stop: int,
        *,
        name: str = "",
        label: str = "",
        flow: bool = True,
        underflow: Optional[bool] = None,
        overflow: Optional[bool] = None,
        growth: bool = False,
        metadata: Any = None,
    ) -> None:
        super().__init__(
            start,
            stop,
            metadata=metadata,
            underflow=flow if underflow is None else underflow,
            overflow=flow if overflow is None else overflow,
            growth=growth,
        )
        self._ax.metadata["name"] = name
        self.label = label


class IntCategory(bha.IntCategory, AxesMixin, family=hist):
    __slots__ = ()

    def __init__(
        self,
        categories: Union[range, List[int]] = None,
        *,
        name: str = "",
        label: str = "",
        growth: bool = False,
        metadata: Any = None,
    ) -> None:
        super().__init__(categories, metadata=metadata, growth=growth)
        self._ax.metadata["name"] = name
        self.label = label


class StrCategory(bha.StrCategory, AxesMixin, family=hist):
    __slots__ = ()

    def __init__(
        self,
        categories: Union[str, List[str]] = None,
        *,
        name: str = "",
        label: str = "",
        growth: bool = False,
        metadata: Any = None,
    ) -> None:
        super().__init__(categories, metadata=metadata, growth=growth)
        self._ax.metadata["name"] = name
        self.label = label
