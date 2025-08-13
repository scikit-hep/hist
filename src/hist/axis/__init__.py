from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Any, Protocol

import boost_histogram.axis as bha

import hist

from ..axestuple import ArrayTuple, NamedAxesTuple
from . import transform

__all__ = (
    "ArrayTuple",
    "AxesMixin",
    "AxisProtocol",
    "Boolean",
    "IntCategory",
    "Integer",
    "NamedAxesTuple",
    "Regular",
    "StrCategory",
    "Variable",
    "transform",
)


def __dir__() -> tuple[str, ...]:
    return __all__


class CoreAxisProtocol(Protocol):
    metadata: dict[str, Any]  # boost-histogram < 1.6
    raw_metadata: dict[str, Any]


class AxisProtocol(Protocol):
    @property
    def name(self) -> str: ...

    label: str
    _ax: CoreAxisProtocol

    @property
    def _raw_metadata(self) -> dict[str, Any]: ...


class AxesMixin:
    __slots__ = ()

    # Support mixing before or after a bh class
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @property
    def _raw_metadata(self: AxisProtocol) -> dict[str, Any]:
        # boost-histogram < 1.6
        if hasattr(self._ax, "metadata"):
            return self._ax.metadata
        return self._ax.raw_metadata

    @property
    def name(self: AxisProtocol) -> str:
        """
        Get the name for the Regular axis
        """
        return typing.cast(str, self._raw_metadata.get("name", ""))

    @property
    def label(self: AxisProtocol) -> str:
        """
        Get or set the label for the Regular axis
        """
        return self._raw_metadata.get("label", "") or self.name

    @label.setter
    def label(self: AxisProtocol, value: str) -> None:
        self._raw_metadata["label"] = value

    def _repr_args_(self: AxisProtocol) -> list[str]:
        """
        Return options for use in repr.
        """
        ret: list[str] = super()._repr_args_()  # type: ignore[misc]

        if self.name:
            ret.append(f"name={self.name!r}")
        if self.label and self.label != self.name:
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
        underflow: bool | None = None,
        overflow: bool | None = None,
        growth: bool = False,
        circular: bool = False,
        # pylint: disable-next=redefined-outer-name
        transform: bha.transform.AxisTransform | None = None,
        __dict__: dict[str, Any] | None = None,
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
        self._raw_metadata["name"] = name
        self.label: str = label


class Boolean(AxesMixin, bha.Boolean, family=hist):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            metadata=metadata,
            __dict__=__dict__,
        )
        self._raw_metadata["name"] = name
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
        underflow: bool | None = None,
        overflow: bool | None = None,
        growth: bool = False,
        circular: bool = False,
        __dict__: dict[str, Any] | None = None,
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
        self._raw_metadata["name"] = name
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
        underflow: bool | None = None,
        overflow: bool | None = None,
        growth: bool = False,
        circular: bool = False,
        __dict__: dict[str, Any] | None = None,
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
        self._raw_metadata["name"] = name
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
        flow: bool = True,
        overflow: bool | None = None,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        has_flow = flow if overflow is None else overflow
        super().__init__(
            categories,
            metadata=metadata,
            growth=growth,
            overflow=has_flow,
            __dict__=__dict__,
        )
        self._raw_metadata["name"] = name
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
        flow: bool = True,
        overflow: bool | None = None,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        has_flow = flow if overflow is None else overflow
        super().__init__(
            categories,
            metadata=metadata,
            growth=growth,
            overflow=has_flow,
            __dict__=__dict__,
        )
        self._raw_metadata["name"] = name
        self.label: str = label
