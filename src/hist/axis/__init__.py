from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Any, Protocol

import boost_histogram as bh
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
    metadata: dict[str, Any]


class AxisProtocol(Protocol):
    metadata: Any

    @property
    def name(self) -> str: ...

    label: str
    _ax: CoreAxisProtocol


class AxesMixin:
    __slots__ = ()

    # Support mixing before or after a bh class
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @property
    def name(self: AxisProtocol) -> str:
        """
        Get the name for the Regular axis
        """
        return typing.cast(str, self._ax.metadata.get("name", ""))

    @property
    def label(self: AxisProtocol) -> str:
        """
        Get or set the label for the Regular axis
        """
        return self._ax.metadata.get("label", "") or self.name

    @label.setter
    def label(self: AxisProtocol, value: str) -> None:
        self._ax.metadata["label"] = value

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
        __dict__: dict[str, Any] | None = None,
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
        flow: bool = True,
        overflow: bool | None = None,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        has_flow = flow if overflow is None else overflow
        if tuple(int(x) for x in bh.__version__.split(".")[:2]) < (1, 4):
            if not has_flow:
                msg = "Boost-histogram 1.4+ required for flowless Category axes"
                raise TypeError(msg)
            kwargs = {}
        else:
            kwargs = {"overflow": has_flow}
        super().__init__(
            categories,
            metadata=metadata,
            growth=growth,
            **kwargs,
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
        flow: bool = True,
        overflow: bool | None = None,
        __dict__: dict[str, Any] | None = None,
    ) -> None:
        has_flow = flow if overflow is None else overflow
        if tuple(int(x) for x in bh.__version__.split(".")[:2]) < (1, 4):
            if not has_flow:
                msg = "Boost-histogram 1.4+ required for flowless Category axes"
                raise TypeError(msg)
            kwargs = {}
        else:
            kwargs = {"overflow": has_flow}
        super().__init__(
            categories,
            metadata=metadata,
            growth=growth,
            **kwargs,
            __dict__=__dict__,
        )
        self._ax.metadata["name"] = name
        self.label: str = label
