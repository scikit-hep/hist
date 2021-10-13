from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

from . import axis, storage
from .axis import AxisProtocol
from .axis.transform import AxisTransform

if TYPE_CHECKING:
    from .basehist import BaseHist


class QuickConstruct:
    """
    Create a quick construct instance. This is the "base" quick constructor; it will
    always require at least one axes to be added before allowing a storage or fill to be performed.
    """

    __slots__ = (
        "hist_class",
        "axes",
    )

    def __repr__(self) -> str:
        inside = ", ".join(repr(ax) for ax in self.axes)
        return f"{self.__class__.__name__}({self.hist_class.__name__}, {inside})"

    def __init__(self, hist_class: type[BaseHist], *axes: AxisProtocol) -> None:
        self.hist_class = hist_class
        self.axes = axes

    def Regular(
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
        transform: AxisTransform | None = None,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(
                bins,
                start,
                stop,
                name=name,
                label=label,
                metadata=metadata,
                flow=flow,
                underflow=underflow,
                overflow=overflow,
                growth=growth,
                circular=circular,
                transform=transform,
                __dict__=__dict__,
            ),
        )

    Reg = Regular

    def Sqrt(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(
                bins,
                start,
                stop,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                transform=axis.transform.sqrt,
            ),
        )

    def Log(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(
                bins,
                start,
                stop,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                transform=axis.transform.log,
            ),
        )

    def Pow(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        power: float,
        metadata: Any = None,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(
                bins,
                start,
                stop,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                transform=axis.transform.Pow(power),
            ),
        )

    def Func(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        forward: Callable[[float], float],
        inverse: Callable[[float], float],
        metadata: Any = None,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(
                bins,
                start,
                stop,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                transform=axis.transform.Function(forward, inverse),
            ),
        )

    def Boolean(
        self,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Boolean(
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
            ),
        )

    Bool = Boolean

    def Variable(
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
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Variable(
                edges,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                flow=flow,
                underflow=underflow,
                overflow=overflow,
                growth=growth,
                circular=circular,
            ),
        )

    Var = Variable

    def Integer(
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
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Integer(
                start,
                stop,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                flow=flow,
                underflow=underflow,
                overflow=overflow,
                growth=growth,
                circular=circular,
            ),
        )

    Int = Integer

    def IntCategory(
        self,
        categories: Iterable[int],
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        growth: bool = False,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.IntCategory(
                categories,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                growth=growth,
            ),
        )

    IntCat = IntCategory

    def StrCat(
        self,
        categories: Iterable[str],
        *,
        name: str = "",
        label: str = "",
        metadata: Any = None,
        growth: bool = False,
        __dict__: dict[str, Any] | None = None,
    ) -> ConstructProxy:
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.StrCategory(
                categories,
                name=name,
                label=label,
                metadata=metadata,
                __dict__=__dict__,
                growth=growth,
            ),
        )

    StrCategory = StrCat


class ConstructProxy(QuickConstruct):
    __slots__ = ()

    def Double(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.Double(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    def Int64(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.Int64(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    def AtomicInt64(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.AtomicInt64(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    def Weight(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.Weight(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    def Mean(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.Mean(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    def WeightedMean(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.WeightedMean(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    def Unlimited(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> BaseHist:
        return self.hist_class(
            *self.axes,
            storage=storage.Unlimited(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )


class MetaConstructor(type):
    @property
    def new(cls: type[BaseHist]) -> QuickConstruct:  # type: ignore[misc]
        return QuickConstruct(cls)
