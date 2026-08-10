from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from . import axis, storage

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np

    from .axis import AxisProtocol
    from .axis.transform import AxisTransform
    from .basehist import BaseHist
    from .hist import Hist
    from .namedhist import NamedHist

# Carries the originating histogram class (Hist, NamedHist, or a user subclass)
# through the construction chain so storage finalizers return the real subclass.
H = TypeVar("H", bound="BaseHist[Any]")


class QuickConstruct(Generic[H]):
    """
    Create a quick construct instance. This is the "base" quick constructor; it will
    always require at least one axes to be added before allowing a storage or fill to be performed.
    """

    __slots__ = (
        "axes",
        "hist_class",
    )

    def __repr__(self) -> str:
        inside = ", ".join(repr(ax) for ax in self.axes)
        return f"{self.__class__.__name__}({self.hist_class.__name__}, {inside})"

    def __init__(self, hist_class: type[H], *axes: AxisProtocol) -> None:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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
    ) -> ConstructProxy[H]:
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


class ConstructProxy(QuickConstruct[H]):
    __slots__ = ()

    # Each finalizer has self-type overloads that select the storage-typed
    # return (e.g. Hist[storage.Double]) for the known classes. Python typing
    # has no higher-kinded types, so user subclasses take the fallback, which
    # keeps the subclass but not the storage type.

    @overload
    def Double(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.Double]: ...
    @overload
    def Double(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.Double]: ...
    @overload
    def Double(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def Double(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.Double(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def Int64(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.Int64]: ...
    @overload
    def Int64(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.Int64]: ...
    @overload
    def Int64(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def Int64(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.Int64(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def AtomicInt64(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.AtomicInt64]: ...
    @overload
    def AtomicInt64(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.AtomicInt64]: ...
    @overload
    def AtomicInt64(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def AtomicInt64(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.AtomicInt64(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def Weight(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.Weight]: ...
    @overload
    def Weight(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.Weight]: ...
    @overload
    def Weight(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def Weight(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.Weight(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def Mean(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.Mean]: ...
    @overload
    def Mean(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.Mean]: ...
    @overload
    def Mean(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def Mean(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.Mean(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def WeightedMean(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.WeightedMean]: ...
    @overload
    def WeightedMean(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.WeightedMean]: ...
    @overload
    def WeightedMean(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def WeightedMean(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.WeightedMean(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def Unlimited(
        self: ConstructProxy[NamedHist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.Unlimited]: ...
    @overload
    def Unlimited(
        self: ConstructProxy[Hist[Any]],
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.Unlimited]: ...
    @overload
    def Unlimited(
        self,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def Unlimited(
        self,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.Unlimited(),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )

    @overload
    def MultiCell(
        self: ConstructProxy[NamedHist[Any]],
        /,
        nelem: int,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> NamedHist[storage.MultiCell]: ...
    @overload
    def MultiCell(
        self: ConstructProxy[Hist[Any]],
        /,
        nelem: int,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> Hist[storage.MultiCell]: ...
    @overload
    def MultiCell(
        self,
        /,
        nelem: int,
        *,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> H: ...
    def MultiCell(
        self,
        /,
        nelem: int,
        *,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> Any:
        return self.hist_class(
            *self.axes,
            storage=storage.MultiCell(nelem),
            metadata=metadata,
            data=data,
            label=label,
            name=name,
        )


class MetaConstructor(type):
    @property
    def new(cls: type[H]) -> QuickConstruct[H]:  # type: ignore[misc]
        return QuickConstruct(cls)
