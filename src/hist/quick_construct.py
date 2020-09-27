# -*- coding: utf-8 -*-
from typing import Callable, TYPE_CHECKING, Type

from .axis import AxisProtocol
from . import storage, axis

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

    def __repr__(self):
        inside = ", ".join(repr(ax) for ax in self.axes)
        return f"{self.__class__.__name__}({self.hist_class.__name__}, {inside})"

    def __init__(self, hist_class: "Type[BaseHist]", *axes: AxisProtocol):
        self.hist_class = hist_class
        self.axes = axes

    def Reg(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class, *self.axes, axis.Regular(*args, **kwargs)
        )

    def Sqrt(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(*args, transform=axis.transform.sqrt, **kwargs),
        )

    def Log(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(*args, transform=axis.transform.log, **kwargs),
        )

    def Pow(self, *args, power: float, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(*args, transform=axis.transform.Pow(power), **kwargs),
        )

    def Func(
        self,
        *args,
        forward: Callable[[float], float],
        inverse: Callable[[float], float],
        **kwargs,
    ) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class,
            *self.axes,
            axis.Regular(
                *args, transform=axis.transform.Function(forward, inverse), **kwargs
            ),
        )

    def Bool(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class, *self.axes, axis.Boolean(*args, **kwargs)
        )

    def Var(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class, *self.axes, axis.Variable(*args, **kwargs)
        )

    def Int(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class, *self.axes, axis.Integer(*args, **kwargs)
        )

    def IntCat(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class, *self.axes, axis.IntCategory(*args, **kwargs)
        )

    def StrCat(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            self.hist_class, *self.axes, axis.StrCategory(*args, **kwargs)
        )


class ConstructProxy(QuickConstruct):
    __slots__ = ()

    def Double(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.Double())

    def Int64(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.Int64())

    def AtomicInt64(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.AtomicInt64())

    def Weight(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.Weight())

    def Mean(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.Mean())

    def WeightedMean(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.WeightedMean())

    def Unlimited(self) -> "BaseHist":
        return self.hist_class(*self.axes, storage=storage.Unlimited())


class MetaConstructor(type):
    @property
    def new(cls: "Type[BaseHist]") -> QuickConstruct:  # type: ignore
        return QuickConstruct(cls)
