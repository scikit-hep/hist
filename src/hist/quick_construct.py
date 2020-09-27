# -*- coding: utf-8 -*-
from typing import Callable, TYPE_CHECKING

from .axis import AxisProtocol
from . import storage, axis

if TYPE_CHECKING:
    from .hist import Hist


class QuickConstruct:
    """
    Create a quick construct instance. This is the "base" quick contructor; it will
    always require at least one axes to be added before allowing a storage or fill to be performed.
    """

    __slots__ = ("axes",)

    def __repr__(self):
        inside = ", ".join(repr(ax) for ax in self.axes)
        return f"{self.__class__.__name__}({inside})"

    def __init__(self, *axes: AxisProtocol):
        self.axes = axes

    def Reg(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(*self.axes, axis.Regular(*args, **kwargs))

    def Sqrt(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            *self.axes, axis.Regular(*args, transform=axis.transform.sqrt, **kwargs)
        )

    def Log(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
            *self.axes, axis.Regular(*args, transform=axis.transform.log, **kwargs)
        )

    def Pow(self, *args, power: float, **kwargs) -> "ConstructProxy":
        return ConstructProxy(
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
            *self.axes,
            axis.Regular(
                *args, transform=axis.transform.Function(forward, inverse), **kwargs
            ),
        )

    def Bool(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(*self.axes, axis.Boolean(*args, **kwargs))

    def Var(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(*self.axes, axis.Variable(*args, **kwargs))

    def Int(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(*self.axes, axis.Integer(*args, **kwargs))

    def IntCat(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(*self.axes, axis.IntCategory(*args, **kwargs))

    def StrCat(self, *args, **kwargs) -> "ConstructProxy":
        return ConstructProxy(*self.axes, axis.StrCategory(*args, **kwargs))


class ConstructProxy(QuickConstruct):
    __slots__ = ()

    def Double(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.Double())

    def Int64(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.Int64())

    def AtomicInt64(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.AtomicInt64())

    def Weight(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.Weight())

    def Mean(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.Mean())

    def WeightedMean(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.WeightedMean())

    def Unlimited(self) -> "Hist":
        from hist.hist import Hist

        return Hist(*self.axes, storage=storage.Unlimited())
