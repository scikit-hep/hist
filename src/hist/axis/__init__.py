# -*- coding: utf-8 -*-
from typing import Dict, List, Union, Any

import sys
import boost_histogram.axis as bha
import hist.utils
from hist.axestuple import NamedAxesTuple, ArrayTuple

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


class AxisProtocol(Protocol):
    metadata: Any
    name: str
    label: str


class AxesMixin:
    __slots__ = ()

    @property
    def name(self: AxisProtocol) -> str:
        """
        Get or set the name for the Regular axis
        """
        return self.metadata.get("name", "") if isinstance(self.metadata, dict) else ""

    @name.setter
    def name(self: AxisProtocol, value: str) -> None:
        self.metadata["name"] = value

    @property
    def label(self: AxisProtocol) -> str:
        """
        Get or set the label for the Regular axis
        """
        label = (
            self.metadata.get("label", "") if isinstance(self.metadata, dict) else ""
        )
        return label or self.name

    @label.setter
    def label(self: AxisProtocol, value: str) -> None:
        self.metadata["label"] = value


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class Regular(bha.Regular, AxesMixin):
    __slots__ = ()

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = None,
        label: str = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False,
        circular: bool = False,
        transform: bha.transform.Function = None
    ) -> None:
        metadata: Dict[str, Any] = {"name": name or "", "label": label or ""}
        super().__init__(
            bins,
            start,
            stop,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
            circular=circular,
            transform=transform,
        )


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class Boolean(bha.Boolean, AxesMixin):
    __slots__ = ()

    def __init__(self, *, name: str = None, label: str = None) -> None:
        metadata: Dict[str, Any] = {"name": name or "", "label": label or ""}
        super().__init__(metadata=metadata)


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class Variable(bha.Variable, AxesMixin):
    __slots__ = ()

    def __init__(
        self,
        edges: Union[range, List[float]],
        *,
        name: str = None,
        label: str = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False
    ) -> None:
        metadata: Dict[str, Any] = {"name": name or "", "label": label or ""}
        super().__init__(
            edges,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
        )


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class Integer(bha.Integer, AxesMixin):
    __slots__ = ()

    def __init__(
        self,
        start: int,
        stop: int,
        *,
        name: str = None,
        label: str = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False
    ) -> None:
        metadata: Dict[str, Any] = {"name": name or "", "label": label or ""}
        super().__init__(
            start,
            stop,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
        )


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class IntCategory(bha.IntCategory, AxesMixin):
    __slots__ = ()

    def __init__(
        self,
        categories: Union[range, List[int]] = None,
        *,
        name: str = None,
        label: str = None,
        growth: bool = False
    ) -> None:
        metadata: Dict[str, Any] = {"name": name or "", "label": label or ""}
        super().__init__(categories, metadata=metadata, growth=growth)


@hist.utils.set_family(hist.utils.HIST_FAMILY)
class StrCategory(bha.StrCategory, AxesMixin):
    __slots__ = ()

    def __init__(
        self,
        categories: Union[str, List[str]] = None,
        *,
        name: str = None,
        label: str = None,
        growth: bool = False
    ) -> None:
        metadata: Dict[str, Any] = {"name": name or "", "label": label or ""}
        super().__init__(categories, metadata=metadata, growth=growth)
