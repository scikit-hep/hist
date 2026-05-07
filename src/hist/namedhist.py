from __future__ import annotations

import typing
from typing import Any, Generic, TypeVar

import boost_histogram as bh

import hist

from . import interop
from .basehist import BaseHist, IndexingExpr

if typing.TYPE_CHECKING:
    from ._compat.typing import ArrayLike, Self

S = TypeVar("S", bound=bh.storage.Storage)


class NamedHist(BaseHist[S], Generic[S], family=hist):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize NamedHist object. Axis params must contain the names.
        """

        super().__init__(*args, **kwargs)
        if args and "" in self.axes.name:
            msg = f"Each axes in the {self.__class__.__name__} instance should have a name"
            raise RuntimeError(msg)

    def project(self, *args: int | str) -> Self:
        """
        Projection of axis idx.
        """

        if not args or all(isinstance(x, str) for x in args):
            return super().project(*args)

        msg = f"Only projections by names are supported for {self.__class__.__name__}"
        raise TypeError(msg)

    # pylint: disable-next=arguments-differ
    def fill(  # type: ignore[override]
        self,
        weight: ArrayLike | None = None,
        sample: ArrayLike | None = None,
        threads: int | None = None,
        **kwargs: ArrayLike,
    ) -> Self:
        """
            Insert data into the histogram using names and return a \
            NamedHist object. NamedHist could only be filled by names.
        """

        if kwargs and all(isinstance(k, str) for k in kwargs):
            return super().fill(weight=weight, sample=sample, threads=threads, **kwargs)

        msg = f"Only fill by names are supported for {self.__class__.__name__}"
        raise TypeError(msg)

    # pylint: disable-next=arguments-differ
    def fill_flattened(  # type: ignore[override]
        self: Self,
        obj: Any = None,
        *,
        weight: Any | None = None,
        sample: Any | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ) -> Self:
        axis_names = {ax.name for ax in self.axes}

        non_user_kwargs = {}
        if weight is not None:
            non_user_kwargs["weight"] = weight
        if sample is not None:
            non_user_kwargs["sample"] = sample

        # Single arguments are either arrays for single-dimensional histograms, or
        # structures for multi-dimensional hists that must first be unpacked
        if obj is not None:
            if kwargs:
                msg = (
                    "Only explicit keyword arguments, or a single structured object is supported by "
                    "`fill_flattened`, but not both."
                )
                raise TypeError(msg)

            destructured = interop.destructure(obj)
            # Try to unpack the array, if it's valid to do so, i.e. is the Awkward Array a record array?
            if destructured is None:
                msg = (
                    f"Only fill by names are supported for {self.__class__.__name__}. A single object was given to "
                    "`fill_flattened`, but it could not be destructed into name-array pairs."
                )
                raise TypeError(msg)

            # Result must be broadcast, so unpack and rebuild
            broadcast = interop.broadcast_and_flatten(
                (*destructured.values(), *non_user_kwargs.values())
            )
            # Partition into user and non-user args
            user_kwargs_broadcast = {
                k: v
                for k, v in zip(
                    destructured, broadcast[: len(destructured)], strict=True
                )
                if k in axis_names
            }
            non_user_kwargs_broadcast = dict(
                zip(non_user_kwargs, broadcast[len(destructured) :], strict=True)
            )
        # Multiple args: broadcast and flatten!
        else:
            inputs = (*kwargs.values(), *non_user_kwargs.values())
            broadcast = interop.broadcast_and_flatten(inputs)
            user_kwargs_broadcast = dict(
                zip(kwargs, broadcast[: len(kwargs)], strict=True)
            )
            non_user_kwargs_broadcast = dict(
                zip(non_user_kwargs, broadcast[len(kwargs) :], strict=True)
            )
        return self.fill(
            **user_kwargs_broadcast,
            threads=threads,
            **non_user_kwargs_broadcast,
        )

    if not typing.TYPE_CHECKING:

        def __getitem__(self, index):
            """
            Get histogram item.
            """

            if isinstance(index, dict) and any(isinstance(k, int) for k in index):
                msg = f"Only access by names are supported for {self.__class__.__name__} in dictionary"
                raise TypeError(msg)

            return super().__getitem__(index)

    def __setitem__(  # type: ignore[override]
        self,
        index: IndexingExpr,
        value: ArrayLike | bh.accumulators.Accumulator,
    ) -> None:
        """
        Set histogram item.
        """

        if isinstance(index, dict) and any(isinstance(k, int) for k in index):
            msg = f"Only access by names are supported for {self.__class__.__name__} in dictionary"
            raise TypeError(msg)

        return super().__setitem__(index, value)
