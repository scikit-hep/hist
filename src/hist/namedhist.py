from typing import Any, Optional, TypeVar, Union

import boost_histogram as bh

import hist

from .basehist import BaseHist, IndexingExpr
from .typing import ArrayLike

T = TypeVar("T", bound="NamedHist")


class NamedHist(BaseHist, family=hist):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize NamedHist object. Axis params must contain the names.
        """

        super().__init__(*args, **kwargs)
        if len(args) and "" in self.axes.name:
            raise RuntimeError(
                f"Each axes in the {self.__class__.__name__} instance should have a name"
            )

    # TODO: This can return a single value
    def project(
        self: T, *args: Union[int, str]
    ) -> Union[T, float, bh.accumulators.Accumulator]:
        """
        Projection of axis idx.
        """

        if len(args) == 0 or all(isinstance(x, str) for x in args):
            return super().project(*args)

        else:
            raise TypeError(
                f"Only projections by names are supported for {self.__class__.__name__}"
            )

    def fill(  # type: ignore
        self: T,
        weight: Optional[ArrayLike] = None,
        sample: Optional[ArrayLike] = None,
        threads: Optional[int] = None,
        **kwargs: ArrayLike,
    ) -> T:
        """
            Insert data into the histogram using names and return a \
            NamedHist object. NamedHist could only be filled by names.
        """

        if len(kwargs) and all(isinstance(k, str) for k in kwargs.keys()):
            return super().fill(weight=weight, sample=sample, threads=threads, **kwargs)

        else:
            raise TypeError(
                f"Only fill by names are supported for {self.__class__.__name__}"
            )

    def __getitem__(  # type: ignore
        self: T,
        index: IndexingExpr,
    ) -> Union[T, float, bh.accumulators.Accumulator]:
        """
        Get histogram item.
        """

        if isinstance(index, dict) and any(isinstance(k, int) for k in index.keys()):
            raise TypeError(
                f"Only access by names are supported for {self.__class__.__name__} in dictionay"
            )

        return super().__getitem__(index)

    def __setitem__(  # type: ignore
        self,
        index: IndexingExpr,
        value: Union[ArrayLike, bh.accumulators.Accumulator],
    ) -> None:
        """
        Set histogram item.
        """

        if isinstance(index, dict) and any(isinstance(k, int) for k in index.keys()):
            raise TypeError(
                f"Only access by names are supported for {self.__class__.__name__} in dictionay"
            )

        return super().__setitem__(index, value)
