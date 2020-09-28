# -*- coding: utf-8 -*-
from .basehist import BaseHist
from typing import Union, Optional


class NamedHist(BaseHist):
    def __init__(self, *args, **kwargs):
        """
        Initialize NamedHist object. Axis params must contain the names.
        """

        super().__init__(*args, **kwargs)
        if len(args):
            if "" in self.axes.name:
                raise RuntimeError(
                    f"Each axes in the {self.__class__.__name__} instance should have a name"
                )

    def project(self, *args: Union[int, str]):
        """
        Projection of axis idx.
        """

        if len(args) == 0 or all(isinstance(x, str) for x in args):
            return super().project(*args)

        else:
            raise TypeError(
                f"Only projections by names are supported for {self.__class__.__name__}"
            )

    def fill(
        self, *args, weight=None, sample=None, threads: Optional[int] = None, **kwargs
    ):
        """
            Insert data into the histogram using names and return a \
            NamedHist object. NamedHist could only be filled by names.
        """

        if len(kwargs) and all(isinstance(k, str) for k in kwargs.keys()):
            return super().fill(
                *args, weight=weight, sample=sample, threads=threads, **kwargs
            )

        else:
            raise TypeError(
                f"Only fill by names are supported for {self.__class__.__name__}"
            )

    def __getitem__(self, index):
        """
        Get histogram item.
        """

        if isinstance(index, dict):
            if any(isinstance(k, int) for k in index.keys()):
                raise TypeError(
                    f"Only access by names are supported for {self.__class__.__name__} in dictionay"
                )

        return super().__getitem__(index)

    def __setitem__(self, index, value):
        """
        Set histogram item.
        """

        if isinstance(index, dict):
            if any(isinstance(k, int) for k in index.keys()):
                raise TypeError(
                    f"Only access by names are supported for {self.__class__.__name__} in dictionay"
                )

        return super().__setitem__(index, value)
