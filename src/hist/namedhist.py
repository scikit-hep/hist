# -*- coding: utf-8 -*-
from .basehist import BaseHist
import numpy as np
from typing import Union


class NamedHist(BaseHist):
    def __init__(self, *args, **kwargs):
        """
            Initialize NamedHist object. Axis params must contain the names.
        """

        super().__init__(*args, **kwargs)
        if "" in self.axes.name:
            raise Exception(
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

    def fill(self, *args, **kwargs):
        """
            Insert data into the histogram using names and return a \
            NamedHist object. NamedHist could only be filled by names.
        """

        indices = []
        values = []
        for name, val in kwargs.items():
            for index, axis in enumerate(self.axes):
                if name == axis.name:
                    indices.append(index)
                    values.append(val)
                    break
            else:
                raise ValueError("The axis names could not be found")

        d = dict(zip(indices, values))
        lst = sorted(d.items(), key=lambda item: item[0])
        nd = np.asarray(lst, dtype=object)
        data = nd.ravel()[1::2]
        super().fill(*data)

        return self

    def __getitem__(self, index):
        """
            Get histogram item.
        """

        if isinstance(index, dict):
            k = list(index.keys())
            if isinstance(k[0], str):
                for key in k:
                    for idx, axis in enumerate(self.axes):
                        if key == axis.name:
                            index[idx] = index.pop(key)
                            break

        return super().__getitem__(index)

    def __setitem__(self, index, value):
        """
            Set histogram item.
        """

        if isinstance(index, dict):
            k = list(index.keys())
            if isinstance(k[0], str):
                for key in k:
                    for idx, axis in enumerate(self.axes):
                        if key == axis.name:
                            index[idx] = index.pop(key)
                            break

        return super().__setitem__(index, value)
