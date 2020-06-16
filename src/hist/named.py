from .core import BaseHist
import numpy as np


class NamedHist(BaseHist):
    def __init__(self, *args, **kwargs):
        """
            Initialize NamedHist object. Axis params must contain the names.
        """

        super().__init__(*args, **kwargs)
        if "" in self.names:
            raise Exception("Each axes in the NamedHist instance should have a name.")

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

        d = dict(zip(indices, values))
        l = sorted(d.items(), key=lambda item: item[0])
        nd = np.asarray(l, dtype=object)
        data = nd.ravel()[1::2]
        if len(data) != len(self.axes):
            raise Exception("The axis names could not be found when filling.")
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
