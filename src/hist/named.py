from .core import BaseHist
import numpy as np


class NamedHist(BaseHist):
    def __init__(self, *args, **kwargs):
        """
            Initialize NamedHist object. Axis params must contain the names.
        """
        super(BaseHist, self).__init__(*args, **kwargs)
        names = dict()
        for ax in self.axes:
            if not ax.name:
                raise Exception(
                    "Each axes in the NamedHist instance should have a name."
                )
            elif ax.name in names:
                raise Exception(
                    "NamedHist instance cannot contain axes with duplicated names."
                )
            else:
                names[ax.name] = True

    def fill(self, *args, **kwargs):
        """
            Insert data into the histogram using names and return a \
            NamedHist object.
        """

        # fill by default
        if len(args):
            super().fill(*args)
            return self

        # fill by name
        indices = []
        values = []
        for name, val in kwargs.items():
            for index, axis in enumerate(self.axes):
                if name == axis.name:
                    indices.append(index)
                    values.append(val)

        d = dict(zip(indices, values))
        d = sorted(d.items(), key=lambda item: item[0])
        nd = np.asarray(d, dtype=object)
        super().fill(*(nd.ravel()[1::2]))

        return self
