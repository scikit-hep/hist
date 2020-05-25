from .core import BaseHist
import numpy as np


class NamedHist(BaseHist):
    
    # ToDo: judge whether have no-named axes when initialization
    
    def fill(self, *args, **kwargs):
        """
            Insert data into the histogram using names and return a \
            NamedHist object. Params must contain the names of the axes.
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
