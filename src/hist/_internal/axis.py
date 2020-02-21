import boost_histogram.axis as bha


class Regular(bha.Regular):
    def __init__(
        self,
        bins,
        start,
        stop,
        *,
        title=None,
        underflow=True,
        overflow=True,
        growth=False,
        circular=False,
        transform=None
    ):
        metadata = dict(title=title)
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

    @property
    def title(self):
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value


class Variable(bha.Variable):
    pass


class Integer(bha.Integer):
    pass


class IntCategory(bha.IntCategory):
    pass


class StrCategory(bha.StrCategory):
    pass
