import re
import boost_histogram.axis as bha


class Regular(bha.Regular):
    def __init__(
        self,
        bins,
        start,
        stop,
        *,
        name=None,
        title=None,
        underflow=True,
        overflow=True,
        growth=False,
        circular=False,
        transform=None
    ):
        metadata = dict()
        if not name:
            metadata["name"] = None
        elif re.match(r"^[0-9a-zA-Z][0-9a-zA-Z_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier.")
        metadata["title"] = title
        super(bha.Regular, self).__init__(
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

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self):
        """
        Get or set the name for the Regular axis
        """
        return self.metadata["name"]

    @name.setter
    def name(self, value):
        self.metadata["name"] = value

    @property
    def title(self):
        """
        Get or set the title for the Regular axis
        """
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value


class Bool(bha.Regular):
    def __init__(
        self,
        *,
        name=None,
        title=None,
        underflow=False,
        overflow=False,
        growth=False,
        circular=False,
        transform=None
    ):
        metadata = dict()
        if not name:
            metadata["name"] = None
        elif re.match(r"^[0-9a-zA-Z][0-9a-zA-Z_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier.")
        metadata["title"] = title
        super(bha.Regular, self).__init__(
            2,
            0,
            2,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
            circular=circular,
            transform=transform,
        )

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self):
        """
        Get or set the name for the Bool axis
        """
        return self.metadata["name"]

    @name.setter
    def name(self, value):
        self.metadata["name"] = value

    @property
    def title(self):
        """
        Get or set the title for the Bool axis
        """
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value


class Variable(bha.Variable):
    def __init__(
        self,
        edges,
        *,
        name=None,
        title=None,
        underflow=True,
        overflow=True,
        growth=False
    ):
        metadata = dict()
        if not name:
            metadata["name"] = None
        elif re.match(r"^[0-9a-zA-Z][0-9a-zA-Z_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier.")
        metadata["title"] = title
        super(bha.Variable, self).__init__(
            edges,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
        )

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self):
        """
        Get or set the name for the Variable axis
        """
        return self.metadata["name"]

    @name.setter
    def name(self, value):
        self.metadata["name"] = value

    @property
    def title(self):
        """
        Get or set the title for the Variable axis
        """
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value


class Integer(bha.Integer):
    def __init__(
        self,
        start,
        stop,
        *,
        name=None,
        title=None,
        underflow=True,
        overflow=True,
        growth=False
    ):
        metadata = dict()
        if not name:
            metadata["name"] = None
        elif re.match(r"^[0-9a-zA-Z][0-9a-zA-Z_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier.")
        metadata["title"] = title
        super(bha.Integer, self).__init__(
            start,
            stop,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
        )

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self):
        """
        Get or set the name for the Integer axis
        """
        return self.metadata["name"]

    @name.setter
    def name(self, value):
        self.metadata["name"] = value

    @property
    def title(self):
        """
        Get or set the title for the Integer axis
        """
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value


class IntCategory(bha.IntCategory):
    def __init__(self, categories=None, *, name=None, title=None, growth=False):
        metadata = dict()
        if not name:
            metadata["name"] = None
        elif re.match(r"^[0-9a-zA-Z][0-9a-zA-Z_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier.")
        metadata["title"] = title
        super(bha.IntCategory, self).__init__(
            categories, metadata=metadata, growth=growth
        )

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self):
        """
        Get or set the name for the IntCategory axis
        """
        return self.metadata["name"]

    @name.setter
    def name(self, value):
        self.metadata["name"] = value

    @property
    def title(self):
        """
        Get or set the title for the IntCategory axis
        """
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value


class StrCategory(bha.StrCategory):
    def __init__(self, categories=None, *, name=None, title=None, growth=False):
        metadata = dict()
        if not name:
            metadata["name"] = None
        elif re.match(r"^[0-9a-zA-Z][0-9a-zA-Z_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier.")
        metadata["title"] = title
        super(bha.StrCategory, self).__init__(
            categories, metadata=metadata, growth=growth
        )

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self):
        """
        Get or set the name for the StrCategory axis
        """
        return self.metadata["name"]

    @name.setter
    def name(self, value):
        self.metadata["name"] = value

    @property
    def title(self):
        """
        Get or set the title for the StrCategory axis
        """
        return self.metadata["title"]

    @title.setter
    def title(self, value):
        self.metadata["title"] = value
