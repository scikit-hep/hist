# -*- coding: utf-8 -*-
from typing import Dict, List, Union

import re
import boost_histogram.axis as bha


class Regular(bha.Regular):
    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = None,
        title: str = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False,
        circular: bool = False,
        transform: bha.transform.Function = None
    ) -> None:
        metadata: Dict = dict()
        if not name:
            metadata["name"] = ""
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier")
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

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self) -> str:
        """
        Get or set the name for the Regular axis
        """
        return self.metadata["name"] if self.metadata else ""

    @name.setter
    def name(self, value: str) -> None:
        self.metadata["name"] = value

    @property
    def title(self) -> str:
        """
        Get or set the title for the Regular axis
        """
        return self.metadata["title"] if self.metadata else ""

    @title.setter
    def title(self, value: str) -> None:
        self.metadata["title"] = value


class Boolean(bha.Regular):  # ToDo: should be derived from bha.Boolean
    def __init__(self, *, name: str = None, title: str = None) -> None:
        metadata: Dict = dict()
        if not name:
            metadata["name"] = ""
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier")
        metadata["title"] = title
        super(bha.Regular, self).__init__(2, 0, 2, metadata=metadata)

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self) -> str:
        """
        Get or set the name for the Bool axis
        """
        return self.metadata["name"] if self.metadata else ""

    @name.setter
    def name(self, value: str) -> None:
        self.metadata["name"] = value

    @property
    def title(self) -> str:
        """
        Get or set the title for the Bool axis
        """
        return self.metadata["title"] if self.metadata else ""

    @title.setter
    def title(self, value: str) -> None:
        self.metadata["title"] = value


class Variable(bha.Variable):
    def __init__(
        self,
        edges: Union[range, List[float]],
        *,
        name: str = None,
        title: str = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False
    ) -> None:
        metadata: Dict = dict()
        if not name:
            metadata["name"] = ""
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier")
        metadata["title"] = title
        super(bha.Variable, self).__init__(
            edges,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
        )

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self) -> str:
        """
        Get or set the name for the Variable axis
        """
        return self.metadata["name"] if self.metadata else ""

    @name.setter
    def name(self, value: str) -> None:
        self.metadata["name"] = value

    @property
    def title(self) -> str:
        """
        Get or set the title for the Variable axis
        """
        return self.metadata["title"] if self.metadata else ""

    @title.setter
    def title(self, value: str) -> None:
        self.metadata["title"] = value


class Integer(bha.Integer):
    def __init__(
        self,
        start: int,
        stop: int,
        *,
        name: str = None,
        title: str = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False
    ) -> None:
        metadata: Dict = dict()
        if not name:
            metadata["name"] = ""
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier")
        metadata["title"] = title
        super(bha.Integer, self).__init__(
            start,
            stop,
            metadata=metadata,
            underflow=underflow,
            overflow=overflow,
            growth=growth,
        )

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self) -> str:
        """
        Get or set the name for the Integer axis
        """
        return self.metadata["name"] if self.metadata else ""

    @name.setter
    def name(self, value: str) -> None:
        self.metadata["name"] = value

    @property
    def title(self) -> str:
        """
        Get or set the title for the Integer axis
        """
        return self.metadata["title"] if self.metadata else ""

    @title.setter
    def title(self, value: str) -> None:
        self.metadata["title"] = value


class IntCategory(bha.IntCategory):
    def __init__(
        self,
        categories: Union[range, List[int]] = None,
        *,
        name: str = None,
        title: str = None,
        growth: bool = False
    ) -> None:
        metadata: Dict = dict()
        if not name:
            metadata["name"] = ""
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier")
        metadata["title"] = title
        super(bha.IntCategory, self).__init__(
            categories, metadata=metadata, growth=growth
        )

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self) -> str:
        """
        Get or set the name for the IntCategory axis
        """
        return self.metadata["name"] if self.metadata else ""

    @name.setter
    def name(self, value: str) -> None:
        self.metadata["name"] = value

    @property
    def title(self) -> str:
        """
        Get or set the title for the IntCategory axis
        """
        return self.metadata["title"] if self.metadata else ""

    @title.setter
    def title(self, value: str) -> None:
        self.metadata["title"] = value


class StrCategory(bha.StrCategory):
    def __init__(
        self,
        categories: Union[str, List[str]] = None,
        *,
        name: str = None,
        title: str = None,
        growth: bool = False
    ) -> None:
        metadata: Dict = dict()
        if not name:
            metadata["name"] = ""
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            metadata["name"] = name
        else:
            raise Exception("Name should be a valid Python identifier")
        metadata["title"] = title
        super(bha.StrCategory, self).__init__(
            categories, metadata=metadata, growth=growth
        )

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    @property
    def name(self) -> str:
        """
        Get or set the name for the StrCategory axis
        """
        return self.metadata["name"] if self.metadata else ""

    @name.setter
    def name(self, value: str) -> None:
        self.metadata["name"] = value

    @property
    def title(self) -> str:
        """
        Get or set the title for the StrCategory axis
        """
        return self.metadata["title"] if self.metadata else ""

    @title.setter
    def title(self, value: str) -> None:
        self.metadata["title"] = value
