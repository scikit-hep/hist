from typing import Type, TypeVar, Union

from .typing import Protocol


class SupportsStr(Protocol):
    def __str__(self) -> str:
        ...


class XML:
    def __init__(
        self, *contents: Union["XML", SupportsStr], **kargs: SupportsStr
    ) -> None:
        self.properties = kargs
        self.contents = contents

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def start(self) -> str:
        if self.properties:
            ending = " " + " ".join(
                '{}="{}"'.format(a.replace("_", "-"), b)
                for a, b in self.properties.items()
            )
        else:
            ending = ""
        return self.name + ending

    def __str__(self) -> str:
        if not self.contents:
            return f"<{self.start}/>"

        contents = "\n".join(str(s) for s in self.contents)
        return f"<{self.start}>\n{contents}\n</{self.name}>"


class html(XML):
    def _repr_xml_(self) -> str:
        return str(self)


class svg(XML):
    def __init__(self, *args: Union[XML, str], **kwargs: str) -> None:
        super().__init__(*args, xmlns="http://www.w3.org/2000/svg", **kwargs)

    def _repr_svg_(self) -> str:
        return str(self)


class circle(XML):
    pass


class line(XML):
    pass


class p(XML):
    pass


class div(XML):
    pass


T = TypeVar("T", bound="rect")


class rect(XML):
    @classmethod
    def pad(
        cls: Type[T],
        x: float,
        y: float,
        scale_x: float,
        scale_y: float,
        height: float,
        left_edge: float,
        right_edge: float,
        pad_x: float = 0,
        pad_y: float = 0,
        opacity: float = 1,
        stroke_width: float = 2,
    ) -> T:
        width = right_edge - left_edge
        top_y = y

        height *= 1 - 2 * pad_y
        top_y = top_y * (1 - 2 * pad_y) + pad_y
        width = width * (1 - 2 * pad_x)
        color = int(opacity * 255)

        return cls(
            x=x * scale_x,
            y=top_y * scale_y,
            width=width * scale_x,
            height=height * scale_y,
            style=f"fill:rgb({color}, {color}, {color});stroke:rgb(0,0,0);stroke-width:{stroke_width}",
        )


class ellipse(XML):
    pass


class polygon(XML):
    pass


class polyline(XML):
    pass


# path
class text(XML):
    pass


class g(XML):
    pass
