# -*- coding: utf-8 -*-
class XML:
    def __init__(self, *contents, **kargs):
        self.properties = kargs
        self.contents = contents

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def start(self):
        if self.properties:
            ending = " " + " ".join(
                '{}="{}"'.format(a.replace("_", "-"), b)
                for a, b in self.properties.items()
            )
        else:
            ending = ""
        return self.name + ending

    def __str__(self):
        if self.contents:
            contents = "\n".join(str(s) for s in self.contents)
            return f"<{self.start}>\n{contents}\n</{self.name}>"
        else:
            return f"<{self.start}/>"


class html(XML):
    def _repr_xml_(self):
        return str(self)


class svg(XML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, xmlns="http://www.w3.org/2000/svg", **kwargs)

    def _repr_svg_(self):
        return str(self)


class circle(XML):
    pass


class line(XML):
    pass


class p(XML):
    pass


class div(XML):
    pass


class rect(XML):
    @classmethod
    def pad(
        cls,
        x,
        y,
        scale_x,
        scale_y,
        height,
        left_edge,
        right_edge,
        pad_x=0,
        pad_y=0,
        opacity=1,
        stroke_width=2,
    ):
        width = right_edge - left_edge
        top_y = y

        height = height * (1 - 2 * pad_y)
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
