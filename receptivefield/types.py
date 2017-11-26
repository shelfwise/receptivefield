from typing import NamedTuple


class Size(NamedTuple):
    w: int
    h: int


class ImageShape(NamedTuple):
    w: int
    h: int
    c: int = 3


class GridPoint(NamedTuple):
    x: int
    y: int


class GridShape(NamedTuple):
    n: int
    w: int
    h: int
    c: int


class ReceptiveFieldRect(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class ReceptiveFieldDescription(NamedTuple):
    offset: GridPoint
    stride: GridPoint
    size: Size

    def get_offset(self) -> GridPoint:
        return GridPoint(*self.offset)

    def get_size(self) -> Size:
        return Size(*self.size)

    def get_stride(self) -> GridPoint:
        return GridPoint(*self.stride)


def to_rf_rect(point: GridPoint, size: Size) -> ReceptiveFieldRect:
    return ReceptiveFieldRect(x=point.x, y=point.y, w=size.w, h=size.h)