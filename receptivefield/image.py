import os
from typing import Union
import numpy
from PIL import Image
from receptivefield.types import ImageShape

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_default_image(
        shape: ImageShape,
        tile_factor: int = 0,
        shade: bool = True,
        as_image: bool = False
) -> Union[numpy.ndarray, Image.Image]:

    """
    Loads default image from resources and reshape it to size
    shape.
    :param shape: [width, height]
    :param tile_factor: tile image, if 0 the resulting image shape is
    [width, height], otherwise the output size is defined by number of
    tiles. tile_factor is a non-negative integer number.
    :param shade: if True and tile_factor > 0 it makes tiles gray scale
    :param as_image: if True, function returns PIL Image object, else
    numpy array.
    :return: numpy array of shape [width, height, 3] if as_image=False,
    wise PIL Image object
    """
    shape = ImageShape(*shape)
    tile_factor = int(tile_factor)

    img = Image.open(
        os.path.join(dir_path, 'resources/lena.jpg'), mode='r')

    img = img.resize((shape.w, shape.h), Image.ANTIALIAS)

    if tile_factor > 0:
        tf = 2 * tile_factor + 1
        new_img = Image.new('RGB', (shape.w * tf, shape.h * tf))
        new_shape = ImageShape(*new_img.size)
        gray_img = img.convert('LA').convert('RGB')

        for n, i in enumerate(range(0, new_shape.w, shape.w)):
            for m, j in enumerate(range(0, new_shape.h, shape.h)):
                distance = (abs(m - tile_factor) + abs(n - tile_factor)) / tf
                alpha = distance > 0 if shade else 0
                # place image at position (i, j)
                new_img.paste(
                    Image.blend(img, gray_img, alpha), (i, j)
                )
        img = new_img

    if as_image:
        return img
    return numpy.array(img)
