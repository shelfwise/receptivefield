from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn

from receptivefield.base import ReceptiveField
from receptivefield.logging import get_logger
from receptivefield.types import ImageShape, GridPoint, GridShape, \
    ReceptiveFieldDescription

_logger = get_logger()


def _define_receptive_field_func(
        model_fn: nn.Module,
        input_shape: GridShape
):
    shape = [input_shape.n, input_shape.c, input_shape.w, input_shape.h]
    x = torch.zeros(*shape)
    model = model_fn()
    _ = model(x)

    output_shape = model.feature_map.size()
    output_shape = GridShape(
        n=output_shape[0],
        c=output_shape[1],
        w=output_shape[2],
        h=output_shape[3])

    _logger.info(f"Feature map shape: {output_shape}")
    _logger.info(f"Input shape      : {input_shape}")

    def gradient_function(receptive_field_mask):
        model.zero_grad()
        input_image = torch.zeros(*shape) / (input_shape.w / input_shape.h / input_shape.c)
        input_tensor = torch.autograd.Variable(
            input_image, requires_grad=True)
        _ = model(input_tensor)

        fm = model.feature_map
        fm = torch.mean(fm, 1, keepdim=True)
        fake_loss = fm * receptive_field_mask
        fake_loss = torch.mean(fake_loss)

        fake_loss.backward()
        return input_tensor.grad

    return gradient_function, input_shape, output_shape


class PytorchReceptiveField(ReceptiveField):
    def __init__(
            self,
            model_func: Callable[[], nn.Module],
    ):
        """
        :param model_func: model function which returns new instance of
            nn.Module.
        """
        super().__init__(model_func)

    def _prepare_gradient_func(
            self,
            input_shape: ImageShape,
            input_layer: str = None,
            output_layer: str = None
    ) -> Tuple[Callable, GridShape, GridShape]:
        input_shape = ImageShape(*input_shape)

        input_shape = GridShape(
            n=1,
            c=input_shape.c,
            w=input_shape.w,
            h=input_shape.h)
        gradient_function, input_shape, output_shape = \
            _define_receptive_field_func(self.model_func, input_shape)

        return gradient_function, input_shape, output_shape

    def _get_gradient_from_grid_point(
            self,
            point: GridPoint,
            intensity: float = 1.0
    ) -> np.ndarray:

        os = self.output_shape
        output_feature_map = torch.zeros([1, 1, os.w, os.h])
        output_feature_map[:, :, point.x, point.y] = intensity
        grad = self.gradient_function(output_feature_map).detach().numpy()
        grad = np.abs(np.transpose(grad, [0, 2, 3, 1]))
        grad = grad / grad.max()
        return grad

    def compute(self, input_shape: ImageShape) -> ReceptiveFieldDescription:
        """
        Compute ReceptiveFieldDescription of given model for image of
        shape input_shape [W, H, C]. If receptive field of the network
        is bigger thant input_shape this method will raise exception.
        In order to solve this problem try to increase input_shape.

        :param input_shape: shape of the input image e.g. (224, 224, 3)

        :return: estimated ReceptiveFieldDescription
        """
        return super().compute(input_shape, 'not used', 'not used')
