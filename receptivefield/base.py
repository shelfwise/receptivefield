from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable, Any, Dict

import numpy as np

from receptivefield.common import estimate_rf_from_gradients
from receptivefield.logging import get_logger
from receptivefield.plotting import plot_receptive_grid, plot_gradient_field
from receptivefield.types import ImageShape, GridPoint, GridShape, \
    ReceptiveFieldDescription, Size

_logger = get_logger()


class ReceptiveField(metaclass=ABCMeta):
    def __init__(
            self,
            model_func: Callable[[ImageShape], Any]):
        self.model_func: Callable[[ImageShape], Any] = model_func
        self.gradient_function: Callable = None
        self.input_shape: GridShape = None
        self.output_shape: GridShape = None
        self.rf_params: ReceptiveFieldDescription = None
        self.built = False

    def build_gradient_func(
            self,
            input_shape: ImageShape,
            input_layer: str,
            output_layer: str
    ) -> None:
        """
        Computes gradient function and additional parameters
        :param input_shape: shape of the input image.
        :param input_layer: name of the input layer
        :param output_layer: name of the target layer
        """
        gradient_function, input_shape, output_shape = \
            self._prepare_gradient_func(
                input_shape=input_shape,
                input_layer=input_layer,
                output_layer=output_layer
            )
        self.built = True
        self.gradient_function = gradient_function
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def _prepare_gradient_func(
            self,
            input_shape: ImageShape,
            input_layer: str,
            output_layer: str
    ) -> Tuple[Callable, GridShape, GridShape]:
        """
        Computes gradient function and additional parameters. Note
        that the receptive field parameters like stride or size, do not
        depend on input image shape. However, if the RF of original network
        is bigger than input_shape this method will fail. Hence it is
        recommended to increase the input shape.
        :param input_shape: shape of the input image. Used in @model_func.
        :param input_layer: name of the input layer
        :param output_layer: name of the target layer
        """
        pass

    @abstractmethod
    def _get_gradient_from_grid_point(
            self,
            point: GridPoint,
            intensity: float = 1.0
    ) -> np.ndarray:
        """
        Computes gradient at  input_layer (image_layer) generated by
        point-like perturbation at  output grid location given by
        @point coordinates.

        :param point: source coordinate of the backpropagated gradient
        :param intensity: scale of the gradient, default = 1
        :return:
        """
        pass

    def _get_gradient_activation_at_map_center(
            self,
            center_offset: GridPoint,
            intensity: float = 1
    ):
        _logger.debug(
            f"Computing receptive field at center "
            f"({self.output_shape.w//2}, {self.output_shape.h//2}) "
            f"with offset {center_offset}")

        # compute grid center
        w = self.output_shape.w
        h = self.output_shape.h
        cx = w // 2 - 1 \
            if w % 2 == 0 else w // 2
        cy = h // 2 - 1 \
            if h % 2 == 0 else h // 2

        cx += center_offset.x
        cy += center_offset.y

        return self._get_gradient_from_grid_point(
            point=GridPoint(x=cx, y=cy),
            intensity=intensity
        )

    def _check(self):
        if not self.built:
            raise Exception("Receptive field not computed. "
                            "Run compute function.")

    def compute(
            self,
            input_shape: ImageShape,
            input_layer: str,
            output_layer: str
    ) -> ReceptiveFieldDescription:
        """
        Compute ReceptiveFieldDescription of given model for image of
        shape input_shape [W, H, C]. If receptive field of the network
        is bigger thant input_shape this method will raise exception.
        In order to solve with problem try to increase input_shape.

        :param input_shape: shape of the input image e.g. (224, 224, 3)
        :param input_layer: name of the input layer
        :param output_layer: name of the target layer
        :return: estimated ReceptiveFieldDescription
        """
        # define gradient function
        self.build_gradient_func(
            input_shape=input_shape,
            input_layer=input_layer,
            output_layer=output_layer
        )

        # receptive field at map center
        rf_grad00 = self._get_gradient_activation_at_map_center(center_offset=GridPoint(0, 0))
        rf_at00 = estimate_rf_from_gradients(rf_grad00)

        # receptive field at map center with offset (1, 1)
        rf_grad11 = self._get_gradient_activation_at_map_center(center_offset=GridPoint(1, 1))
        rf_at11 = estimate_rf_from_gradients(rf_grad11)

        # receptive field at feature map grid start x=0, y=0
        rf_grad_point00 = self._get_gradient_from_grid_point(point=GridPoint(0, 0))
        rf_at_point00 = estimate_rf_from_gradients(rf_grad_point00)

        # compute position of the first anchor, center point of rect
        x0 = rf_at_point00.w - rf_at00.w / 2
        y0 = rf_at_point00.h - rf_at00.h / 2

        # compute feature map/input image offsets
        dx = rf_at11.x - rf_at00.x
        dy = rf_at11.y - rf_at00.y

        # compute receptive field size
        size = Size(rf_at00.w, rf_at00.h)

        rf_params = ReceptiveFieldDescription(
            offset=(x0, y0),
            stride=(dx, dy),
            size=size
        )
        self.rf_params = rf_params
        return rf_params

    def plot_gradient_at(
            self,
            point: GridPoint,
            image: np.ndarray = None, **plot_params):
        receptive_field_grad = self._get_gradient_from_grid_point(
            self.gradient_function, self.input_shape,
            self.output_shape, point=GridPoint(*point)
        )

        plot_gradient_field(
            receptive_field_grad=receptive_field_grad,
            image=image,
            **plot_params
        )

    def plot_rf_grid(
            self,
            custom_image: np.ndarray = None,
            plot_naive_rf: bool = False,
            **plot_params
    ) -> None:
        plot_receptive_grid(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            rf_params=self.rf_params,
            custom_image=custom_image,
            plot_naive_rf=plot_naive_rf,
            **plot_params
        )
