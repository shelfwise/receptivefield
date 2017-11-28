from typing import Tuple, Callable, Any

import keras.backend as K
import numpy as np
from keras.engine import Layer
from keras.layers import Conv2D, MaxPool2D, Activation
from keras.layers import Input, Lambda
from keras.models import Model

from receptivefield.base import ReceptiveField
from receptivefield.common import scaled_constant
from receptivefield.logging import get_logger
from receptivefield.types import ImageShape, GridPoint, GridShape

_logger = get_logger()


def _check_activation(layer: Layer):
    if issubclass(type(layer), Layer):
        layer_act = layer.activation.__name__
    else:
        layer_act = None

    if layer_act != 'linear':
        _logger.warning(f"Layer {layer.name} activation should be linear "
                        f"but is: {layer_act}")


def setup_model_weights(model: Model) -> None:
    """
    Set all weights to be a constant values. Biases are set to zero.
    Only Conv2D are supported.
    :param model: Keras model
    """
    for layer in model.layers:
        # check layer type
        if type(layer) == MaxPool2D:
            _logger.warning(f"MaxPool2D detected: {layer.name}. Replace it with"
                            f" AvgPool2D in order to obtain better receptive "
                            f"field mapping estimation")
        # set weights
        if type(layer) == Conv2D:
            _logger.debug(f"Setting weights for layer {layer.name}")
            weights = layer.get_weights()
            w_kernel = scaled_constant(1, weights[0].shape)
            w = [w_kernel]
            if len(weights) > 1:
                w_bias = np.zeros_like(weights[1])
                w.append(w_bias)
            layer.set_weights(w)
            _check_activation(layer)
        elif type(layer) == Activation:
            _check_activation(layer)
        else:
            _logger.warning(f"Layer type {type(layer)} is not supported")


def _define_receptive_field_func(
        model: Model,
        input_layer: str,
        feature_map_layer: str
):
    output_shape = model.get_layer(feature_map_layer).output_shape
    input_shape = model.get_layer(input_layer).output_shape

    _logger.info(f"Feature map shape: {output_shape}")
    _logger.info(f"Input shape      : {input_shape}")
    output_shape = [*output_shape[:-1], 1]
    receptive_field_mask = Input(shape=output_shape[1:])
    x = model.get_layer(feature_map_layer).output
    x = Lambda(lambda x: K.mean(x, -1, keepdims=True))(x)

    fake_loss = x * receptive_field_mask
    fake_loss = K.mean(fake_loss)
    grads = K.gradients(fake_loss, model.input)
    gradient_function = K.function(
        inputs=[receptive_field_mask, model.input, K.learning_phase()],
        outputs=grads)

    return gradient_function, input_shape, output_shape


class KerasReceptiveField(ReceptiveField):
    def __init__(
            self,
            model_func: Callable[[ImageShape], Any],
            init_weights: bool = False
    ):
        """
        :param model_func: model creation function
        :param init_weights: if True all conv2d weights are overwritten
        by constant value.
        """
        super().__init__(model_func)
        self.init_weights = init_weights

    def _prepare_gradient_func(
            self,
            input_shape: ImageShape,
            input_layer: str,
            output_layer: str
    ) -> Tuple[Callable, GridShape, GridShape]:
        model = self.model_func(ImageShape(*input_shape))
        if self.init_weights:
            setup_model_weights(model)

        gradient_function, input_shape, output_shape = \
            _define_receptive_field_func(model, input_layer, output_layer)

        return gradient_function, \
               GridShape(*input_shape), \
               GridShape(*output_shape)

    def _get_gradient_from_grid_point(
            self,
            point: GridPoint,
            intensity: float = 1.0
    ) -> np.ndarray:
        output_shape = self.output_shape.replace(n=1)
        input_shape = self.input_shape.replace(n=1)

        output_feature_map = np.zeros(shape=output_shape)
        output_feature_map[:, point.x, point.y, 0] = intensity

        receptive_field_grad = self.gradient_function([
            output_feature_map, np.zeros(shape=input_shape), 0])[0]

        return receptive_field_grad
