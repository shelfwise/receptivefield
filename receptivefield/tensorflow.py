from typing import Tuple, Callable, Any, List

import tensorflow as tf
import tensorflow.contrib.keras as keras

from receptivefield.keras import KerasReceptiveField
from receptivefield.logging import get_logger
from receptivefield.types import ImageShape, GridShape

_logger = get_logger()


def _get_tensor_shape(tensor: tf.Tensor) -> List[int]:
    """
    Parse TensorShape to python tuple:
    Example: TensorShape([
        Dimension(1), Dimension(8), Dimension(8), Dimension(512)
    ]) will return [1, 8, 8, 32]
    :param tensor: tensorflow Tensor
    :return: integer type tuple
    """
    return list(map(int, tensor.get_shape()))


class TFReceptiveField(KerasReceptiveField):
    def __init__(
            self,
            model_func: Callable[[ImageShape], Any]
    ):
        """
        :param model_func: model creation function. Function which
        returns tensorflow graph.
        """
        super().__init__(model_func)

    def _prepare_gradient_func(
            self,
            input_shape: ImageShape,
            input_layer: str,
            output_layers: List[str]
    ) -> Tuple[Callable, GridShape, List[GridShape]]:

        # this function will create default graph
        _ = self._model_func(ImageShape(*input_shape))

        default_graph = tf.get_default_graph()

        # get graph tensors by names
        input_tensor = default_graph \
            .get_operation_by_name(input_layer).outputs[0]

        grads = []
        receptive_field_masks = []
        output_shapes = []

        for output_layer in output_layers:
            output_tensor = default_graph \
                .get_operation_by_name(output_layer).outputs[0]

            # shapes
            output_shape = _get_tensor_shape(output_tensor)
            input_shape = _get_tensor_shape(input_tensor)
            output_shape = (1, output_shape[1], output_shape[2], 1)
            output_shapes.append(output_shape)

            # define loss function
            receptive_field_mask = tf.placeholder(
                tf.float32, shape=output_shape, name='grid'
            )

            x = tf.reduce_mean(output_tensor, -1, keep_dims=True)
            fake_loss = x * receptive_field_mask
            fake_loss = tf.reduce_mean(fake_loss)
            grad = tf.gradients(fake_loss, input_tensor)

            grads.append(grad[0])
            receptive_field_masks.append(receptive_field_mask)

        _logger.info(f"Feature maps shape: {output_shapes}")
        _logger.info(f"Input shape       : {input_shape}")

        # here we use Keras API to define gradient function which is simpler
        # than native tf
        gradient_function = keras.backend.function(
            inputs=[*receptive_field_masks, input_tensor, keras.backend.learning_phase()],
            outputs=grads
        )

        return gradient_function, GridShape(*input_shape), \
            [GridShape(*output_shape) for output_shape in output_shapes]