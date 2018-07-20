from typing import Tuple, Callable, Any, List

import numpy as np
import tensorflow as tf

from receptivefield.base import ReceptiveField
from receptivefield.logging import get_logger
from receptivefield.types import ImageShape, GridShape, GridPoint, \
    FeatureMapDescription

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


def _define_fm_gradient(
    input_tensor: tf.Tensor, output_tensor: tf.Tensor, receptive_field_mask: tf.Tensor
) -> tf.Tensor:
    """
    Define gradient of feature map w.r.t. the input image
    :param input_tensor: an input image tensor [w, h, 3]
    :param output_tensor: an feature map tensor [fm_w, fm_h, num_channels]
    :param receptive_field_mask: a backpropagation mask [fm_wm, fm_h, 1]
    :return: a gradient tensor [w, h, 3]
    """
    x = tf.reduce_mean(output_tensor, -1, keep_dims=True)
    fake_loss = x * receptive_field_mask
    fake_loss = tf.reduce_mean(fake_loss)
    # define gradient w.r.t. image
    return tf.gradients(fake_loss, input_tensor)[0]


class TFReceptiveField(ReceptiveField):
    def __init__(self, model_func: Callable[[ImageShape], Any]):
        """
        :param model_func: model creation function. Function which
        returns tensorflow graph.
        """
        self._session = None
        super().__init__(model_func)

    def _prepare_gradient_func(
        self, input_shape: ImageShape, input_layer: str, output_layers: List[str]
    ) -> Tuple[Callable, GridShape, List[GridShape]]:

        if self._session is not None:
            tf.reset_default_graph()
            self._session.close()

        with tf.Graph().as_default() as graph:
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):

                # this function will create default graph
                _ = self._model_func(ImageShape(*input_shape))

                # default_graph = tf.get_default_graph()
                # get graph tensors by names
                input_tensor = graph.get_operation_by_name(input_layer).outputs[0]
                input_shape = _get_tensor_shape(input_tensor)

                grads = []
                receptive_field_masks = []
                output_shapes = []

                for output_layer in output_layers:
                    output_tensor = graph.get_operation_by_name(output_layer).outputs[0]

                    # shapes
                    output_shape = _get_tensor_shape(output_tensor)
                    output_shape = (1, output_shape[1], output_shape[2], 1)
                    output_shapes.append(output_shape)

                    # define loss function
                    receptive_field_mask = tf.placeholder(
                        tf.float32, shape=output_shape, name="grid"
                    )
                    grad = _define_fm_gradient(
                        input_tensor, output_tensor, receptive_field_mask
                    )
                    grads.append(grad)
                    receptive_field_masks.append(receptive_field_mask)

                _logger.info(f"Feature maps shape: {output_shapes}")
                _logger.info(f"Input shape       : {input_shape}")

            self._session = tf.Session(graph=graph)
            self._session.run(tf.global_variables_initializer())

            def gradient_fn(fm_masks, input_image):
                fetch_dict = {
                    mask_t: mask_np
                    for mask_t, mask_np in zip(receptive_field_masks, fm_masks)
                }
                fetch_dict[input_tensor] = input_image
                return self._session.run(grads, feed_dict=fetch_dict)

        return (
            gradient_fn,
            GridShape(*input_shape),
            [GridShape(*output_shape) for output_shape in output_shapes],
        )

    def _get_gradient_from_grid_points(
        self, points: List[GridPoint], intensity: float = 1.0
    ) -> List[np.ndarray]:

        input_shape = self._input_shape.replace(n=1)
        output_feature_maps = []
        for fm in range(self.num_feature_maps):
            output_shape = self._output_shapes[fm].replace(n=1)
            output_feature_map = np.zeros(shape=output_shape)
            output_feature_map[:, points[fm].x, points[fm].y, 0] = intensity
            output_feature_maps.append(output_feature_map)

        return self._gradient_function(
            output_feature_maps, np.zeros(shape=input_shape)
        )


class TFFeatureMapsReceptiveField(TFReceptiveField):
    def __init__(self, model_func: Callable[[tf.Tensor], List[tf.Tensor]]):
        """
        :param model_func: model creation function. Function which
            accepts image_tensor as an input and returns list of tensors
            which correspond to selected feature maps.
        """
        self._session = None
        super().__init__(model_func)

    def _prepare_gradient_func(
        self, input_shape: ImageShape
    ) -> Tuple[Callable, GridShape, List[GridShape]]:

        if self._session is not None:
            tf.reset_default_graph()
            self._session.close()

        with tf.Graph().as_default() as graph:
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):

                input_tensor = tf.placeholder(
                    tf.float32, shape=[1, *input_shape], name="input_image"
                )
                input_shape = _get_tensor_shape(input_tensor)
                feature_maps = self._model_func(input_tensor)

                grads = []
                receptive_field_masks = []
                output_shapes = []

                for output_tensor in feature_maps:
                    # shapes
                    output_shape = _get_tensor_shape(output_tensor)
                    output_shape = (1, output_shape[1], output_shape[2], 1)
                    output_shapes.append(output_shape)

                    # define loss function
                    receptive_field_mask = tf.placeholder(
                        tf.float32, shape=output_shape, name="grid"
                    )
                    grad = _define_fm_gradient(
                        input_tensor, output_tensor, receptive_field_mask
                    )
                    grads.append(grad)
                    receptive_field_masks.append(receptive_field_mask)

            _logger.info(f"Feature maps shape: {output_shapes}")
            _logger.info(f"Input shape       : {input_shape}")

            self._session = tf.Session(graph=graph)
            self._session.run(tf.global_variables_initializer())

            def gradient_fn(
                fm_masks: List[np.ndarray], input_image: np.ndarray
            ) -> List[np.ndarray]:
                fetch_dict = {
                    mask_t: mask_np
                    for mask_t, mask_np in zip(receptive_field_masks, fm_masks)
                }
                fetch_dict[input_tensor] = input_image
                return self._session.run(grads, feed_dict=fetch_dict)

        return (
            gradient_fn,
            GridShape(*input_shape),
            [GridShape(*output_shape) for output_shape in output_shapes],
        )

    def _build_gradient_func(
            self,
            input_shape: ImageShape,
            input_layer: str = None,
            output_layers: List[str] = None
    ) -> None:
        """
        Creates gradient function and some additional parameters.

        :param input_shape: shape of the input image.
        :param input_layer: not used in this implementation
        :param output_layers: not used in this implementation
        """
        gradient_function, input_shape, output_shapes = \
            self._prepare_gradient_func(input_shape=input_shape)

        self._built = True
        self._gradient_function = gradient_function
        self._input_shape = input_shape
        self._output_shapes = output_shapes

    def compute(self, input_shape: ImageShape) -> List[FeatureMapDescription]:
        """
        Compute ReceptiveFieldDescription of given model for image of
        shape input_shape [W, H, C]. If receptive field of the network
        is bigger thant input_shape this method will raise exception.
        In order to solve with problem try to increase input_shape.

        :param input_shape: shape of the input image e.g. (224, 224, 3)
        :return a list of estimated FeatureMapDescription for each feature
            map.
        """

        return super().compute(input_shape, '', [])