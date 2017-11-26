import pytest
import tensorflow as tf
from numpy.testing import assert_allclose
from tensorflow.contrib import slim

from receptivefield.image import get_default_image
from receptivefield.tensorflow import TFReceptiveField
from receptivefield.types import ImageShape


def linear(x):
    return x


def model_build_func(input_shape=None):
    print(input_shape)
    if len(input_shape) == 3:
        input_shape = [1, *input_shape]

    # Important - reset graph and session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    # input_image
    input_image = tf.placeholder(
        tf.float32, shape=input_shape, name='input_image'
    )

    with slim.arg_scope([slim.conv2d],
                        activation_fn=linear,
                        weights_initializer=tf.constant_initializer(0.1),
                        biases_initializer=tf.constant_initializer(0.0)):

        net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.avg_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.avg_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = tf.identity(net, name='feature_map')

    sess.run(tf.global_variables_initializer())
    return net


def get_test_image(shape=(64, 64), tile_factor=0):
    image = get_default_image(shape=shape, tile_factor=tile_factor)
    return image


def test_tensorflow():
    image = get_test_image(tile_factor=0)
    rf = TFReceptiveField(model_build_func)
    rf_params0 = rf.compute(input_shape=ImageShape(*image.shape),
                            input_layer='input_image',
                            output_layer='feature_map')

    image = get_test_image(tile_factor=1)
    rf = TFReceptiveField(model_build_func)
    rf_params1 = rf.compute(input_shape=ImageShape(*image.shape),
                            input_layer='input_image',
                            output_layer='feature_map')

    assert_allclose(rf_params0, rf_params1)


if __name__ == '__main__':
    pytest.main([__file__])