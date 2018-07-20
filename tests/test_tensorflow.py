import pytest
import tensorflow as tf
from numpy.testing import assert_allclose
from tensorflow.contrib import slim

from receptivefield.image import get_default_image
from receptivefield.tensorflow import TFReceptiveField, TFFeatureMapsReceptiveField
from receptivefield.types import ImageShape


def linear(x):
    return x


def model_build_func(input_shape=None):
    if len(input_shape) == 3:
        input_shape = [1, *input_shape]
    # input_image
    input_image = tf.placeholder(
        tf.float32, shape=input_shape, name="input_image"
    )

    with slim.arg_scope(
        [slim.conv2d],
        activation_fn=linear,
        weights_initializer=tf.constant_initializer(0.1),
        biases_initializer=tf.constant_initializer(0.0),
    ):
        net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], scope="conv1")
        net = slim.avg_pool2d(net, [2, 2], scope="pool1")
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
        net = tf.identity(net, name="feature_map0")
        net = slim.avg_pool2d(net, [2, 2], scope="pool4")
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
        net = tf.identity(net, name="feature_map1")

    return net


def get_test_image(shape=(64, 64), tile_factor=0):
    image = get_default_image(shape=shape, tile_factor=tile_factor)
    return image


def test_tensorflow():
    image = get_test_image(tile_factor=0)
    rf = TFReceptiveField(model_build_func)
    rf_params0 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["feature_map0"],
    )[0]

    image = get_test_image(tile_factor=1)
    rf = TFReceptiveField(model_build_func)
    rf_params1 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["feature_map0"],
    )[0]

    assert_allclose(rf_params0.rf, rf_params1.rf)


def test_multiple_feature_maps():
    image = get_test_image(tile_factor=0)
    rf = TFReceptiveField(model_build_func)

    rf_params = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["feature_map0", "feature_map1"],
    )

    rfs = 3 + 2 + 1 + 2 * 2 + 2 * 2
    assert_allclose(rf_params[0].rf.size, (rfs, rfs))
    assert_allclose(rf_params[0].rf.stride, (2, 2))

    rfs = 3 + 2 + 1 + 2 * 2 + 2 * 2 + 1 * 2 + 2 * 4 + 2 * 4 + 2 * 4
    assert_allclose(rf_params[1].rf.size, (rfs, rfs))
    assert_allclose(rf_params[1].rf.stride, (4, 4))


def model_fm_build_func(input_image: tf.Tensor):

    with slim.arg_scope(
        [slim.conv2d],
        activation_fn=linear,
        weights_initializer=tf.constant_initializer(0.1),
        biases_initializer=tf.constant_initializer(0.0),
    ):
        net = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], scope="conv1")
        net = slim.avg_pool2d(net, [2, 2], scope="pool1")
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
        fm0 = net
        net = slim.avg_pool2d(net, [2, 2], scope="pool4")
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
        fm1 = net
    return [fm0, fm1]


def test_multiple_feature_maps_secondary_api():
    image = get_test_image(tile_factor=0)
    rf = TFFeatureMapsReceptiveField(model_fm_build_func)

    rf_params = rf.compute(input_shape=ImageShape(*image.shape))

    rfs = 3 + 2 + 1 + 2 * 2 + 2 * 2
    assert_allclose(rf_params[0].rf.size, (rfs, rfs))
    assert_allclose(rf_params[0].rf.stride, (2, 2))

    rfs = 3 + 2 + 1 + 2 * 2 + 2 * 2 + 1 * 2 + 2 * 4 + 2 * 4 + 2 * 4
    assert_allclose(rf_params[1].rf.size, (rfs, rfs))
    assert_allclose(rf_params[1].rf.stride, (4, 4))


def test_compare_apis():
    image = get_test_image(tile_factor=0)
    rf_api0 = TFFeatureMapsReceptiveField(model_fm_build_func)
    rf_params_api0 = rf_api0.compute(input_shape=ImageShape(*image.shape))

    rf_api1 = TFReceptiveField(model_build_func)
    rf_params_api1 = rf_api1.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["feature_map0", "feature_map1"],
    )
    for api0, api1 in zip(rf_params_api0, rf_params_api1):
        assert_allclose(api0.rf, api1.rf)
        assert_allclose(api0.size, api1.size)


if __name__ == "__main__":
    pytest.main([__file__])
