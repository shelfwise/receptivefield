import pytest
from numpy.testing import assert_allclose
from keras.layers import Conv2D, Input, AvgPool2D
from keras.models import Model

from receptivefield.image import get_default_image
from receptivefield.keras import KerasReceptiveField
from receptivefield.types import ImageShape


def get_build_func(padding='same', activation='linear'):
    def model_build_func(input_shape):
        inp = Input(shape=input_shape, name='input_image')
        x = Conv2D(32, (5, 5), padding=padding, activation=activation)(inp)
        x = Conv2D(32, (3, 3), padding=padding, activation=activation)(x)
        x = AvgPool2D()(x)
        x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
        x = Conv2D(64, (3, 3), activation=activation, padding=padding, name='conv')(x)
        model = Model(inp, x)
        return model

    return model_build_func


def get_test_image(shape=(64, 64), tile_factor=0):
    image = get_default_image(shape=shape, tile_factor=tile_factor)
    return image


def test_same():
    image = get_test_image(tile_factor=0)
    rf = KerasReceptiveField(get_build_func(padding='same'), init_weights=True)
    rf_params0 = rf.compute(input_shape=ImageShape(*image.shape),
                            input_layer='input_image',
                            output_layer='conv')

    print(rf_params0)

    image = get_test_image(tile_factor=1)
    rf = KerasReceptiveField(get_build_func(padding='same'), init_weights=True)
    rf_params1 = rf.compute(input_shape=ImageShape(*image.shape),
                            input_layer='input_image',
                            output_layer='conv')

    print(rf_params1)
    assert_allclose(rf_params0, rf_params1)


def test_valid():
    image = get_test_image(tile_factor=0)
    rf = KerasReceptiveField(get_build_func(padding='valid'), init_weights=True)
    rf_params0 = rf.compute(input_shape=ImageShape(*image.shape),
                            input_layer='input_image',
                            output_layer='conv')

    print(rf_params0)

    image = get_test_image(tile_factor=1)
    rf = KerasReceptiveField(get_build_func(padding='valid'), init_weights=True)
    rf_params1 = rf.compute(input_shape=ImageShape(*image.shape),
                            input_layer='input_image',
                            output_layer='conv')

    print(rf_params1)
    assert_allclose(rf_params0, rf_params1)


if __name__ == '__main__':
    pytest.main([__file__])
