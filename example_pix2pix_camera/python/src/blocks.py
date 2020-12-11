from config import *
import tensorflow as tf
from tensorflow import random_normal_initializer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2DTranspose, Dropout,
    BatchNormalization, ReLU,
    Conv2D, LeakyReLU, Activation
)
from tensorflow.keras.utils import get_custom_objects


def downsample_block(filters, kernel_size, batch_norm = True, use_config_activation = False):
    '''Pix2Pix Downsample Block for building the generator
    Reference: https://arxiv.org/abs/1611.07004
    Params:
        filters     -> Number of filters in the convolution layer
        kernel_size -> Sizer of convolution kernel
        batch_norm  -> Apply BatchNormalization (Flag)
    '''
    initializer = random_normal_initializer(0.0, 0.02)
    block = Sequential()
    block.add(
        Conv2D(
            filters, kernel_size,
            strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    if batch_norm:
        block.add(BatchNormalization())
    if use_config_activation:
        block.add(Activation(ACTIVATION))
    else:
        block.add(LeakyReLU())
    return block



def upsample_block(filters, kernel_size, dropout = False, use_config_activation = False):
    '''Pix2Pix Upsample Block
    Reference: https://arxiv.org/abs/1611.07004
    Params:
        filters     -> Number of filters in the convolution layer
        kernel_size -> Sizer of convolution kernel
        dropout     -> Apply Dropout (Flag)
    '''
    initializer = random_normal_initializer(0.0, 0.02)
    block = Sequential()
    block.add(
        Conv2DTranspose(
            filters, kernel_size,
            strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    block.add(BatchNormalization())
    if dropout:
        block.add(Dropout(0.5))
    if use_config_activation:
        block.add(Activation(ACTIVATION))
    else:
        block.add(ReLU())
    return block



class Mish(Activation):
    '''
    Code Credits: https://github.com/digantamisra98/Mish
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    '''Code Credits: https://github.com/digantamisra98/Mish
    '''
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})