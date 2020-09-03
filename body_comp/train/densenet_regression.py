# The implementation in this file was adapted from the following:
# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
# Copyright Thibault de Boissiere, distributed under the MIT License
# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/LICENSE.md
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Conv2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D,
    Input, Concatenate, BatchNormalization
)
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def conv_factory(x, nb_filter, dropout_rate=None, bottleneck=None, activation_type='relu',
                 batch_norm=True, initializer='glorot_uniform'):

    if K.image_data_format() == "channels_first":
        norm_axis = 1
    elif K.image_data_format() == "channels_last":
        norm_axis = 3

    if bottleneck:
        if batch_norm:
            x = BatchNormalization(axis=norm_axis)(x)
        x = Activation(activation_type)(x)
        x = Conv2D(4*nb_filter, (1, 1),
                padding="same",
                kernel_initializer=initializer,
                bias_initializer=initializer)(x)

    if batch_norm:
        x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation_type)(x)
    x = Conv2D(nb_filter, (3, 3),
            padding="same",
            kernel_initializer=initializer,
            bias_initializer=initializer)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None, compression_rate=1.0, activation_type='relu',
               initializer='glorot_uniform', batch_norm=True):

    if K.image_data_format() == "channels_first":
        norm_axis = 1
    elif K.image_data_format() == "channels_last":
        norm_axis = 3

    if batch_norm:
        x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation_type)(x)

    nb_filter = int(compression_rate * nb_filter)

    x = Conv2D(nb_filter, (1, 1),
          padding="same",
          kernel_initializer=initializer,
          bias_initializer=initializer)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x, nb_filter


def denseblock(x, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, activation_type='relu',
               initializer='glorot_uniform',
               batch_norm=True):

    output_feats = []

    if K.image_data_format() == "channels_first":
        concat_axis = 1
    elif K.image_data_format() == "channels_last":
        concat_axis = -1

    for i in range(nb_layers):
        if i == 0:
            dense_in = x
        else:
            dense_in = Concatenate(axis=concat_axis)([x] + output_feats)

        new_feats = conv_factory(dense_in, growth_rate, dropout_rate, bottleneck=True, activation_type=activation_type,
                                 initializer=initializer, batch_norm=batch_norm)
        output_feats.append(new_feats)
        nb_filter += growth_rate

    if len(output_feats) == 1:
        outputs = output_feats[0]
    else:
        outputs = Concatenate(axis=concat_axis)(output_feats)

    return outputs, nb_filter


def DenseNet(img_dim, nb_layers_per_block, nb_dense_block, growth_rate,
             nb_initial_filters, dropout_rate=None, compression_rate=1.0, activation_type='relu',
             sigmoid_output_activation=False, initializer='glorot_uniform', batch_norm=True,
             output_dimension=1):

    model_input = Input(shape=img_dim)
    nb_filter = nb_initial_filters

    # Initial convolution and pooling
    x = Conv2D(nb_filter, (7, 7),
                name="initial_conv2D",
                padding="same",
                strides=(2, 2),
                kernel_initializer=initializer,
                bias_initializer=initializer)(model_input)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Add dense blocks
    for _ in range(nb_dense_block - 1):
        x, nb_filter = denseblock(
            x,
            nb_layers_per_block,
            nb_filter,
            growth_rate,
            dropout_rate=dropout_rate,
            initializer=initializer,
            batch_norm=batch_norm
        )

        # add transition
        x, nb_filter = transition(
            x,
            nb_filter,
            dropout_rate=dropout_rate,
            compression_rate=compression_rate,
            activation_type=activation_type,
            initializer=initializer,
            batch_norm=batch_norm
        )

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(
        x,
        nb_layers_per_block,
        nb_filter,
        growth_rate,
        dropout_rate=dropout_rate,
        activation_type=activation_type,
        initializer=initializer,
        batch_norm=batch_norm
    )

    if K.image_data_format() == "channels_first":
      norm_axis = 1
    elif K.image_data_format() == "channels_last":
      norm_axis = 3

    if batch_norm:
        x = BatchNormalization(axis=norm_axis)(x)
    x = Activation(activation_type)(x)
    x = GlobalAveragePooling2D()(x)

    # Regression output
    output_activation = 'sigmoid' if sigmoid_output_activation else 'linear'
    x = Dense(output_dimension)(x)
    x = Activation(output_activation)(x)

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNetRegression")

    return densenet
