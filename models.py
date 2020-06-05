#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# Tensorflow Reference Implementation of used Deep Neural Models

from tensorflow.python import TruncatedNormal
from tensorflow.python.keras import Input, optimizers
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Conv1D, Activation, Add
from tensorflow.python.keras.regularizers import l2


def DC_CNN_Block_Zero(nb_filter, filter_length, dilation, l2_layer_reg):
    def block(block_input):
        residual = block_input

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                           dilation_rate=dilation,
                           activation='linear', padding='causal', use_bias=False,
                           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                              seed=42), kernel_regularizer=l2(l2_layer_reg))(
            block_input)
        selu_out = Activation('selu')(layer_out)

        skip_out = Conv1D(1, 1, activation='linear', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                             seed=42), kernel_regularizer=l2(l2_layer_reg))(selu_out)

        c1x1_out = Conv1D(1, 1, activation='linear', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                             seed=42), kernel_regularizer=l2(l2_layer_reg))(selu_out)

        block_out = Add()([residual, c1x1_out])

        return block_out, skip_out

    return block


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def block(block_input):
        residual = block_input

        k_init = TruncatedNormal(mean=0.0, stddev=0.05, seed=42)

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length, dilation_rate=dilation,
                           activation='linear', padding='causal', use_bias=False, kernel_initializer=k_init,
                           kernel_regularizer=l2(l2_layer_reg), name="ConvP_Input" + str(dilation))(block_input)

        selu_out = Activation('selu')(layer_out)
        skip_out = Conv1D(filter_length, kernel_size=1, activation='linear', use_bias=False, kernel_initializer=k_init,
                          kernel_regularizer=l2(l2_layer_reg))(selu_out)

        c1x1_out = Conv1D(filter_length, kernel_size=1, activation='linear', use_bias=False, kernel_initializer=k_init,
                          kernel_regularizer=l2(l2_layer_reg))(selu_out)

        block_out = Add()([residual, c1x1_out])

        return block_out, skip_out

    return block


def WaveNet(length):
    visible = Input(shape=(length, 1))

    l1a, l1b = DC_CNN_Block_Zero(32, 2, 1, 0.001)(visible)
    l2a, l2b = DC_CNN_Block_Zero(32, 2, 2, 0.001)(l1a)
    l3a, l3b = DC_CNN_Block_Zero(32, 2, 4, 0.001)(l2a)
    l4a, l4b = DC_CNN_Block_Zero(32, 2, 8, 0.001)(l3a)
    l5a, l5b = DC_CNN_Block_Zero(32, 2, 16, 0.001)(l4a)
    l6a, l6b = DC_CNN_Block_Zero(32, 2, 32, 0.001)(l5a)
    l7a, l7b = DC_CNN_Block_Zero(32, 2, 64, 0.001)(l6a)

    l8 = Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])

    l9 = Activation('relu')(l8)

    yhat = Conv1D(1, 1, activation='linear', use_bias=False,
                  kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                  kernel_regularizer=l2(0.001))(l9)

    model = Model(inputs=visible, outputs=yhat)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    return model


def StockNet(prices_length, nb_features, model_name="stocknet"):
    visible_price = Input(shape=[prices_length, nb_features], name="Input_P")

    l1a, l1b = DC_CNN_Block(32, 6, 1, 0.001)(visible_price)
    l2a, l2b = DC_CNN_Block(32, 6, 2, 0.001)(l1a)
    l3a, l3b = DC_CNN_Block(32, 6, 4, 0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32, 6, 8, 0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32, 6, 16, 0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32, 6, 32, 0.001)(l5a)
    l7a, l7b = DC_CNN_Block(32, 6, 64, 0.001)(l6a)
    l8a, l8b = DC_CNN_Block(32, 6, 128, 0.001)(l6a)

    l8 = Add(name="addition")([l1b, l2b, l3b, l4b, l5b, l6b, l7b, l8b])

    l9 = Activation('relu', name="last_activation")(l8)

    yhat = Conv1D(1, kernel_size=1, activation='linear', use_bias=False,
                  kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                  kernel_regularizer=l2(0.001), name="last_conv")(l9)

    model = Model(inputs=[visible_price], outputs=yhat)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'accuracy'])

    # model.summary()

    return model
