import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

class AutoEncRec(Model):
    '''Vanilla Autoencer'''
    
    def __init__(self, input_dim, n_dims = [64, 32, 64], dropout_rate = 0.2):
        super(AutoEncRec,self).__init__()
        self.input_dim = input_dim

        self.enc_1 = Dense(n_dims[0], input_shape = (input_dim, ), activation='selu')
        self.enc_2 = Dense(n_dims[1], activation='selu')
        self.dec_1 = Dense(n_dims[2], activation='selu')
        self.dec_2 = Dense(input_dim, activation='linear')     
        self.dropout = Dropout(dropout_rate)

    def encoder(self, x):
        net = self.enc_1(x)
        net = self.enc_2(net)
        return net
    
    def decoder(self, x):
        net = self.dec_1(x)
        net = self.dec_2(net)
        return net

    def call(self, inputs):
        net = self.decoder(self.dropout(self.encoder(inputs)))
        return net

def masked_mse(mask_value):
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        # in case mask_true is 0 everywhere, the error would be nan, therefore divide by at least 1
        # this doesn't change anything as where sum(mask_true)==0, sum(masked_squared_error)==0 as well
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse
    f.__name__ = str('Masked MSE (mask_value={})'.format(mask_value))
    return f