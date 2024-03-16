from keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, MaxPooling2D, Lambda, Activation, Multiply
from tensorflow.keras import backend as K
import tensorflow as tf


def PAB(inputs, channel_size):

    b, h, w, c = inputs.shape.as_list()

    x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    x = tf.reshape(x, [-1, 1, h*w])
    x = Activation('sigmoid')(x)
    x = K.batch_dot(x, tf.reshape(inputs, [-1, h*w, channel_size]))
    x = tf.reshape(x, [-1, 1, 1, channel_size])
    
    return x

def CAB(inputs, reduction=8):
    shape = K.int_shape(inputs)

    x = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
    x = Conv2D(shape[3] // reduction, 1, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(shape[3], 1, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = Activation('sigmoid')(x)
    
    return x
    
    
def CSDA(inputs,batch_size):

    f1 = []
    f2 = []
    half_size = inputs.shape[-1] // 2

    SPC_out = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(inputs)
    

    f1 = SPC_out[0]
    f2 = SPC_out[1]

    x = Concatenate()([CAB(f1), PAB(f1, half_size)])
    x = Conv2D(half_size, 1, padding='same')(x)
    x = Activation('sigmoid')(x)
    f1 = Multiply()([x, f1])

    y = Concatenate()([CAB(f2), PAB(f2, half_size)])
    y = Conv2D(half_size, 1, padding='same')(y)
    y = Activation('sigmoid')(y)
    f2 = Multiply()([y, f2])

    outputs = Concatenate()([f1, f2])
        
    return outputs