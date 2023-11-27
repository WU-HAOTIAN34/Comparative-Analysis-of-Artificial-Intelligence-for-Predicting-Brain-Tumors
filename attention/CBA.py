from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Concatenate, Dropout, Dense, Lambda, Activation, Multiply, Add, AveragePooling2D
from keras import backend as K

# CBAM: Convolutional Block Attention Module
# Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon, 2018


def Spatial_attention_block(C_A, kernel_size=3):
    
    x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(C_A)
    y = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(C_A)
    
    x = Concatenate()([x, y])
    x = Activation('relu')(x)
    x = Conv2D(1, kernel_size=kernel_size, padding='same')(x)
    x = Activation('sigmoid')(x)
    
    return x


def channel_attention(inputs, reduction=8):
    
    shape = K.int_shape(inputs)
    x = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)

    se = tf.keras.Sequential([
            Conv2D(shape[3] // reduction, 1, padding='same'),
            Activation('relu'),
            Conv2D(shape[3], 1, padding='same')
        ])
        
    x = se(x)
    y = se(y)

    res = Add()([x, y])
    x = Activation('sigmoid')(res)

    return res

def CBA_block(inputs, reduction=8, kernel_size=3):

    x = channel_attention(inputs, reduction)
    x = Multiply()([x, inputs])
    y = Spatial_attention_block(x, kernel_size)
    y = Multiply()([y, x])

    return y