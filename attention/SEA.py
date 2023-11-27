from keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Add, Lambda, Activation, Multiply, Dense

# Squeeze-and-Excitation Networks
# Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, 2019

def SEA_block(inputs, channel=512, reduction=4):
    
    b, h, w, c = inputs.shape.as_list()

    SE_out = tf.keras.layers.AveragePooling2D(pool_size=(w, h))(inputs)
    SE_out = tf.reshape(SE_out, [-1,c])
    
    SE_out = tf.keras.Sequential([
            Dense(units=c // reduction, use_bias=False),
            ReLU(),
            Dense(units=c, use_bias=False),
            Activation('sigmoid')
        ])(SE_out)
    
    SE_out = tf.reshape(SE_out, [-1,1,1,c])
    PSA_out = tf.multiply(inputs, SE_out)
    
    return PSA_out