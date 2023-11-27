from keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Add, Lambda, Activation, Multiply


def PSA_block(inputs, channel=512, reduction=4, S=4, ):
    
    b, h, w, c = inputs.shape.as_list()

    SPC_out = Lambda(lambda x: tf.split(x, num_or_size_splits=S, axis=-1))(inputs)
    SPC_out = tf.stack(SPC_out, axis=3)

    SPC_ = []
    for i in range(S):
        x = Conv2D(c//S, kernel_size=2*(i+1)+1, padding='same')(SPC_out[:, :, :, i, :])
        x = BatchNormalization()(x)
        x = ReLU()(x)
        SPC_.append(x)
    SPC_out = tf.stack(SPC_, axis=3)


    se_blocks = []
    se_out_=[]
    for i in range(S):
        input_tensor = SPC_out[:, :, :, i, :]
        se_blocks.append(tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=(w, h)),
            Conv2D(c // (S*reduction), kernel_size=1, use_bias=False),
            ReLU(),
            Conv2D(c // S, kernel_size=1, use_bias=False),
            Activation('sigmoid')
        ]))

        se_out_.append(se_blocks[i](input_tensor))
    SE_out = tf.stack(se_out_, axis=3)


    softmax_out = tf.nn.softmax(SE_out, axis=3)


    PSA_out = tf.multiply(SPC_out, softmax_out)
    PSA_out = Concatenate()([PSA_out[:,:,:,0,:], PSA_out[:,:,:,1,:], PSA_out[:,:,:,2,:], PSA_out[:,:,:,3,:]])

    return PSA_out
