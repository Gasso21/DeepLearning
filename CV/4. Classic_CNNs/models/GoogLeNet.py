from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, concatenate, BatchNormalization, AveragePooling2D, Dropout, Dense

def inception_module(x, filters_1x1,
                     filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5,
                     filters_pool_proj, name=None):
    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    # 1x1 ConV
    conv_1x1 = Conv2D(filters=filters_1x1,
                      kernel_size=1,
                      padding='same',
                      activation='relu',
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    # 1x1 ConV + 3x3 ConV
    pre_conv_3x3 = Conv2D(filters=filters_3x3_reduce,
                          kernel_size=1,
                          padding='same',
                          activation='relu',
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters=filters_3x3,
                      kernel_size=3,
                      padding='same',
                      activation='relu',
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(pre_conv_3x3)

    # 1x1 ConV + 5x5 ConV
    pre_conv_5x5 = Conv2D(filters=filters_5x5_reduce,
                          kernel_size=1,
                          padding='same',
                          activation='relu',
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters=filters_5x5,
                      kernel_size=5,
                      padding='same',
                      activation='relu',
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(pre_conv_5x5)

    # Pool + 1x1 ConV
    pool_proj = MaxPool2D(pool_size=3,
                          strides=1,
                          padding='same')(x)
    pool_proj = Conv2D(filters=filters_pool_proj,
                       kernel_size=1,
                       padding='same',
                       activation='relu',
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

def build_GoogLeNet(input_shape, num_classes=10):
    inputs = keras.Input(shape=input_shape)
    input_tensor = keras.layers.experimental.preprocessing.Resizing(224, 224,
                                                              interpolation="bilinear",
                                                              input_shape=input_shape)(inputs)

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    # A-Part
    x = Conv2D(filters=64,
               kernel_size=7,
               padding='same',
               strides=2,
               activation='relu',
               name='conv_1_7x7/2',
               kernel_initializer=kernel_init,
               bias_initializer=bias_init)(input_tensor)
    x = MaxPool2D(pool_size=3,
                  padding='same',
                  strides=2,
                  name='max_pool_1_3x3/2')(x)
    x = BatchNormalization()(x)
    #---
    x = Conv2D(filters=64,
               kernel_size=1,
               padding='same',
               strides=1,
               activation='relu')(x)
    x = Conv2D(filters=192,
               kernel_size=3,
               padding='same',
               strides=1,
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=3,
                  padding='same',
                  strides=2)(x)

    # B-Part
    x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                         filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32,
                         name='inception_3a')
    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                         filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64,
                         name='inception_3b')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
    #---
    x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
                         filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64,
                         name='inception_4a')
    x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                         filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64,
                         name='inception_4b')
    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
                         filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64,
                         name='inception_4c')
    x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
                         filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64,
                         name='inception_4d')
    x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                         filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128,
                         name='inception_4e')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    #---
    x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                         filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128,
                         name='inception_5a')
    x = inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
                         filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128,
                         name='inception_5b')

    # C-Part (Fully Connected Layers)
    x = AveragePooling2D(pool_size=7,
                         strides=1,
                         padding='valid')(x)
    x = Dropout(0.4)(x) 

    x = keras.layers.Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='GoogLeNet')

    return model

if __name__=='__main__':
    model = build_GoogLeNet((32, 32, 3))
    model.summary()