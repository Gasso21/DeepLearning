from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

def build_alexnet(input_shape, num_classes=1000):
    inputs = keras.Input(shape=input_shape)
    # 1-layer
    x = Conv2D(kernel_size=11,
               strides=4,
               activation='relu',
               filters=96,
               padding='valid')(inputs)
    x = MaxPooling2D(pool_size=3,
                     strides=2)(x)
    first_layer = BatchNormalization()(x)
    # 2-layer
    x = Conv2D(kernel_size=5,
               strides=1,
               kernel_regularizer=keras.regularizers.L2(0.0005),
               activation='relu',
               filters=256,
               padding='same')(first_layer)
    x = MaxPooling2D(pool_size=3,
                     strides=2,
                     padding='valid')(x)
    second_layer = BatchNormalization()(x)
    # 3-layer
    x = Conv2D(kernel_size=3,
               strides=1,
               kernel_regularizer=keras.regularizers.L2(0.0005),
               activation='relu',
               filters=384,
               padding='same')(second_layer)
    third_layer = BatchNormalization()(x)
    # 4-layer
    x = Conv2D(kernel_size=3,
               strides=1,
               kernel_regularizer=keras.regularizers.L2(0.0005),
               activation='relu',
               filters=384,
               padding='same')(third_layer)
    fourth_layer = BatchNormalization()(x)
    # 5-layer
    x = Conv2D(kernel_size=3,
               strides=1,
               kernel_regularizer=keras.regularizers.L2(0.0005),
               activation='relu',
               filters=256,
               padding='same')(fourth_layer)
    x = BatchNormalization()(x)
    fifth_layer = MaxPooling2D(pool_size=3,
                               strides=2,
                               padding='valid'
                               )(x)

    x = Flatten()(fifth_layer)

    x = Dense(units=4096,
              activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=4096,
              activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(units=num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='AlexNet')

    return model

if __name__=='__main__':
    model = build_alexnet((227, 227, 3))
    model.summary()