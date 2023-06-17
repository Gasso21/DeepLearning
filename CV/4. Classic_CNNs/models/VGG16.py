from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


def build_vgg16(input_shape, num_classes=1000):
    inputs = keras.Input(shape=input_shape)

    x = Conv2D(kernel_size=3, strides=1, filters=64, activation='relu', padding='same')(inputs)
    x = Conv2D(kernel_size=3, strides=1, filters=64, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Conv2D(kernel_size=3, strides=1, filters=128, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=128, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Conv2D(kernel_size=3, strides=1, filters=256, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=256, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=256, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Conv2D(kernel_size=3, strides=1, filters=512, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=512, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=512, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Conv2D(kernel_size=3, strides=1, filters=512, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=512, activation='relu', padding='same')(x)
    x = Conv2D(kernel_size=3, strides=1, filters=512, activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='VGG16')

    return model

if __name__=='__main__':
    model = build_vgg16((227, 227, 3))
    model.summary()