from tensorflow import keras
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

def build_lenet_5(input_shape, num_classes=10):
    inputs = keras.Input(shape=input_shape)

    conv1 = Conv2D(kernel_size=5,
                   strides=1,
                   activation='tanh',
                   filters=6,
                   padding='same')(inputs)
    pool2 = AveragePooling2D(pool_size=2,
                             strides=2,
                             padding='valid')(conv1)
    conv3 = Conv2D(kernel_size=5,
                   strides=1,
                   activation='tanh',
                   filters=16,
                   padding='valid')(pool2)
    pool4 = AveragePooling2D(pool_size=2,
                             strides=2,
                             padding='valid')(conv3)
    conv5 = Conv2D(kernel_size=5,
                   strides=1,
                   activation='tanh',
                   filters=120,
                   padding='valid')(pool4)

    flat = Flatten()(conv5)

    dense6 = Dense(units=84,
                   activation='tanh')(flat)

    outputs = Dense(units=num_classes,
                    activation='softmax')(dense6)

    model = keras.Model(inputs=inputs, outputs=outputs, name='LeNet_5')

    return model

def lr_schedule(epoch):
    if epoch <= 2:
        lr = 5e-4
    elif epoch > 2 and epoch <= 5:
        lr = 2e-4
    elif epoch >5 and epoch <= 9:
        lr = 5e-5
    else:
        lr = 1e-5
    return lr

if __name__=='__main__':
    model = build_lenet_5(input_shape=(28,28,1))
    model.summary()