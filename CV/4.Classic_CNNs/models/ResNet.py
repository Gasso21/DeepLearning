import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, Flatten, Dense

def bottleneck_residual_block(X, kernel_size, filters, reduce=False, s=2):
    F1, F2, F3 = filters
    X_shortcut = X
    
    if reduce:
        X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=s)(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)
        
        X = Conv2D(filters=F1, kernel_size=1, strides=s, padding='valid')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        
    else:
        # 1-layers Conv
        X = Conv2D(filters=F1, kernel_size=1, strides=1, padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
    
    # 2-layers Conv
    X = Conv2D(filters=F2, kernel_size=kernel_size, strides=1, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    # 3-layers Conv
    X = Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    
    # Last Stage
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape, classes):
    X_input = keras.Input(shape=input_shape)
    
    # Step 1
    X = Conv2D(filters=64, kernel_size=7, strides=2, name='conv1')(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=3, strides=2)(X)
    
    # Step 2
    X = bottleneck_residual_block(X, 3, [64, 64, 256], reduce=True, s=1)
    X = bottleneck_residual_block(X, 3, [64, 64, 256])
    X = bottleneck_residual_block(X, 3, [64, 64, 256])
    
    # Step 3
    X = bottleneck_residual_block(X, 3, [128, 128, 512], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [128, 128, 512])
    X = bottleneck_residual_block(X, 3, [128, 128, 512])
    X = bottleneck_residual_block(X, 3, [128, 128, 512])
    
    # Step 4
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    
    # Step 5
    X = bottleneck_residual_block(X, 3, [512, 512, 2048], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [512, 512, 2048])
    X = bottleneck_residual_block(X, 3, [512, 512, 2048])
    
    X = AveragePooling2D((1,1))(X)
    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    model = keras.Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model