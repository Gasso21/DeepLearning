# Classic CNNs
## LeNet
Input Shape: 28x28x1(MNIST)
**Convolution Layers**
1. ConV_1 (5x5, s=1, activation='tanh', filters=  6, padding='same')
2. Pooling_2 (f=2, s=2, padding='valid')
3. ConV_3 (5x5, s=1, activation='tanh', filters= 16, padding='valid')
4. Pooling_4 (f=2, s=2, padding='valid')
5. ConV_5 (5x5, s=1, activation='tanh', filters=120, padding='valid')

**Fully Connected Layers**

6. Dense(units=84, activation='tanh')

**Outputs**

7. Dense(units=10, activation='softmax')

- **AlexNet**
### Input Shape: 227x227x3(ImageNet)
**Convolution Layers**

1. ConV (11x11, s=4, activation='relu', filters=96, padding='valid')
2. MaxPool (f=3, s=2)
3. BatchNorm
4. ConV (5x5, s=1, activation='relu', filters=256, padding='same', regularizer='l2')
5. MaxPool (f=3, s=2, padding='valid')
6. BatchNorm
7. ConV (3x3, s=1, activation='relu', filters=384, pading='same', regularizer='l2')
8. BatchNorm
9. ConV (3x3, s=1, activation='relu', filters=384, pading='same', regularizer='l2')
10. BatchNorm
11. ConV (3x3, s=1, activation='relu', filters=256, pading='same', regularizer='l2')
12. BatchNorm
13. MaxPool (f=3, s=2, padding='valid')
14. Flatten

**Fully Connected Layers**

15. Dense(units=4096, activation='relu')
16. Dropout(0.5)
17. Dense(units=4096, activation='relu')
18. Dropout(0.5)

**Outputs**

19. Dense(units=1000, activation='softmax')

- **VGG16**
- **GoogleNet**
- **ResNet**