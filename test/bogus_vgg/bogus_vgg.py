import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

## 生成虚拟数据
#x_train = np.random.random((100, 100, 100, 3))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
from keras.datasets import cifar10
cifar10 = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器。
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.1, shuffle = True)
score = model.evaluate(x_test, y_test)
model.save_weights('my_bogus_vgg_weight.h5')
print('Test Loss:', score[0])
print('Test accuracy:', score[1])