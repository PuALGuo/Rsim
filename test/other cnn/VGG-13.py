##coding=utf-8
#from keras.models import Sequential
#from keras.layers import Dense, Flatten, Dropout
#from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.utils import to_categorical
#from keras.optimizers import Adam
#
#import numpy as np
#seed = 7
#np.random.seed(seed)
#
#from keras.datasets import cifar10
#cifar10 = cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = cifar10
#y_train = y_train.reshape(y_train.shape[0])
#y_test = y_test.reshape(y_test.shape[0])
#
#x_train = x_train.astype("float32")
#x_test = x_test.astype("float32")
#x_train /= 255
#x_test /=255
#y_train = to_categorical(y_train, 10)
#y_test = to_categorical(y_test, 10)
#print(x_train.shape)
#
#model = Sequential()
#model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(32,32,3),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(Dense(4096,activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(4096,activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10,activation='softmax'))
#adam = Adam(lr=1e-5)
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=1, batch_size=32)



from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
seed = 7
np.random.seed(seed)
from keras.datasets import cifar10
cifar10 = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /=255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(x_train.shape)
model = Sequential()
model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu',
                 kernel_initializer='uniform', name='conv1'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv3'))
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv5'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv6'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv7'))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv8'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv9'))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform', name='conv10'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu', name='den1'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='den2'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', name='den3'))
adam = Adam(lr=1e-5)
model.load_weights('my_vgg_weight_1.h5', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)
score = model.evaluate(x_test, y_test)
model.save_weights('my_vgg_weight_2.h5')
print('Test Loss:', score[0])
print('Test accuracy:', score[1])