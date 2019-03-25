import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import numpy as np
from keras.layers import Lambda
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    if epoch > 200:
        lrate = 0.0001
    return lrate
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

##z-score
#mean = np.mean(x_train,axis=(0,1,2,3))
#std = np.std(x_train,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)

x_train /= 255
x_test /= 255

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
 
weight_decay = 1e-4
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:], name = 'conv1'))
model.add(Activation('elu'))
model.add(BatchNormalization(name = 'norm1'))
model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name = 'conv2'))
model.add(Activation('elu'))
model.add(BatchNormalization(name = 'norm2'))
model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name = 'conv3'))
model.add(Activation('elu'))
model.add(BatchNormalization(name = 'norm3'))
model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name = 'conv4'))
model.add(Activation('elu'))
model.add(BatchNormalization(name = 'norm4'))
model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name = 'conv5'))
model.add(Activation('elu'))
model.add(BatchNormalization(name = 'norm5'))
model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name = 'conv6'))
model.add(Activation('elu'))
model.add(BatchNormalization(name = 'norm6'))
model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
#model.add(Dense(1024, activation='relu', name='den1'))
#model.add(Lambda(lambda x : x / (keras.backend.max(x)-keras.backend.min(x))))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='den2'))
 
model.summary()
 
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)
 
#training
batch_size = 64
model.load_weights('bvgg_4_3.h5') 
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=210,\
                    verbose=1,validation_data=(x_test,y_test),initial_epoch=200,callbacks=[LearningRateScheduler(lr_schedule)])
#save to disk
#model.save_weights('model.h5') 
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
#model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.1, shuffle = True)
model.save_weights('bvgg_4_4.h5')
#testing
score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])