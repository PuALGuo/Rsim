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
import tensorflow as tf
 
def lr_schedule(epoch):
    lrate = 0.00001
    if epoch > 75:
        lrate = 0.000005
    if epoch > 100:
        lrate = 0.000003
    return lrate

def bit_check(x):
	max = tf.reduce_max(x)
	min = tf.reduce_min(x)
	result = (x-min) / (max-min)
	
	condition = keras.backend.less_equal(result,1)
	tmp1 = keras.backend.zeros_like(result)
	tmp2 = tf.floor(result*256) / 256
	result = tf.where(condition,tmp2,tmp1)
	
	condition_1 = keras.backend.less_equal(tf.floor(result*256),192)
	condition_2 = keras.backend.greater_equal(tf.floor(result*256),128)
	condition = tf.reduce_all([condition_1,condition_2],0)
	tmp = keras.backend.ones_like(result)
	tmp = tmp * 193 / 256
	condition_1 = keras.backend.less_equal(tf.floor(result*256),127)
	condition_2 = keras.backend.greater_equal(tf.floor(result*256),64)
	condition = tf.reduce_all([condition_1,condition_2],0)
	tmp = keras.backend.ones_like(result)
	tmp = tmp * 63 / 256
	result = tf.where(condition,tmp,result)
	
	result = result * (max-min) + min
	return result
	
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)


weight_decay = 1e-4
model = Sequential()
model.add(Lambda(bit_check, input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:], kernel_initializer='random_uniform', name='conv1'))
model.add(Activation('elu'))
model.add(BatchNormalization(name='norm1'))
model.add(Lambda(bit_check))
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', name='conv2'))
model.add(Activation('elu'))
model.add(BatchNormalization(name='norm2'))
model.add(Lambda(bit_check))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', name='conv3'))
model.add(Activation('elu'))
model.add(BatchNormalization(name='norm3'))
model.add(Lambda(bit_check))
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', name='conv4'))
model.add(Activation('elu'))
model.add(BatchNormalization(name='norm4'))
model.add(Lambda(bit_check))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', name='conv5'))
model.add(Activation('elu'))
model.add(BatchNormalization(name='norm5'))
model.add(Lambda(bit_check))
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', name='conv6'))
model.add(Activation('elu'))
model.add(BatchNormalization(name='norm6'))
model.add(Lambda(bit_check))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_uniform', name='den1'))
 
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
model.load_weights('model7.h5', by_name=True) 
opt_rms = keras.optimizers.rmsprop(lr=0.00001,decay=1e-9)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
#                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=4,\
#                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)],shuffle=True)
#score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
#model.save_weights('model4.h5')
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
#                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=4,\
#                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)],shuffle=True)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
#model.save_weights('model5.h5')
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
#                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=4,\
#                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)],shuffle=True)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
#model.save_weights('model6.h5')
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)],shuffle=True)
model.save_weights('model8.h5')
#save to disk
#model.save_weights('model.h5') 
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
#model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1, validation_split=0.1, shuffle = True)
#score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
#model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.1, shuffle = True, initial_epoch=2)
#testing
score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])