import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Lambda
from tensorflow import floor
import tensorflow as tf
import numpy as np

#def bit_check(x):
#	if (floor(x*256) >= 64) and (floor(x*256) <= 128):
#		return -63/256
#	elif (floor(x*256) >= 128) and (floor(x*256) <= 192):
#		return -193/256
#	elif (floor(x*256) >= -128) and (floor(x*256) <= -64):
#		return -63/256
#	elif (floor(x*256) <= -128) and (floor(x*256) >= -192):
#		return -193/256
#	else:
#		return x
def bit_check(x):
	#condition_1 = keras.backend.less_equal(tf.floor(x*256),254)
	#condition_2 = keras.backend.greater_equal(tf.floor(x*256),2)
	#condition = tf.reduce_all([condition_1,condition_2],0)
	#tmp = keras.backend.ones_like(x)
	#tmp = tmp *1 / 256
	#result = tf.where(condition,tmp,x)
	#condition = keras.backend.less(tf.floor(result*256),1)
	#tmp = keras.backend.zeros_like(result)
	#result = tf.where(condition,tmp,result)
	condition = keras.backend.less(tf.floor(x*65536),1)
	tmp = tf.floor(x*256) / 256
	#tmp = x
	#tmp = keras.backend.zeros_like(x)
	result = tf.where(condition,tmp,x)
	#condition_1 = keras.backend.less_equal(tf.floor(result*256),128)
	#condition_2 = keras.backend.greater_equal(tf.floor(result*256),128)
	#condition = tf.reduce_all([condition_1,condition_2],0)
	#tmp = keras.backend.ones_like(result)
	#tmp = tmp * 127 / 256
	#result = tf.where(condition,tmp,result)
	
	return result

path='C:/Users/PaUlGuO/Downloads/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 256
x_test = x_test / 256

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Lambda(bit_check, input_shape=(28, 28, 1)))
model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Lambda(bit_check))
model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Lambda(bit_check))
model.add(Dense(120, activation='tanh'))
model.add(Lambda(bit_check))
model.add(Dense(84, activation='tanh'))
model.add(Lambda(bit_check))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.SGD(momentum=0.6), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.1, shuffle = True)

score = model.evaluate(x_test, y_test)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])