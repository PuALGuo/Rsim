import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Lambda
from tensorflow import floor

import numpy as np

def high_check(x):
	if (floor(x*16) ==  8) or (floor(x*16) == - 8):
		return x - 5 * 0.0625
	elif (floor(x*16) ==  7) or (floor(x*16) == - 7):
		return x - 4 * 0.0625
	elif (floor(x*16) ==  6) or (floor(x*16) == - 6):
		return x - 3 * 0.0625
	elif (floor(x*16) ==  5) or (floor(x*16) == - 5):
		return x - 2 * 0.0625
	elif (floor(x*16) ==  4) or (floor(x*16) == - 4):
		return x - 1 * 0.0625
	elif (floor(x*16) ==  9) or (floor(x*16) == - 9):
		return x + 4 * 0.0625
	elif (floor(x*16) == 10) or (floor(x*16) == -10):
		return x + 3 * 0.0625
	elif (floor(x*16) == 11) or (floor(x*16) == -11):
		return x + 2 * 0.0625
	elif (floor(x*16) == 12) or (floor(x*16) == -12):
		return x + 1 * 0.0625
	else:
		return x

def low_check(x):
	if (floor(x*256%16) ==  8) or (floor(x*256%16) == - 8):
		return x - 5 * 0.00390625
	elif (floor(x*256%16) ==  7) or (floor(x*256%16) == - 7):
		return x - 4 * 0.00390625
	elif (floor(x*256%16) ==  6) or (floor(x*256%16) == - 6):
		return x - 3 * 0.00390625
	elif (floor(x*256%16) ==  5) or (floor(x*256%16) == - 5):
		return x - 2 * 0.00390625
	elif (floor(x*256%16) ==  4) or (floor(x*256%16) == - 4):
		return x - 1 * 0.00390625
	elif (floor(x*256%16) ==  9) or (floor(x*256%16) == - 9):
		return x + 4 * 0.00390625
	elif (floor(x*256%16) == 10) or (floor(x*256%16) == -10):
		return x + 3 * 0.00390625
	elif (floor(x*256%16) == 11) or (floor(x*256%16) == -11):
		return x + 2 * 0.00390625
	elif (floor(x*256%16) == 12) or (floor(x*256%16) == -12):
		return x + 1 * 0.00390625
	else:
		return x
path='C:/Users/PaUlGuO/Downloads/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

#model.add(Lambda(high_check, input_shape=(28, 28, 1)))
model.add(Lambda(low_check, input_shape=(28, 28, 1)))
model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Lambda(high_check))
model.add(Lambda(low_check))
model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#model.add(Lambda(high_check))
model.add(Lambda(low_check))
model.add(Dense(120, activation='tanh'))

#model.add(Lambda(high_check))
model.add(Lambda(low_check))
model.add(Dense(84, activation='tanh'))

#model.add(Lambda(high_check))
model.add(Lambda(low_check))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.1, shuffle = True)

score = model.evaluate(x_test, y_test)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])