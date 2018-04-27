import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical

# Generate dummy data
import numpy as np
import pandas

data_set = pandas.read_csv('data_set.csv', index_col=False)
data_set = np.array(data_set)
number_of_rows, number_of_cols = data_set.shape
x_train = data_set[:, :number_of_cols - 1]
y_train = data_set[:, number_of_cols - 1]

genre = ''
classi = 0
for i in range(len(y_train)):
	if genre == '':
		genre = y_train[i]
		# y_train[i] = classi
	else:
		if not (genre == y_train[i]):
			classi += 1
			genre = y_train[i]
	y_train[i] = classi
y_train = to_categorical(y_train)

# print(x_train)
# print(y_train)

data_set = pandas.read_csv('test_set.csv', index_col=False)
data_set = np.array(data_set)
number_of_rows, number_of_cols = data_set.shape
x_test = data_set[:, :number_of_cols - 1]
y_test = data_set[:, number_of_cols - 1]

genre = ''
classi = 0
for i in range(len(y_test)):
	if genre == '':
		genre = y_test[i]
		# y_train[i] = classi
	else:
		if not (genre == y_test[i]):
			classi += 1
			genre = y_test[i]
	y_test[i] = classi
y_test = to_categorical(y_test)

print(y_test)

# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 36-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=36))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,validation_data=(x_test, y_test),
          epochs=200)