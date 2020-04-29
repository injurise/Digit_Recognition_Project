import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU') #This prevents memory occupation of GPU
tf.config.experimental.set_memory_growth(gpus[0], True)

img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:] #start from 1 to avoid labels
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

def data_prep_test(raw):
    #out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x#, out_y

train_file = "dataset/train.csv"


#raw_data = np.loadtxt(train_file, skiprows=1, delimiter=',')


raw_data = pd.read_csv(train_file)
x, y = data_prep(raw_data)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)


test_file = "dataset/test.csv"
#raw_data_test = np.loadtxt(test_file, skiprows=1, delimiter=',')
raw_data_test = pd.read_csv(test_file)

x_test= data_prep_test(raw_data_test)


score = model.evaluate(x_test, verbose=1)

print(score)