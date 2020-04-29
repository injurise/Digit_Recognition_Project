#note I have made the validation set very small. I think that means it trains on more examples but 
# we get a less accurate accuracy estimate. However, this might be better to score higher scores on kaggel. 
# I am not sure about this however. 
# Also changed epochs (they were at 2). Might mean closer fit to the data.Too many might lead to overfitting
# I now try to add droput layers to prevent overfitting and use it to be able to increase the n of epochs. Works pretty well

# I made the model very deep right now, just to try it. What is kind of funny and interesting:
# In order to do this I had to decrease the Kernel size otherwise it wouldn't let me add more layers
# I think the reason for this is that each conv and pooling layer decreases the dimensions and if the image file
# is quite small (pixel size) you might have reduced it to the limit. (Just a guess though)
#or maybe they don't converge like this, as in the article I send you

#Network also works perfectly fine if you just use the first 2 "layer-sets"
#You can then increase the Kernel size to 3,3 again. My accuracy for that was 0.987 which seemed decent

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D



img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

def test_data_prep(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:,:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    test_x = x_shaped_array / 255
    return test_x
    

train_file = "dataset/train.csv"
raw_data = np.loadtxt(train_file, skiprows=1, delimiter=',')
test_file = "dataset/test.csv"
test_data = pd.read_csv(test_file)
raw_data = pd.read_csv(train_file)
x_test = test_data_prep(test_data)

x, y = data_prep(raw_data)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.5))

model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))

model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(20, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])



model.fit(x, y,
          batch_size=128,
          epochs=6,
          validation_split = 0.1)
prediction = model.predict(x_test)
final_pred = prediction.argmax(axis = 1)
submission = np.zeros((28000,2))
submission[:,1]=final_pred
for i in range(28000):
    submission[i,0]= i+1
submission = submission.astype(int)
pd.DataFrame(submission,columns=["ImageId","Label"]).to_csv("submission.csv",index = False)
    