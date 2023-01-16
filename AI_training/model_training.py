import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

resources = []
labels = []
image_path = "all_letters"

# imports image list
imgs = paths.list_images(image_path)

for file in imgs:
    label = file.split(os.path.sep)[-2]
    # process the image
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = resize_to_fit(img, 20, 20)

    img = np.expand_dims(img, axis=2)

    labels.append(label)
    resources.append(img)

resources = np.array(resources, dtype="float") / 255
labels = np.array(labels)

# test data (20%) and train data (80%)
(X_train, X_test, Y_train, Y_test) = train_test_split(resources, labels, test_size=0.20, random_state=0)

# converts with one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# saves the label binarizer in a file using pickle
with open('labels_model.dat', 'wb') as pickle_file:
    pickle.dump(lb, pickle_file)

# creates and trains IA
model = Sequential()

# creates the neural network layers
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 2nd layer
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 3rd layer
model.add(Flatten())
model.add(Dense(500, activation="relu"))
# out layer
model.add(Dense(26, activation="softmax"))
# compiles all layers
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# trains AI
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=10, verbose=1)

# saves the model in a file hdf5
model.save("trained_model.hdf5")
