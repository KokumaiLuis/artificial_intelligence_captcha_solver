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

imgs = paths.list_images(image_path)

for file in imgs:
    label = file.split(os.path.sep)[-2]
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = resize_to_fit(img, 20, 20)

    img = np.expand_dims(img, axis=2)

    labels.append(label)
    resources.append(img)

resources = np.array(resources, dtype="float") / 255
labels = np.array(labels)

# separação em dados de treino (75%) e dados de teste (25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(resources, labels, test_size=0.20, random_state=0)

# Converter com one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o labelbinarizer em um arquivo com o pickle
with open('labels_model.dat', 'wb') as pickle_file:
    pickle.dump(lb, pickle_file)

# criar e treinar a inteligência artificial
model = Sequential()

# criar as camadas da rede neural
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# criar a 2ª camada
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# mais uma camada
model.add(Flatten())
model.add(Dense(500, activation="relu"))
# camada de saída
model.add(Dense(26, activation="softmax"))
# compilar todas as camadas
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# treinar a inteligência artificial
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=10, verbose=1)

# salvar o modelo em um arquivo
model.save("trained_model.hdf5")
