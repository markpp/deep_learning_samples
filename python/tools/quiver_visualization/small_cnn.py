import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop

import cv2, numpy as np

from quiver_engine import server

class CNN:
    def __init__(self, weights_path=None):
        self.img_width, self.img_height = 224, 224  # dimensions of our images.
        self.model = Sequential()
        self.model.add(Convolution2D(32, 3, 3, input_shape=(3, self.img_width, self.img_height)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        if weights_path:
            self.model.load_weights(weights_path)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def start_server(self):
        server.launch(
          self.model, # a Keras Model

          # where to store temporary files generatedby quiver (e.g. image files of layers)
          temp_folder='./tmp',

          # a folder where input images are stored
          input_folder='../../data',

          # the localhost port the dashboard is to be served on
          port=5001
        )
