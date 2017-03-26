from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop

import cv2, numpy as np

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

    def train(self):
        train_data_dir = '../../../data/medium_vicera_dataset/training'
        validation_data_dir = '../../../data/medium_vicera_dataset/validation'
        nb_train_samples = 3600
        nb_validation_samples = 400
        nb_epoch = 25

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_set = train_datagen.flow_from_directory(train_data_dir,
                                                      target_size=(self.img_width, self.img_height),
                                                      batch_size=32,
                                                      class_mode='binary')  # we need binary labels

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # this is a similar generator, for validation data
        val_set = test_datagen.flow_from_directory(validation_data_dir,
                                                   target_size=(self.img_width, self.img_height),
                                                   batch_size=32,
                                                   class_mode='binary')  # we need binary labels

        self.model.fit_generator(train_set,
                                 samples_per_epoch=nb_train_samples,
                                 nb_epoch=nb_epoch,
                                 validation_data=val_set,
                                 nb_val_samples=nb_validation_samples)

        self.model.save_weights('weights/small_cnn.h5')  # always save your weights after training or during training

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)

    def predict(self, im):
        #im[:,:,0] -= 103.939
        #im[:,:,1] -= 116.779
        #im[:,:,2] -= 123.68
        #im = im/255.0
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        # Test pretrained model
        self.model.compile(optimizer=RMSprop, loss='binary_crossentropy')
        out = self.model.predict(im)
        print 'predicted label: {}'.format(out[0])
