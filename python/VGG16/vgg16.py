from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.optimizers import SGD

import cv2
import numpy as np


class CNN:
    def __init__(self, weights_path=None):
        # Path to the pre-trained model weights.
        self.weights_path = '../../../models/full/vgg16_weights.h5'
        # Where to put the top model when trained?
        self.top_model_weights_path = 'fc_model.h5'
        # dimensions of our images.
        self.img_width, self.img_height = 224, 224

        self.train_data_dir = '../../../data/medium_vicera_dataset/training'
        self.validation_data_dir = '../../../data/medium_vicera_dataset/validation'
        self.nb_train_samples = 32*8*4
        self.nb_validation_samples = 32*3*4
        self.nb_epoch = 25
    '''
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
        self.model.add(Dense(4))
        self.model.add(Activation('softmax'))

        if weights_path:
            self.model.load_weights(weights_path)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
    '''
    def save_bottlebeck_features():
        datagen = ImageDataGenerator(rescale=1./255)

        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, self.img_width, self.img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        assert os.path.exists(self.weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(self.weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

        generator = datagen.flow_from_directory(
                self.train_data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=32,
                class_mode=None,
                shuffle=False)
        bottleneck_features_train = model.predict_generator(generator, self.nb_train_samples)
        np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

        generator = datagen.flow_from_directory(
                self.validation_data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=32,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(generator, self.nb_validation_samples)
        np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


    def train_top_model():
        train_data = np.load(open('bottleneck_features_train.npy'))
        train_labels = np.array([0] * (self.nb_train_samples / 2) + [1] * (self.nb_train_samples / 2))

        validation_data = np.load(open('bottleneck_features_validation.npy'))
        validation_labels = np.array([0] * (self.nb_validation_samples / 2) + [1] * (self.nb_validation_samples / 2))

        model = Sequential()
        model.add(Convolution2D(4096,7,7,activation="relu",name="dense_1"))
        model.add(Convolution2D(4096,1,1,activation="relu",name="dense_2"))
        model.add(Convolution2D(4,1,1,name="dense_3"))
        model.add(Softmax4D(axis=1,name="softmax"))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='mse')

        model.fit(train_data, train_labels,
                  nb_epoch=self.nb_epoch, batch_size=32,
                  validation_data=(validation_data, validation_labels))
        model.save_weights(self.top_model_weights_path)

    '''
    def train(self):
        train_data_dir = '/Users/markpp/Desktop/code/data/black_border_224/training'
        validation_data_dir = '/Users/markpp/Desktop/code/data/black_border_224/validation'
        #train_data_dir = '/Users/markpp/Desktop/code/data/medium_vicera_dataset/training'
        #validation_data_dir = '/Users/markpp/Desktop/code/data/medium_vicera_dataset/validation'
        #nb_train_samples = 1800*2
        #nb_validation_samples = 400
        nb_train_samples = 32*8
        nb_validation_samples = 32*3
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
                                                      batch_size=32)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # this is a similar generator, for validation data
        val_set = test_datagen.flow_from_directory(validation_data_dir,
                                                   target_size=(self.img_width, self.img_height),
                                                   batch_size=32)

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
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = self.model.predict(im)
        print 'predicted label: {}, probs: {}'.format(np.argmax(out), out)
    '''
