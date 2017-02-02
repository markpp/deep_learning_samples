# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

def vgg_std16_model(img_rows, img_cols, channel=1, num_class=None):
    """
    VGG 16 Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_class - number of class labels for our classification task
    """
    feature_layers = [
        ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)),
        Convolution2D(64, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Convolution2D(128, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(128, 3, 3, activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Convolution2D(256, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(256, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(256, 3, 3, activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Convolution2D(512, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(512, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(512, 3, 3, activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Convolution2D(512, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(512, 3, 3, activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(512, 3, 3, activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2))
    ]
    classification_layers = [
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1000, activation='softmax')
    ]

    model = Sequential()
    for l in feature_layers + classification_layers:
        model.add(l)
    for l in feature_layers:
        l.trainable = False
    # Loads ImageNet pre-trained data
    model.load_weights('C:/Users/maph/Documents/github/models/full/vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_class, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    # Fine-tune Example
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_class = 2
    batch_size = 32
    nb_epoch = 30

    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    # Load our model
    model = vgg_std16_model(img_rows, img_cols, channel, num_class)

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
    '''
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'C:/Users/maph/Documents/github/black_border_224/training',
            target_size=(img_rows, img_cols),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            'C:/Users/maph/Documents/github/black_border_224/validation',
            target_size=(img_rows, img_cols),
            batch_size=32,
            class_mode='categorical')
    # Start Fine-tuning
    model.fit(train_generator,
              samples_per_epoch=32*8*4,
              nb_epoch=nb_epoch,
              validation_data=validation_generator,
              nb_val_samples=32*3*4
              )
    '''
