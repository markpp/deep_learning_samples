from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model

from keras.optimizers import SGD

import numpy as np

def predict(config):
    # load json and create model
    json_file = open(config['output_model_path'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config['output_weight_path'])
    print("Loaded model from disk")
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


def train(config, model):
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config['train_dir'],
        target_size=(config['n_rows'], config['n_cols']),
        batch_size=config['batch_size'],
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        config['val_dir'],
        target_size=(config['n_rows'], config['n_cols']),
        batch_size=config['batch_size'],
        class_mode='categorical')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=config['n_train_samples'],
        nb_epoch=config['n_epoch'],
        validation_data=validation_generator,
        nb_val_samples=config['n_validation_samples'])

    model.save_weights(config['output_weight_path'])

    model_json = model.to_json()
    with open(config['output_model_path'], "w") as json_file:
        json_file.write(model_json)

def load_custom_vgg16(config):
    #https://github.com/fchollet/keras/issues/4465
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    model_vgg16_conv.summary()

    #Create your own input format (here 3x200x200)
    input = Input(shape=(config['n_channel'], config['n_rows'], config['n_cols']),name = 'image_input')

    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(input)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    #Create your own model 
    my_model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    my_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #Then training with your data !
    return my_model

if __name__ == '__main__':
    config_file = '../../config/rog_setup_organs.json'
    config = json.load(open(config_file))

    train(config, load_custom_vgg16(config))
    predict(config)


