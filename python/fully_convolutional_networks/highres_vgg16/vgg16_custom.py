import os
import json
import numpy as np

with open("configs/mac_vgg16_organs.json") as config_file:
    json_config = json.load(config_file)
if json_config['gpu'] == 1:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
else:
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model

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
from keras.preprocessing.image import ImageDataGenerator

from heatmap import to_heatmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def predict(config):
    # load json and create model
    json_file = open(config['output_model_path'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config['output_weight_path'])
    print("Loaded model from disk")

    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    new_model = to_heatmap(loaded_model)

    img = image.load_img('COLOR_MAP_0_C_A.png', target_size=(350, 150))
    #im = preprocess_image_batch(['examples/3.png'], color_mode="bgr")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    out = new_model.predict(x)

    heatmap0 = out[0,0]
    heatmap1 = out[0,1]
    heatmap2 = out[0,2]
    heatmap3 = out[0,3]

    plt.imsave("heatmap0.png",heatmap0)
    plt.imsave("heatmap1.png",heatmap1)
    plt.imsave("heatmap2.png",heatmap2)
    plt.imsave("heatmap3.png",heatmap3)


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

    class_weights = {0:1/(config['n_empty']/config['n_train_samples']), 1:1/(config['n_heart']/config['n_train_samples']), 2:1/(config['n_liver']/config['n_train_samples']), 3:1/(config['n_lung']/config['n_train_samples']), 4:1/(config['n_misc']/config['n_train_samples'])}
    print("class_weights: empty : heart : liver : lung : misc")
    print(class_weights)
    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=config['n_train_samples'],
        nb_epoch=config['n_epoch'],
        validation_data=validation_generator,
        nb_val_samples=config['n_validation_samples'])
        #class_weight=class_weights)

    model.save_weights(config['output_weight_path'])

    model_json = model.to_json()
    with open(config['output_model_path'], "w") as json_file:
        json_file.write(model_json)

def load_custom_vgg16(config):
    #https://github.com/fchollet/keras/issues/4465
    #Get back the convolutional part of a VGG network trained on ImageNet
    #model_vgg16 = VGG16(weights='imagenet', include_top=True)

    #model_json = model_vgg16.to_json()
    #with open('vgg16.json', "w") as json_file:
    #    json_file.write(model_json)

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(config['n_channel'], config['n_rows'], config['n_cols']))

    #print("Number of Layers: {}".format(len(model_vgg16_conv.layers)))
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    model_json = model_vgg16_conv.to_json()
    with open('conv.json', "w") as json_file:
        json_file.write(model_json)

    model_vgg16_conv.summary()


    #alexnet = convnet('alexnet', weights_path=config['pre_weight_path'])
    #input = alexnet.input
    #img_representation = alexnet.get_layer("dense_2").output

    #classifier = Dense(4,name='dense_3')(img_representation)
    #classifier = Activation("softmax", name="softmax")(classifier)
    #model = Model(input=input,output=classifier)


    #Create your own input format (here 3x200x200)
    #input = Input(shape=(config['n_channel'], config['n_rows'], config['n_cols']),name = 'input_1')
    input = model_vgg16_conv.input

    output_vgg16_conv = model_vgg16_conv.get_layer("block5_pool").output

    #Use the generated model
    #output_vgg16_conv = model_vgg16_conv(input)

    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(5, activation='softmax', name='predictions')(x)

    #Create your own model
    my_model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()

    sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)
    my_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #Then training with your data !
    return my_model

if __name__ == '__main__':
    config_file = 'configs/mac_vgg16_organs.json'
    config = json.load(open(config_file))

    train(config, load_custom_vgg16(config))
    predict(config)
