from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, Input, Activation
from convnetskeras. convnets import convnet
from keras.preprocessing.image import ImageDataGenerator

import json


def alex_model(config):
    alexnet = convnet('alexnet', weights_path=config['pre_weight_path'])

    input = alexnet.input
    img_representation = alexnet.get_layer("dense_2").output

    #dense_3 = Dropout(0.5)(dense_2)
    #dense_3 = Dense(1000,name='dense_3')(dense_3)
    #prediction = Activation("softmax",name="softmax")(dense_3)
    classifier = Dense(4,name='dense_3')(img_representation)
    classifier = Activation("softmax", name="softmax")(classifier)
    model = Model(input=input,output=classifier)

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:14]:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=["accuracy"])

    return model

if __name__ == '__main__':
    config_file = '../../config/rog_alex_setup_organs.json'
    config = json.load(open(config_file))

    model = alex_model(config)

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

    #KerasIntegration('markpp/test1', 'f7560908f0a18d5aa14c0583bc1a2f89', model, insights = True )

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=config['n_train_samples'],
        nb_epoch=config['n_epoch'],
        validation_data=validation_generator,
        nb_val_samples=config['n_validation_samples'])

    model.save_weights(config['output_weight_path'])

    #model_json = model.to_json()
    #with open(config['output_model_path'], "w") as json_file:
    #    json_file.write(model_json)
