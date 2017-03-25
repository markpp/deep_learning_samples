from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten, Dropout
from convnetskeras.convnets import convnet
from keras.preprocessing.image import ImageDataGenerator
import json


def load_model(config):
    vgg = convnet('vgg_16', weights_path=config['pre_weight_path'])

    input = vgg.input
    img_representation = vgg.get_layer("flatten").output
    #print img_representation
    #dense_3 = Dropout(0.5)(dense_2)
    #dense_3 = Dense(1000,name='dense_3')(dense_3)
    #prediction = Activation("softmax",name="softmax")(dense_3)
    #classifier = Flatten(name="flatten")(img_representation)
    classifier = Dense(4096, activation="relu",name='dense_1')(img_representation)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(4096, activation="relu", name='dense_2')(classifier)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(5,name='dense_3')(classifier)
    classifier = Activation("softmax", name="softmax")(classifier)
    model = Model(input=input,output=classifier)

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    print("Number of Layers: {}".format(len(model.layers)))
    for idx, layer in enumerate(model.layers):
        if idx < 18:
            layer.trainable = False
        else:
            print("Layer {} is trainable".format(idx))
            print("input shape {}".format(layer.input_shape))
            print("output shape {}".format(layer.output_shape))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=["accuracy"])

    return model

def train(config_path):

    config = json.load(open(config_path))

    model = load_model(config)

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

    class_weights = {0:1/(config['n_empty']/config['n_train_samples']), 1:1/(config['n_heart']/config['n_train_samples']), 2:1/(config['n_liver']/config['n_train_samples']), 3:1/(config['n_lung']/config['n_train_samples']), 4:1/(config['n_misc']/config['n_train_samples'])}
    print("class_weights: heart : liver : lung : misc : empty")
    print(class_weights)
    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=config['n_train_samples'],
        nb_epoch=config['n_epoch'],
        validation_data=validation_generator,
        nb_val_samples=config['n_validation_samples'],
        class_weight=class_weights)

    model.save_weights(config['output_weight_path'])

    #model_json = model.to_json()
    #with open(config['output_model_path'], "w") as json_file:
    #    json_file.write(model_json)


if __name__ == '__main__':
    train('configs/p50_vgg_organs.json')
